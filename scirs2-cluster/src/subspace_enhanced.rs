//! Enhanced subspace clustering algorithms
//!
//! This module provides state-of-the-art subspace clustering algorithms that
//! exploit self-expression properties of data lying on a union of subspaces.
//!
//! # Algorithms
//!
//! - [`SparseSubspaceClustering`] (SSC): Finds a sparse self-representation
//!   coefficient matrix C such that X ≈ XC, then clusters via the affinity |C|+|Cᵀ|.
//! - [`LowRankSubspaceClustering`] (LRSC): Finds a low-rank self-representation
//!   using nuclear-norm regularisation.
//! - [`ThresholdingSubspace`]: Fast method combining random projections with
//!   thresholding to build the affinity graph cheaply.
//!
//! # References
//!
//! * Elhamifar, E. & Vidal, R. (2013). *Sparse Subspace Clustering: Algorithm,
//!   Theory, and Applications.* TPAMI.
//! * Liu, G. et al. (2010). *Robust Subspace Segmentation by Low-Rank
//!   Representation.* ICML.
//! * Dyer, E. L. et al. (2013). *Greedy Feature Selection for Subspace
//!   Clustering.* JMLR.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// SparseSubspaceClustering (SSC)
// ---------------------------------------------------------------------------

/// Sparse Subspace Clustering (SSC) via elastic-net self-expression.
///
/// Solves the elastic-net self-expression problem:
///
/// ```text
/// min_C   0.5 ‖X − X C‖_F²  +  λ₁ ‖C‖₁  +  0.5 λ₂ ‖C‖_F²
/// subject to  diag(C) = 0
/// ```
///
/// using the ADMM (Alternating Direction Method of Multipliers) algorithm,
/// then builds a symmetric affinity W = |C| + |Cᵀ| and applies spectral
/// clustering to obtain final labels.
///
/// The elastic-net (combined ℓ₁ + ℓ₂) regularisation improves numerical
/// stability and connectivity compared to pure ℓ₁ regularisation.
pub struct SparseSubspaceClustering {
    /// ℓ₁ regularisation parameter λ₁ (sparsity; default 0.1).
    pub lambda1: f64,
    /// ℓ₂ regularisation parameter λ₂ (ridge; default 0.01).
    pub lambda2: f64,
    /// ADMM penalty parameter ρ (default 1.0).
    pub rho: f64,
    /// Maximum ADMM iterations (default 200).
    pub max_iter: usize,
    /// Convergence tolerance on primal residual (default 1e-6).
    pub tol: f64,
    /// Gauss-Seidel inner iterations for C-update (default 20).
    pub gs_iter: usize,
}

impl Default for SparseSubspaceClustering {
    fn default() -> Self {
        Self {
            lambda1: 0.1,
            lambda2: 0.01,
            rho: 1.0,
            max_iter: 200,
            tol: 1e-6,
            gs_iter: 20,
        }
    }
}

impl SparseSubspaceClustering {
    /// Fit SSC to data matrix X (n_samples × n_features).
    ///
    /// # Arguments
    /// * `x`         – Data matrix. Columns are data points; internally the
    ///   algorithm operates on the (n_samples × n_samples) Gram matrix.
    /// * `lambda`    – Override λ₁ (if `None`, uses `self.lambda1`).
    /// * `max_iter`  – Override maximum iterations (if `None`, uses `self.max_iter`).
    /// * `n_clusters` – Number of clusters for the spectral clustering step.
    ///
    /// # Returns
    /// Cluster labels `Array1<usize>` of length `n_samples`.
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        lambda: Option<f64>,
        max_iter: Option<usize>,
        n_clusters: usize,
    ) -> Result<Array1<usize>> {
        let (n, _) = (x.shape()[0], x.shape()[1]);
        let lambda1 = lambda.unwrap_or(self.lambda1);
        let max_it = max_iter.unwrap_or(self.max_iter);

        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty data matrix".into()));
        }
        if n_clusters == 0 || n_clusters > n {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({n_clusters}) must be in [1, n_samples ({n})]"
            )));
        }
        if lambda1 < 0.0 {
            return Err(ClusteringError::InvalidInput(
                "lambda must be non-negative".into(),
            ));
        }

        // Compute Gram matrix G = X Xᵀ  (n × n).
        let gram = compute_gram(x);

        // ADMM for elastic-net self-expression.
        let c_mat = ssc_admm_elasticnet(
            &gram,
            lambda1,
            self.lambda2,
            self.rho,
            max_it,
            self.tol,
            self.gs_iter,
        )?;

        // Symmetric affinity W = |C| + |Cᵀ|.
        let affinity = symmetrise_abs(&c_mat);

        // Spectral clustering on affinity.
        let labels = spectral_on_affinity(&affinity, n_clusters)?;

        Ok(labels)
    }
}

// ---------------------------------------------------------------------------
// LowRankSubspaceClustering (LRSC)
// ---------------------------------------------------------------------------

/// Low-Rank Subspace Clustering (LRSC) via nuclear-norm regularisation.
///
/// Solves:
///
/// ```text
/// min_C   ‖C‖_*  +  0.5 λ ‖X − X C‖_F²
/// subject to  diag(C) = 0
/// ```
///
/// using the ADMM with singular-value thresholding (SVT) for the nuclear-norm
/// proximal step.  The low-rank C captures global subspace structure rather than
/// local sparse connections.
pub struct LowRankSubspaceClustering {
    /// Regularisation parameter λ (default 1.0).
    pub lambda: f64,
    /// ADMM penalty parameter ρ (default 0.5).
    pub rho: f64,
    /// Maximum iterations (default 100).
    pub max_iter: usize,
    /// Convergence tolerance (default 1e-5).
    pub tol: f64,
}

impl Default for LowRankSubspaceClustering {
    fn default() -> Self {
        Self {
            lambda: 1.0,
            rho: 0.5,
            max_iter: 100,
            tol: 1e-5,
        }
    }
}

impl LowRankSubspaceClustering {
    /// Fit LRSC.
    ///
    /// # Arguments
    /// * `x`          – Data matrix (n_samples × n_features).
    /// * `lambda`     – Override λ (if `None`, uses `self.lambda`).
    /// * `max_iter`   – Override iterations (if `None`, uses `self.max_iter`).
    /// * `n_clusters` – Number of clusters for the spectral step.
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        lambda: Option<f64>,
        max_iter: Option<usize>,
        n_clusters: usize,
    ) -> Result<Array1<usize>> {
        let n = x.shape()[0];
        let lam = lambda.unwrap_or(self.lambda);
        let max_it = max_iter.unwrap_or(self.max_iter);

        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty data matrix".into()));
        }
        if n_clusters == 0 || n_clusters > n {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({n_clusters}) must be in [1, n_samples ({n})]"
            )));
        }
        if lam < 0.0 {
            return Err(ClusteringError::InvalidInput(
                "lambda must be non-negative".into(),
            ));
        }

        let gram = compute_gram(x);
        let c_mat = lrsc_admm_svt(&gram, lam, self.rho, max_it, self.tol)?;

        let affinity = symmetrise_abs(&c_mat);
        let labels = spectral_on_affinity(&affinity, n_clusters)?;

        Ok(labels)
    }
}

// ---------------------------------------------------------------------------
// ThresholdingSubspace
// ---------------------------------------------------------------------------

/// Thresholding-based subspace clustering.
///
/// A computationally cheap alternative to SSC and LRSC that builds an affinity
/// graph by:
///
/// 1. Projecting the data onto a random Gaussian subspace of dimension `proj_dim`.
/// 2. For each data point, finding its `k_neighbors` nearest neighbours in the
///    projected space.
/// 3. Keeping only affinity edges whose (normalised) inner product in the
///    *original* space exceeds `threshold`.
/// 4. Applying spectral clustering on the resulting sparse affinity.
///
/// This approach is motivated by compressed-sensing arguments: random projections
/// approximately preserve the angular structure of the data.
pub struct ThresholdingSubspace {
    /// Fraction of the original dimensionality to use for random projections (default 0.5).
    pub proj_ratio: f64,
    /// Number of nearest neighbours to retain per point (default 5).
    pub k_neighbors: usize,
    /// Cosine similarity threshold in original space (default 0.5).
    pub threshold: f64,
    /// RNG seed for the random projection matrix.
    pub seed: u64,
}

impl Default for ThresholdingSubspace {
    fn default() -> Self {
        Self {
            proj_ratio: 0.5,
            k_neighbors: 5,
            threshold: 0.5,
            seed: 42,
        }
    }
}

impl ThresholdingSubspace {
    /// Fit thresholding subspace clustering.
    ///
    /// # Arguments
    /// * `x`          – Data matrix (n_samples × n_features).
    /// * `threshold`  – Override cosine similarity threshold (if `None`, uses `self.threshold`).
    /// * `n_clusters` – Number of clusters for spectral clustering.
    pub fn fit(
        &self,
        x: ArrayView2<f64>,
        threshold: Option<f64>,
        n_clusters: usize,
    ) -> Result<Array1<usize>> {
        let (n, d) = (x.shape()[0], x.shape()[1]);
        let thresh = threshold.unwrap_or(self.threshold);

        if n == 0 {
            return Err(ClusteringError::InvalidInput("Empty data matrix".into()));
        }
        if n_clusters == 0 || n_clusters > n {
            return Err(ClusteringError::InvalidInput(format!(
                "n_clusters ({n_clusters}) must be in [1, n_samples ({n})]"
            )));
        }
        if d == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data must have at least one feature".into(),
            ));
        }

        // Step 1: L2-normalise rows of X.
        let x_norm = l2_normalise_rows_copy(x);

        // Step 2: Random projection (Gaussian, n_features → proj_dim).
        let proj_dim = ((d as f64 * self.proj_ratio).ceil() as usize).max(1).min(d);
        let mut rng = self.seed;
        let proj_mat = random_gaussian_matrix(d, proj_dim, &mut rng);

        // Projected data: (n × proj_dim)
        let projected = mat_mul(&x_norm, &proj_mat);

        // Step 3: For each point, find k_neighbors nearest neighbours in projected space.
        let k = self.k_neighbors.min(n - 1).max(1);
        let mut affinity = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            // Compute projected distances to all other points.
            let mut dists: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let mut s = 0.0;
                    for l in 0..proj_dim {
                        let diff = projected[[i, l]] - projected[[j, l]];
                        s += diff * diff;
                    }
                    (j, s)
                })
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Retain only the top-k, filtered by original-space cosine similarity.
            let mut kept = 0usize;
            for (j, _proj_dist) in dists.iter().take(k * 4) {
                // Compute cosine similarity in original (normalised) space.
                let cos: f64 = (0..d).map(|l| x_norm[[i, l]] * x_norm[[*j, l]]).sum();
                let cos_abs = cos.abs();
                if cos_abs >= thresh {
                    affinity[[i, *j]] = cos_abs;
                    affinity[[*j, i]] = cos_abs;
                    kept += 1;
                    if kept >= k {
                        break;
                    }
                }
            }

            // Fall-back: if no neighbour survived thresholding, take the closest one.
            if kept == 0 {
                if let Some(&(j, _)) = dists.first() {
                    let cos: f64 = (0..d).map(|l| x_norm[[i, l]] * x_norm[[j, l]]).sum();
                    affinity[[i, j]] = cos.abs().max(1e-10);
                    affinity[[j, i]] = affinity[[i, j]];
                }
            }
        }

        let labels = spectral_on_affinity(&affinity, n_clusters)?;
        Ok(labels)
    }
}

// ---------------------------------------------------------------------------
// Shared internal helpers
// ---------------------------------------------------------------------------

/// Compute Gram matrix G = X Xᵀ  (n × n).
fn compute_gram(x: ArrayView2<f64>) -> Array2<f64> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let mut g = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = 0.0;
            for l in 0..d {
                s += x[[i, l]] * x[[j, l]];
            }
            g[[i, j]] = s;
            g[[j, i]] = s;
        }
    }
    g
}

/// ADMM solver for elastic-net self-expression:
///
/// ```text
/// min_C  0.5 ‖X − XC‖_F²  +  λ₁ ‖C‖₁  +  0.5 λ₂ ‖C‖_F²   s.t.  diag(C)=0
/// ```
///
/// Uses the Gram matrix G = XᵀX to avoid redundant matrix multiplications.
/// The C-update is solved column-wise using Gauss-Seidel iterations applied
/// to the linear system  (G + (ρ + λ₂) I) c_j = G_{·j} + ρ (z_j − u_j).
fn ssc_admm_elasticnet(
    gram: &Array2<f64>,
    lambda1: f64,
    lambda2: f64,
    rho: f64,
    max_iter: usize,
    tol: f64,
    gs_iter: usize,
) -> Result<Array2<f64>> {
    let n = gram.shape()[0];

    let mut c_mat = Array2::<f64>::zeros((n, n));
    let mut z_mat = Array2::<f64>::zeros((n, n));
    let mut u_mat = Array2::<f64>::zeros((n, n));

    // Build system matrix A = G + (ρ + λ₂) I
    let mut a_mat = gram.clone();
    let diag_add = rho + lambda2;
    for i in 0..n {
        a_mat[[i, i]] += diag_add;
    }

    let thresh = lambda1 / rho;

    for _iter in 0..max_iter {
        // C-update: for each column j solve A c_j = G_{·j} + ρ (z_j − u_j)
        for j in 0..n {
            let mut rhs = Array1::<f64>::zeros(n);
            for i in 0..n {
                rhs[i] = gram[[i, j]] + rho * (z_mat[[i, j]] - u_mat[[i, j]]);
            }

            // Gauss-Seidel inner solve.
            let mut x_col = Array1::<f64>::zeros(n);
            for _gs in 0..gs_iter {
                for i in 0..n {
                    let mut sum = rhs[i];
                    for k in 0..n {
                        if k != i {
                            sum -= a_mat[[i, k]] * x_col[k];
                        }
                    }
                    let diag = a_mat[[i, i]];
                    if diag.abs() > f64::EPSILON {
                        x_col[i] = sum / diag;
                    }
                }
            }
            // Enforce diag(C) = 0.
            x_col[j] = 0.0;
            for i in 0..n {
                c_mat[[i, j]] = x_col[i];
            }
        }

        // Z-update: element-wise soft threshold.
        let mut primal_res = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let v = c_mat[[i, j]] + u_mat[[i, j]];
                let new_z = soft_threshold(v, thresh);
                primal_res += (c_mat[[i, j]] - new_z).powi(2);
                z_mat[[i, j]] = new_z;
            }
        }

        // U-update: dual ascent.
        for i in 0..n {
            for j in 0..n {
                u_mat[[i, j]] += c_mat[[i, j]] - z_mat[[i, j]];
            }
        }

        if primal_res.sqrt() < tol {
            break;
        }
    }

    Ok(c_mat)
}

/// ADMM with singular-value thresholding (SVT) for nuclear-norm minimisation:
///
/// ```text
/// min_C  ‖C‖_*  +  0.5 λ ‖X − XC‖_F²   s.t.  diag(C)=0
/// ```
///
/// The C-update solves  (G + ρ I) C = G + ρ (Z − U)  column-wise.
/// The Z-update applies soft-thresholding to singular values of (C + U).
fn lrsc_admm_svt(
    gram: &Array2<f64>,
    lambda: f64,
    rho: f64,
    max_iter: usize,
    tol: f64,
) -> Result<Array2<f64>> {
    let n = gram.shape()[0];

    let mut c_mat = Array2::<f64>::zeros((n, n));
    let mut z_mat = Array2::<f64>::zeros((n, n));
    let mut u_mat = Array2::<f64>::zeros((n, n));

    // System matrix for C-update: A = λ G + ρ I.
    let mut a_mat = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            a_mat[[i, j]] = lambda * gram[[i, j]];
        }
        a_mat[[i, i]] += rho;
    }

    let svt_thresh = 1.0 / rho;

    for _iter in 0..max_iter {
        // C-update: A c_j = λ G_{·j} + ρ (z_j − u_j).
        for j in 0..n {
            let mut rhs = Array1::<f64>::zeros(n);
            for i in 0..n {
                rhs[i] = lambda * gram[[i, j]] + rho * (z_mat[[i, j]] - u_mat[[i, j]]);
            }
            // Conjugate gradient for the symmetric positive-definite system.
            let mut x_col = cg_solve(&a_mat, &rhs, 30, tol * 0.1);
            x_col[j] = 0.0;
            for i in 0..n {
                c_mat[[i, j]] = x_col[i];
            }
        }

        // Z-update: singular value thresholding of (C + U).
        let cu = {
            let mut m = Array2::<f64>::zeros((n, n));
            for i in 0..n {
                for j in 0..n {
                    m[[i, j]] = c_mat[[i, j]] + u_mat[[i, j]];
                }
            }
            m
        };

        let (u_svd, s_svd, vt_svd) = compact_svd(cu.view(), n, 42)?;

        // Threshold singular values.
        let mut primal_res = 0.0f64;
        z_mat = Array2::<f64>::zeros((n, n));
        for k in 0..s_svd.len() {
            let sv = (s_svd[k] - svt_thresh).max(0.0);
            if sv > 0.0 {
                for i in 0..n {
                    for j in 0..n {
                        z_mat[[i, j]] += sv * u_svd[[i, k]] * vt_svd[[k, j]];
                    }
                }
            }
        }

        // U-update.
        for i in 0..n {
            for j in 0..n {
                let r = c_mat[[i, j]] - z_mat[[i, j]];
                u_mat[[i, j]] += r;
                primal_res += r * r;
            }
        }

        if primal_res.sqrt() < tol {
            break;
        }
    }

    Ok(c_mat)
}

/// Conjugate gradient solver for symmetric positive-definite system A x = b.
fn cg_solve(a: &Array2<f64>, b: &Array1<f64>, max_iter: usize, tol: f64) -> Array1<f64> {
    let n = b.len();
    let mut x = Array1::<f64>::zeros(n);

    // r = b - A x  (initially b since x = 0).
    let mut r = b.clone();
    let mut p = r.clone();
    let mut r_sq: f64 = r.iter().map(|v| v * v).sum();

    for _ in 0..max_iter {
        if r_sq < tol * tol {
            break;
        }
        // q = A p
        let mut q = Array1::<f64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                q[i] += a[[i, j]] * p[j];
            }
        }
        let pq: f64 = p.iter().zip(q.iter()).map(|(a, b)| a * b).sum();
        if pq.abs() < f64::EPSILON {
            break;
        }
        let alpha = r_sq / pq;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
        }
        let r_sq_new: f64 = r.iter().map(|v| v * v).sum();
        let beta = r_sq_new / r_sq.max(f64::EPSILON);
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        r_sq = r_sq_new;
    }
    x
}

/// Symmetrise absolute values: W[i,j] = |C[i,j]| + |C[j,i]|.
fn symmetrise_abs(c: &Array2<f64>) -> Array2<f64> {
    let n = c.shape()[0];
    let mut w = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            w[[i, j]] = c[[i, j]].abs() + c[[j, i]].abs();
        }
    }
    w
}

/// Spectral clustering on a precomputed affinity matrix.
///
/// Constructs the normalised graph Laplacian L_sym, extracts its k smallest
/// eigenvectors via power-iteration deflation, row-normalises the resulting
/// embedding, and applies k-means.
fn spectral_on_affinity(affinity: &Array2<f64>, k: usize) -> Result<Array1<usize>> {
    let n = affinity.shape()[0];
    if n <= k {
        // Trivial case: each point is its own cluster.
        let labels: Vec<usize> = (0..n).map(|i| i % k).collect();
        return Ok(Array1::from_vec(labels));
    }

    // Degree vector.
    let mut degree = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            degree[i] += affinity[[i, j]];
        }
    }

    // Normalised Laplacian L_sym = I − D^{-1/2} W D^{-1/2}.
    // We work with the matrix M = I − L_sym = D^{-1/2} W D^{-1/2}
    // (shift to find the k *largest* eigenvectors of M, which correspond to
    //  the k *smallest* eigenvectors of L_sym).
    let mut m = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let di = if degree[i] > f64::EPSILON {
                1.0 / degree[i].sqrt()
            } else {
                0.0
            };
            let dj = if degree[j] > f64::EPSILON {
                1.0 / degree[j].sqrt()
            } else {
                0.0
            };
            m[[i, j]] = di * affinity[[i, j]] * dj;
        }
    }

    // Extract k largest eigenvectors of M via power-iteration deflation.
    let mut embedding = Array2::<f64>::zeros((n, k));
    let mut deflated = m.clone();
    let mut seed = 0xDEADBEEF_u64;

    for kk in 0..k {
        let v = power_iteration_vec(&deflated, 150, &mut seed);
        for i in 0..n {
            embedding[[i, kk]] = v[i];
        }
        // Rayleigh quotient for deflation.
        let mut eigenval = 0.0f64;
        for i in 0..n {
            let mut av_i = 0.0f64;
            for j in 0..n {
                av_i += deflated[[i, j]] * v[j];
            }
            eigenval += v[i] * av_i;
        }
        // Deflate: A ← A − λ v vᵀ.
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] -= eigenval * v[i] * v[j];
            }
        }
    }

    // Row-normalise embedding.
    for i in 0..n {
        let norm: f64 = (0..k)
            .map(|kk| embedding[[i, kk]] * embedding[[i, kk]])
            .sum::<f64>()
            .sqrt();
        if norm > 1e-12 {
            for kk in 0..k {
                embedding[[i, kk]] /= norm;
            }
        }
    }

    // K-means on the embedding rows.
    let mut rng = 0xCAFEBABE_u64;
    let i_labels = kmeans_usize(embedding.view(), k, 100, &mut rng);
    Ok(i_labels)
}

/// Power iteration to find the dominant eigenvector of a symmetric matrix.
fn power_iteration_vec(mat: &Array2<f64>, max_iter: usize, rng: &mut u64) -> Vec<f64> {
    let n = mat.shape()[0];
    let mut v: Vec<f64> = (0..n).map(|_| lcg_f64(rng) - 0.5).collect();
    normalise_vec_f64(&mut v);

    for _ in 0..max_iter {
        let mut av = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += mat[[i, j]] * v[j];
            }
        }
        normalise_vec_f64(&mut av);
        v = av;
    }
    v
}

/// Normalise a Vec<f64> in place; no-op if near-zero.
fn normalise_vec_f64(v: &mut Vec<f64>) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// K-means returning `Array1<usize>` labels.
fn kmeans_usize(
    features: ArrayView2<f64>,
    k: usize,
    max_iter: usize,
    rng: &mut u64,
) -> Array1<usize> {
    let (n, d) = (features.shape()[0], features.shape()[1]);
    if n == 0 || k == 0 {
        return Array1::zeros(n);
    }
    let k = k.min(n);

    // Initialise centroids.
    let step = (n as f64 / k as f64).max(1.0);
    let mut centroids: Vec<Vec<f64>> = (0..k)
        .map(|ci| {
            let idx = ((ci as f64 * step) as usize).min(n - 1);
            features.row(idx).to_vec()
        })
        .collect();

    let mut labels = vec![0usize; n];

    for _iter in 0..max_iter {
        let mut changed = false;
        for i in 0..n {
            let best = (0..k)
                .map(|j| {
                    let dist: f64 = (0..d)
                        .map(|l| (features[[i, l]] - centroids[j][l]).powi(2))
                        .sum();
                    (j, dist)
                })
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            if labels[i] != best {
                labels[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; d]; k];
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for l in 0..d {
                sums[c][l] += features[[i, l]];
            }
        }
        for j in 0..k {
            if counts[j] == 0 {
                // Re-initialise to a random point.
                let ri = lcg_usize(rng, n);
                centroids[j] = features.row(ri).to_vec();
            } else {
                for l in 0..d {
                    centroids[j][l] = sums[j][l] / counts[j] as f64;
                }
            }
        }
    }

    Array1::from_vec(labels)
}

/// Soft-threshold operator: sign(x) max(|x| − t, 0).
fn soft_threshold(x: f64, t: f64) -> f64 {
    let ax = x.abs();
    if ax <= t {
        0.0
    } else if x > 0.0 {
        ax - t
    } else {
        t - ax
    }
}

/// L2-normalise rows of a matrix view into a new array.
fn l2_normalise_rows_copy(x: ArrayView2<f64>) -> Array2<f64> {
    let (n, d) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::<f64>::zeros((n, d));
    for r in 0..n {
        let norm: f64 = (0..d).map(|c| x[[r, c]] * x[[r, c]]).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for c in 0..d {
                out[[r, c]] = x[[r, c]] / norm;
            }
        }
    }
    out
}

/// Dense matrix multiplication A (m × k) × B (k × n) → C (m × n).
fn mat_mul(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            let a_il = a[[i, l]];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
    c
}

/// Generate a random Gaussian matrix (rows × cols) from a seeded LCG.
fn random_gaussian_matrix(rows: usize, cols: usize, rng: &mut u64) -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((rows, cols));
    let scale = 1.0 / (rows as f64).sqrt();
    for v in m.iter_mut() {
        let u1 = lcg_f64(rng).max(1e-15);
        let u2 = lcg_f64(rng);
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * scale;
    }
    m
}

/// LCG float in [0, 1).
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// LCG usize in [0, n).
fn lcg_usize(state: &mut u64, n: usize) -> usize {
    (lcg_f64(state) * n as f64) as usize % n
}

/// Compact randomised SVD: returns (U, s, Vt) with shapes (m,k), (k,), (k,n).
fn compact_svd(
    x: ArrayView2<f64>,
    k: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array1<f64>, Array2<f64>)> {
    let (m, n) = (x.shape()[0], x.shape()[1]);
    let k = k.min(m).min(n).max(1);
    let mut rng = seed;

    // Random Gaussian sketch Ω (n × k).
    let mut omega = Array2::<f64>::zeros((n, k));
    for v in omega.iter_mut() {
        let u1 = lcg_f64(&mut rng).max(1e-15);
        let u2 = lcg_f64(&mut rng);
        *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }

    // Y = X Ω  (m × k).
    let y = mat_mul_arr(x, omega.view());
    let q = gram_schmidt_arr(y.view())?;

    // B = Qᵀ X  (k × n).
    let b = mat_mul_arr_t(q.view(), x);

    // B Bᵀ  (k × k).
    let bbt = mat_mul_arr(
        b.view(),
        {
            let mut bt = Array2::<f64>::zeros((n, k));
            for i in 0..k {
                for j in 0..n {
                    bt[[j, i]] = b[[i, j]];
                }
            }
            bt
        }
        .view(),
    );

    let (ub, sigma) = power_iter_eig_local(bbt.view(), k, seed.wrapping_add(1))?;
    let s: Array1<f64> = sigma.mapv(|v| v.max(0.0).sqrt());

    // U = Q U_B  (m × k).
    let u = mat_mul_arr(q.view(), ub.view());

    // Vᵀ = U_Bᵀ B / σ  (k × n).
    let mut vt = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        let si = s[i];
        if si < 1e-12 {
            continue;
        }
        for j in 0..n {
            let mut val = 0.0;
            for l in 0..k {
                val += ub[[l, i]] * b[[l, j]];
            }
            vt[[i, j]] = val / si;
        }
    }

    Ok((u, s, vt))
}

/// Dense matrix multiplication for ArrayView2 operands.
fn mat_mul_arr(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let mut c = Array2::<f64>::zeros((m, n));
    for i in 0..m {
        for l in 0..k {
            let a_il = a[[i, l]];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..n {
                c[[i, j]] += a_il * b[[l, j]];
            }
        }
    }
    c
}

/// Dense matrix multiplication Aᵀ B.
fn mat_mul_arr_t(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let mut c = Array2::<f64>::zeros((k, n));
    for l in 0..k {
        for i in 0..m {
            let a_il = a[[i, l]];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..n {
                c[[l, j]] += a_il * b[[i, j]];
            }
        }
    }
    c
}

/// Gram-Schmidt orthonormalisation.
fn gram_schmidt_arr(a: ArrayView2<f64>) -> Result<Array2<f64>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut q = Array2::<f64>::zeros((m, n));
    for j in 0..n {
        let mut v: Vec<f64> = (0..m).map(|i| a[[i, j]]).collect();
        for i in 0..j {
            let dot: f64 = (0..m).map(|r| v[r] * q[[r, i]]).sum();
            for r in 0..m {
                v[r] -= dot * q[[r, i]];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            if j < m {
                q[[j, j]] = 1.0;
            }
        } else {
            for r in 0..m {
                q[[r, j]] = v[r] / norm;
            }
        }
    }
    Ok(q)
}

/// Power-iteration eigen-decomposition of a symmetric matrix (top-k).
fn power_iter_eig_local(
    a: ArrayView2<f64>,
    k: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = a.shape()[0];
    let k = k.min(n);
    let mut rng = seed;

    let mut eigvecs = Array2::<f64>::zeros((n, k));
    let mut eigvals = Array1::<f64>::zeros(k);
    let mut deflated = a.to_owned();

    for col in 0..k {
        let mut v: Vec<f64> = (0..n).map(|_| lcg_f64(&mut rng) - 0.5).collect();
        normalise_vec_f64(&mut v);

        for _ in 0..200 {
            let mut av = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    av[i] += deflated[[i, j]] * v[j];
                }
            }
            for prev in 0..col {
                let dot: f64 = (0..n).map(|i| av[i] * eigvecs[[i, prev]]).sum();
                for i in 0..n {
                    av[i] -= dot * eigvecs[[i, prev]];
                }
            }
            normalise_vec_f64(&mut av);
            v = av;
        }

        let mut av = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += deflated[[i, j]] * v[j];
            }
        }
        let eigenvalue: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
        eigvals[col] = eigenvalue;
        for i in 0..n {
            eigvecs[[i, col]] = v[i];
        }
        for i in 0..n {
            for j in 0..n {
                deflated[[i, j]] -= eigenvalue * v[i] * v[j];
            }
        }
    }

    Ok((eigvecs, eigvals))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Three linearly separable subspaces in R^6:
    /// - Subspace 0: data lives in span{e_0, e_1}
    /// - Subspace 1: data lives in span{e_2, e_3}
    /// - Subspace 2: data lives in span{e_4, e_5}
    fn three_subspace_data() -> Array2<f64> {
        let n_per = 10;
        let n = n_per * 3;
        let d = 6;
        let mut data = Array2::<f64>::zeros((n, d));
        let mut rng = 0xABCDEF_u64;

        fn lcg(s: &mut u64) -> f64 {
            *s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }

        for i in 0..n_per {
            let a = lcg(&mut rng);
            let b = lcg(&mut rng);
            data[[i, 0]] = a;
            data[[i, 1]] = b;
        }
        for i in 0..n_per {
            let a = lcg(&mut rng);
            let b = lcg(&mut rng);
            data[[n_per + i, 2]] = a;
            data[[n_per + i, 3]] = b;
        }
        for i in 0..n_per {
            let a = lcg(&mut rng);
            let b = lcg(&mut rng);
            data[[2 * n_per + i, 4]] = a;
            data[[2 * n_per + i, 5]] = b;
        }
        data
    }

    // ------------------------------------------------------------------
    // SparseSubspaceClustering
    // ------------------------------------------------------------------

    #[test]
    fn test_ssc_output_shape() {
        let x = three_subspace_data();
        let ssc = SparseSubspaceClustering::default();
        let labels = ssc.fit(x.view(), None, None, 3).expect("SSC failed");
        assert_eq!(labels.len(), 30);
    }

    #[test]
    fn test_ssc_labels_in_range() {
        let x = three_subspace_data();
        let ssc = SparseSubspaceClustering::default();
        let labels = ssc.fit(x.view(), None, None, 3).expect("SSC failed");
        for &l in labels.iter() {
            assert!(l < 3, "label {l} out of range");
        }
    }

    #[test]
    fn test_ssc_lambda_override() {
        let x = three_subspace_data();
        let ssc = SparseSubspaceClustering {
            lambda1: 0.5,
            ..Default::default()
        };
        // Override lambda and max_iter via parameters.
        let labels = ssc
            .fit(x.view(), Some(0.2), Some(50), 3)
            .expect("SSC lambda override");
        assert_eq!(labels.len(), 30);
    }

    #[test]
    fn test_ssc_invalid_input() {
        let x = three_subspace_data();
        let ssc = SparseSubspaceClustering::default();
        // n_clusters > n_samples.
        assert!(ssc.fit(x.view(), None, None, 100).is_err());
        // n_clusters == 0.
        assert!(ssc.fit(x.view(), None, None, 0).is_err());
        // Empty data.
        let empty = Array2::<f64>::zeros((0, 6));
        assert!(ssc.fit(empty.view(), None, None, 3).is_err());
        // Negative lambda.
        assert!(ssc.fit(x.view(), Some(-0.1), None, 3).is_err());
    }

    #[test]
    fn test_ssc_two_clusters() {
        let mut data = Array2::<f64>::zeros((10, 4));
        for i in 0..5 {
            data[[i, 0]] = (i as f64 + 1.0) * 0.3;
            data[[i, 1]] = (i as f64 + 1.0) * 0.2;
        }
        for i in 5..10 {
            data[[i, 2]] = (i as f64 - 4.0) * 0.3;
            data[[i, 3]] = (i as f64 - 4.0) * 0.2;
        }
        let ssc = SparseSubspaceClustering::default();
        let labels = ssc.fit(data.view(), None, None, 2).expect("SSC 2 clusters");
        assert_eq!(labels.len(), 10);
    }

    // ------------------------------------------------------------------
    // LowRankSubspaceClustering
    // ------------------------------------------------------------------

    #[test]
    fn test_lrsc_output_shape() {
        let x = three_subspace_data();
        let lrsc = LowRankSubspaceClustering::default();
        let labels = lrsc.fit(x.view(), None, None, 3).expect("LRSC failed");
        assert_eq!(labels.len(), 30);
    }

    #[test]
    fn test_lrsc_labels_in_range() {
        let x = three_subspace_data();
        let lrsc = LowRankSubspaceClustering::default();
        let labels = lrsc.fit(x.view(), None, None, 3).expect("LRSC failed");
        for &l in labels.iter() {
            assert!(l < 3, "LRSC label {l} out of range");
        }
    }

    #[test]
    fn test_lrsc_lambda_override() {
        let x = three_subspace_data();
        let lrsc = LowRankSubspaceClustering::default();
        let labels = lrsc
            .fit(x.view(), Some(2.0), Some(30), 3)
            .expect("LRSC lambda override");
        assert_eq!(labels.len(), 30);
    }

    #[test]
    fn test_lrsc_invalid_input() {
        let x = three_subspace_data();
        let lrsc = LowRankSubspaceClustering::default();
        assert!(lrsc.fit(x.view(), None, None, 0).is_err());
        assert!(lrsc.fit(x.view(), None, None, 100).is_err());
        assert!(lrsc.fit(x.view(), Some(-0.5), None, 3).is_err());
        let empty = Array2::<f64>::zeros((0, 4));
        assert!(lrsc.fit(empty.view(), None, None, 3).is_err());
    }

    // ------------------------------------------------------------------
    // ThresholdingSubspace
    // ------------------------------------------------------------------

    #[test]
    fn test_thresholding_output_shape() {
        let x = three_subspace_data();
        let ts = ThresholdingSubspace::default();
        let labels = ts.fit(x.view(), None, 3).expect("Thresholding failed");
        assert_eq!(labels.len(), 30);
    }

    #[test]
    fn test_thresholding_labels_in_range() {
        let x = three_subspace_data();
        let ts = ThresholdingSubspace::default();
        let labels = ts.fit(x.view(), None, 3).expect("Thresholding failed");
        for &l in labels.iter() {
            assert!(l < 3, "Thresholding label {l} out of range");
        }
    }

    #[test]
    fn test_thresholding_threshold_override() {
        let x = three_subspace_data();
        let ts = ThresholdingSubspace {
            threshold: 0.3,
            k_neighbors: 3,
            ..Default::default()
        };
        let labels = ts
            .fit(x.view(), Some(0.4), 3)
            .expect("Thresholding override");
        assert_eq!(labels.len(), 30);
    }

    #[test]
    fn test_thresholding_invalid_input() {
        let x = three_subspace_data();
        let ts = ThresholdingSubspace::default();
        assert!(ts.fit(x.view(), None, 0).is_err());
        assert!(ts.fit(x.view(), None, 100).is_err());
        let empty = Array2::<f64>::zeros((0, 4));
        assert!(ts.fit(empty.view(), None, 3).is_err());
        let no_features = Array2::<f64>::zeros((5, 0));
        assert!(ts.fit(no_features.view(), None, 3).is_err());
    }

    #[test]
    fn test_thresholding_two_subspaces() {
        let mut data = Array2::<f64>::zeros((12, 4));
        for i in 0..6 {
            data[[i, 0]] = (i as f64 + 1.0) * 0.4;
            data[[i, 1]] = (i as f64 + 1.0) * 0.3;
        }
        for i in 6..12 {
            data[[i, 2]] = (i as f64 - 5.0) * 0.4;
            data[[i, 3]] = (i as f64 - 5.0) * 0.3;
        }
        let ts = ThresholdingSubspace {
            threshold: 0.2,
            k_neighbors: 4,
            ..Default::default()
        };
        let labels = ts
            .fit(data.view(), None, 2)
            .expect("Thresholding 2 subspaces");
        assert_eq!(labels.len(), 12);
    }

    // ------------------------------------------------------------------
    // Internal helper tests
    // ------------------------------------------------------------------

    #[test]
    fn test_soft_threshold() {
        assert!((soft_threshold(2.0, 1.0) - 1.0).abs() < 1e-10);
        assert!((soft_threshold(-2.0, 1.0) - (-1.0)).abs() < 1e-10);
        assert!(soft_threshold(0.5, 1.0).abs() < 1e-10);
        assert!(soft_threshold(-0.5, 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_gram_positive_semidefinite() {
        let x = three_subspace_data();
        let g = compute_gram(x.view());
        let n = g.shape()[0];
        // Diagonal should be non-negative (squared norms).
        for i in 0..n {
            assert!(g[[i, i]] >= 0.0, "g[{i},{i}] < 0: {}", g[[i, i]]);
        }
        // Symmetry.
        for i in 0..n {
            for j in 0..n {
                let diff = (g[[i, j]] - g[[j, i]]).abs();
                assert!(diff < 1e-10, "Gram not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_symmetrise_abs_symmetry() {
        let c = Array2::from_shape_vec((3, 3), vec![0.0, 0.3, -0.5, 0.1, 0.0, 0.4, -0.2, 0.6, 0.0])
            .expect("shape");
        let w = symmetrise_abs(&c);
        for i in 0..3 {
            for j in 0..3 {
                let diff = (w[[i, j]] - w[[j, i]]).abs();
                assert!(diff < 1e-10, "symmetrise_abs not symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn test_cg_solve_identity() {
        // For A = I, CG should return b exactly.
        let n = 5;
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            a[[i, i]] = 1.0;
        }
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let x = cg_solve(&a, &b, 20, 1e-10);
        for i in 0..n {
            assert!((x[i] - b[i]).abs() < 1e-8, "CG solve error at {i}");
        }
    }
}
