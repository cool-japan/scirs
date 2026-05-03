//! Co-clustering and biclustering algorithms
//!
//! Co-clustering (also called biclustering) simultaneously partitions both rows
//! and columns of a data matrix. Unlike standard clustering which only groups
//! rows (samples), co-clustering finds groups of rows that behave similarly
//! across specific groups of columns.
//!
//! # Algorithms
//!
//! - [`SpectralCoclustering`]: Dhillon (2001) — SVD-based bipartite graph embedding
//! - [`SpectralBiclustering`]: Kluger et al. (2003) — bistochastic normalisation + SVD
//! - [`InformationCoclustering`]: Dhillon et al. (2003) — minimise mutual information loss
//!
//! # References
//!
//! * Dhillon, I. S. (2001). Co-clustering documents and words using bipartite spectral
//!   graph partitioning. *KDD 2001*.
//! * Kluger, Y. et al. (2003). Spectral biclustering of microarray data: coclustering
//!   genes and conditions. *Genome Research*.
//! * Dhillon, I. S., Mallela, S., & Modha, D. S. (2003). Information-theoretic
//!   co-clustering. *KDD 2003*.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};

use crate::error::{ClusteringError, Result};

// ---------------------------------------------------------------------------
// Core result types
// ---------------------------------------------------------------------------

/// Result of a co-clustering algorithm.
///
/// Stores flat per-row and per-column cluster assignments.
#[derive(Debug, Clone)]
pub struct CoclusterResult {
    /// Cluster assignment for each row; length == n_rows.
    pub row_labels: Array1<usize>,
    /// Cluster assignment for each column; length == n_cols.
    pub col_labels: Array1<usize>,
    /// Number of row clusters.
    pub n_row_clusters: usize,
    /// Number of column clusters.
    pub n_col_clusters: usize,
}

/// A single bicluster module: a contiguous block of rows and columns.
///
/// Extracted post-hoc from a `CoclusterResult` or `BiclusResult`.
#[derive(Debug, Clone)]
pub struct BiclusModule {
    /// Row indices belonging to this bicluster.
    pub rows: Vec<usize>,
    /// Column indices belonging to this bicluster.
    pub cols: Vec<usize>,
    /// Average value of the bicluster sub-matrix.
    pub score: f64,
}

/// Result of a spectral biclustering algorithm.
///
/// Adds the full `BiclusModule` list on top of flat label arrays.
#[derive(Debug, Clone)]
pub struct BiclusResult {
    /// Flat row cluster assignment.
    pub row_labels: Array1<usize>,
    /// Flat column cluster assignment.
    pub col_labels: Array1<usize>,
    /// Number of row clusters.
    pub n_row_clusters: usize,
    /// Number of column clusters.
    pub n_col_clusters: usize,
    /// All bicluster modules (n_row_clusters × n_col_clusters combinations).
    pub biclusters: Vec<BiclusModule>,
}

// ---------------------------------------------------------------------------
// SpectralCoclustering
// ---------------------------------------------------------------------------

/// Spectral co-clustering via Dhillon's bipartite graph SVD.
///
/// Algorithm overview
/// ------------------
/// 1. Construct the bipartite adjacency matrix A (n_rows × n_cols).
/// 2. Form D_r^{-1/2} A D_c^{-1/2}, where D_r = diag(row sums) and
///    D_c = diag(col sums).
/// 3. Compute the `k − 1` singular vectors (beyond the trivial first one).
/// 4. Stack the left and right singular vectors into a feature matrix and
///    apply k-means to obtain simultaneous row and column assignments.
///
/// Reference: Dhillon, I. S. (2001). *Co-clustering documents and words using
/// bipartite spectral graph partitioning.* KDD 2001.
pub struct SpectralCoclustering {
    /// Number of k-means iterations (default 100).
    pub km_max_iter: usize,
    /// RNG seed for k-means initialisation.
    pub seed: u64,
    /// Numerical floor added to row/col sums before inversion (default 1e-10).
    pub epsilon: f64,
}

impl Default for SpectralCoclustering {
    fn default() -> Self {
        Self {
            km_max_iter: 100,
            seed: 42,
            epsilon: 1e-10,
        }
    }
}

impl SpectralCoclustering {
    /// Fit co-clustering to `data`.
    ///
    /// # Arguments
    /// * `data`      – Non-negative data matrix (n_rows × n_cols).
    /// * `n_clusters` – `(n_row_clusters, n_col_clusters)`.
    ///
    /// # Returns
    /// A [`CoclusterResult`] with per-row and per-column assignments.
    pub fn fit(
        &self,
        data: ArrayView2<f64>,
        n_clusters: (usize, usize),
    ) -> Result<CoclusterResult> {
        let (n_rows, n_cols) = (data.shape()[0], data.shape()[1]);
        let (n_rc, n_cc) = n_clusters;

        if n_rows == 0 || n_cols == 0 {
            return Err(ClusteringError::InvalidInput("Empty data matrix".into()));
        }
        if n_rc == 0 || n_cc == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_clusters must both be > 0".into(),
            ));
        }
        if n_rc > n_rows {
            return Err(ClusteringError::InvalidInput(format!(
                "n_row_clusters ({n_rc}) must not exceed n_rows ({n_rows})"
            )));
        }
        if n_cc > n_cols {
            return Err(ClusteringError::InvalidInput(format!(
                "n_col_clusters ({n_cc}) must not exceed n_cols ({n_cols})"
            )));
        }

        // Ensure non-negative values by shifting.
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let shift = if min_val < 0.0 { -min_val } else { 0.0 };

        // Row sums and column sums.
        let mut row_sums = vec![0.0f64; n_rows];
        let mut col_sums = vec![0.0f64; n_cols];
        for r in 0..n_rows {
            for c in 0..n_cols {
                let v = data[[r, c]] + shift;
                row_sums[r] += v;
                col_sums[c] += v;
            }
        }

        // D_r^{-1/2} and D_c^{-1/2}.
        let dr_inv_sqrt: Vec<f64> = row_sums
            .iter()
            .map(|&s| 1.0 / (s + self.epsilon).sqrt())
            .collect();
        let dc_inv_sqrt: Vec<f64> = col_sums
            .iter()
            .map(|&s| 1.0 / (s + self.epsilon).sqrt())
            .collect();

        // Normalized matrix: A_hat[r,c] = dr_inv_sqrt[r] * (data[r,c]+shift) * dc_inv_sqrt[c]
        let mut a_hat = Array2::<f64>::zeros((n_rows, n_cols));
        for r in 0..n_rows {
            for c in 0..n_cols {
                a_hat[[r, c]] = dr_inv_sqrt[r] * (data[[r, c]] + shift) * dc_inv_sqrt[c];
            }
        }

        // Number of singular vectors needed: max(n_rc, n_cc) - 1 (skip trivial).
        // We need at least 1 non-trivial vector; in practice we compute
        // n_components = n_rc + n_cc - 1 to have enough features for both row/col kmeans.
        let n_components = (n_rc + n_cc).saturating_sub(1).max(1);
        let n_sv = n_components.min(n_rows.min(n_cols));

        let (u, _s, vt) = compact_svd(a_hat.view(), n_sv + 1, self.seed)?;

        // Skip the first (trivial) singular triplet if we have more than 1.
        // The trivial vector corresponds to the constant all-ones mode.
        let skip = if n_sv > 0 { 1 } else { 0 };

        // Build row features from left singular vectors (u[:,skip..]).
        let row_k = (n_rc).min(u.shape()[1].saturating_sub(skip));
        let col_k = (n_cc).min(vt.shape()[0].saturating_sub(skip));

        let row_feats = extract_cols(&u, skip, skip + row_k)?;
        // vt is (sv, n_cols); columns of v (transpose rows of vt) are col features.
        let col_feats = extract_rows_transposed(&vt, skip, skip + col_k)?;

        // Scale features by singular values for better separation.
        let row_features = l2_normalise_rows(row_feats);
        let col_features = l2_normalise_rows(col_feats);

        let mut rng = self.seed;
        let row_labels_vec = kmeans_labels(row_features.view(), n_rc, self.km_max_iter, &mut rng)?;
        let col_labels_vec = kmeans_labels(col_features.view(), n_cc, self.km_max_iter, &mut rng)?;

        Ok(CoclusterResult {
            row_labels: Array1::from_vec(row_labels_vec),
            col_labels: Array1::from_vec(col_labels_vec),
            n_row_clusters: n_rc,
            n_col_clusters: n_cc,
        })
    }
}

// ---------------------------------------------------------------------------
// SpectralBiclustering
// ---------------------------------------------------------------------------

/// Spectral biclustering via bistochastic normalisation (Kluger et al. 2003).
///
/// Algorithm overview
/// ------------------
/// 1. Apply independent bistochastic normalisation (Sinkhorn-Knopp until row
///    and column sums are both ≈ 1).
/// 2. Subtract the grand mean from the log-transformed normalised matrix.
/// 3. Extract top-`p` left and right singular vectors via randomised SVD.
/// 4. Apply k-means independently to row feature vectors and column feature
///    vectors to produce row and column cluster assignments.
/// 5. Build `BiclusModule`s from the Cartesian product of (row cluster × col cluster).
///
/// Reference: Kluger, Y. et al. (2003). *Spectral biclustering of microarray
/// data: coclustering genes and conditions.* Genome Research.
pub struct SpectralBiclustering {
    /// Maximum Sinkhorn iterations for bistochastic normalisation (default 20).
    pub sinkhorn_max_iter: usize,
    /// Convergence tolerance for Sinkhorn (default 1e-6).
    pub sinkhorn_tol: f64,
    /// K-means maximum iterations (default 100).
    pub km_max_iter: usize,
    /// RNG seed.
    pub seed: u64,
    /// Numerical floor for log/division (default 1e-12).
    pub epsilon: f64,
}

impl Default for SpectralBiclustering {
    fn default() -> Self {
        Self {
            sinkhorn_max_iter: 20,
            sinkhorn_tol: 1e-6,
            km_max_iter: 100,
            seed: 42,
            epsilon: 1e-12,
        }
    }
}

impl SpectralBiclustering {
    /// Fit spectral biclustering.
    ///
    /// # Arguments
    /// * `data`          – Data matrix (n_rows × n_cols). Shifted to positive before use.
    /// * `n_row_clusters` – Number of row clusters.
    /// * `n_col_clusters` – Number of column clusters.
    pub fn fit(
        &self,
        data: ArrayView2<f64>,
        n_row_clusters: usize,
        n_col_clusters: usize,
    ) -> Result<BiclusResult> {
        let (n_rows, n_cols) = (data.shape()[0], data.shape()[1]);

        if n_rows == 0 || n_cols == 0 {
            return Err(ClusteringError::InvalidInput("Empty data matrix".into()));
        }
        if n_row_clusters == 0 || n_col_clusters == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_row_clusters and n_col_clusters must be > 0".into(),
            ));
        }
        if n_row_clusters > n_rows {
            return Err(ClusteringError::InvalidInput(format!(
                "n_row_clusters ({n_row_clusters}) exceeds n_rows ({n_rows})"
            )));
        }
        if n_col_clusters > n_cols {
            return Err(ClusteringError::InvalidInput(format!(
                "n_col_clusters ({n_col_clusters}) exceeds n_cols ({n_cols})"
            )));
        }

        // Step 1: Shift data to strictly positive.
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let shift = if min_val <= 0.0 {
            (-min_val) + 1.0
        } else {
            0.0
        };
        let mut a = data.to_owned();
        a.mapv_inplace(|v| (v + shift).max(self.epsilon));

        // Step 2: Bistochastic normalisation (Sinkhorn-Knopp).
        bistochastise(
            &mut a,
            self.sinkhorn_max_iter,
            self.sinkhorn_tol,
            self.epsilon,
        );

        // Step 3: Log-transform and centre.
        a.mapv_inplace(|v| v.max(self.epsilon).ln());
        let global_mean = a.mean().unwrap_or(0.0);
        a.mapv_inplace(|v| v - global_mean);

        // Step 4: Compact SVD.
        let n_sv = (n_row_clusters + n_col_clusters)
            .saturating_sub(1)
            .max(1)
            .min(n_rows.min(n_cols));

        let (u, _s, vt) = compact_svd(a.view(), n_sv, self.seed)?;

        // Step 5: Build row/col features.
        let row_k = n_row_clusters.min(u.shape()[1]);
        let col_k = n_col_clusters.min(vt.shape()[0]);

        let row_feats = extract_cols(&u, 0, row_k)?;
        let col_feats = extract_rows_transposed(&vt, 0, col_k)?;

        let row_features = l2_normalise_rows(row_feats);
        let col_features = l2_normalise_rows(col_feats);

        let mut rng = self.seed;
        let row_labels_vec = kmeans_labels(
            row_features.view(),
            n_row_clusters,
            self.km_max_iter,
            &mut rng,
        )?;
        let col_labels_vec = kmeans_labels(
            col_features.view(),
            n_col_clusters,
            self.km_max_iter,
            &mut rng,
        )?;

        // Step 6: Build BiclusModules from Cartesian product of cluster pairs.
        let mut biclusters: Vec<BiclusModule> = Vec::with_capacity(n_row_clusters * n_col_clusters);
        for rc in 0..n_row_clusters {
            for cc in 0..n_col_clusters {
                let rows: Vec<usize> = (0..n_rows).filter(|&r| row_labels_vec[r] == rc).collect();
                let cols: Vec<usize> = (0..n_cols).filter(|&c| col_labels_vec[c] == cc).collect();

                let score = if rows.is_empty() || cols.is_empty() {
                    0.0
                } else {
                    let sum: f64 = rows
                        .iter()
                        .flat_map(|&r| cols.iter().map(move |&c| data[[r, c]]))
                        .sum();
                    sum / (rows.len() * cols.len()) as f64
                };

                biclusters.push(BiclusModule { rows, cols, score });
            }
        }

        Ok(BiclusResult {
            row_labels: Array1::from_vec(row_labels_vec),
            col_labels: Array1::from_vec(col_labels_vec),
            n_row_clusters,
            n_col_clusters,
            biclusters,
        })
    }
}

// ---------------------------------------------------------------------------
// InformationCoclustering
// ---------------------------------------------------------------------------

/// Information-theoretic co-clustering (Dhillon, Mallela & Modha 2003).
///
/// Minimises the loss of mutual information I(X;Y) incurred by replacing X (row
/// variable) and Y (column variable) with their respective cluster variables X̂
/// and Ŷ.
///
/// Algorithm
/// ----------
/// 1. Treat data as a joint probability table: p[r,c] = data[r,c] / Σ data.
/// 2. Initialise row and column assignments with round-robin.
/// 3. Alternating optimisation:
///    a. Compute cluster-conditional column distributions q[r_hat, c] = Σ_{r: rc=r_hat} p[r,c] / Σ_r(rc=r_hat).
///    b. Reassign each row to the cluster that minimises the KL divergence
///       between its row distribution and the cluster's column distribution.
///    c. Compute cluster-conditional row distributions q_c[r, c_hat] similarly.
///    d. Reassign each column analogously.
/// 4. Repeat until convergence or `max_iter` exceeded.
///
/// Reference: Dhillon, I. S., Mallela, S., & Modha, D. S. (2003).
/// *Information-theoretic co-clustering.* KDD 2003.
pub struct InformationCoclustering {
    /// Maximum number of alternating update iterations (default 50).
    pub max_iter: usize,
    /// Convergence tolerance on objective (default 1e-6).
    pub tol: f64,
    /// RNG seed for initial assignments.
    pub seed: u64,
}

impl Default for InformationCoclustering {
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-6,
            seed: 42,
        }
    }
}

impl InformationCoclustering {
    /// Fit information-theoretic co-clustering.
    ///
    /// # Arguments
    /// * `data`          – Non-negative data matrix (n_rows × n_cols).
    /// * `n_row_clusters` – Number of row clusters.
    /// * `n_col_clusters` – Number of column clusters.
    /// * `max_iter`      – Override the maximum iterations (pass `None` to use `self.max_iter`).
    pub fn fit(
        &self,
        data: ArrayView2<f64>,
        n_row_clusters: usize,
        n_col_clusters: usize,
        max_iter: Option<usize>,
    ) -> Result<CoclusterResult> {
        let (n_rows, n_cols) = (data.shape()[0], data.shape()[1]);
        let max_it = max_iter.unwrap_or(self.max_iter);

        if n_rows == 0 || n_cols == 0 {
            return Err(ClusteringError::InvalidInput("Empty data matrix".into()));
        }
        if n_row_clusters == 0 || n_col_clusters == 0 {
            return Err(ClusteringError::InvalidInput(
                "n_row_clusters and n_col_clusters must both be > 0".into(),
            ));
        }
        if n_row_clusters > n_rows {
            return Err(ClusteringError::InvalidInput(format!(
                "n_row_clusters ({n_row_clusters}) exceeds n_rows ({n_rows})"
            )));
        }
        if n_col_clusters > n_cols {
            return Err(ClusteringError::InvalidInput(format!(
                "n_col_clusters ({n_col_clusters}) exceeds n_cols ({n_cols})"
            )));
        }

        // Build joint probability table p[r,c].
        let total: f64 = data.iter().cloned().sum::<f64>().max(1e-300);
        let mut p = data.to_owned();
        p.mapv_inplace(|v| (v.max(0.0)) / total);

        // Marginals p_r[r] and p_c[c].
        let p_r: Vec<f64> = (0..n_rows)
            .map(|r| (0..n_cols).map(|c| p[[r, c]]).sum::<f64>())
            .collect();
        let p_c: Vec<f64> = (0..n_cols)
            .map(|c| (0..n_rows).map(|r| p[[r, c]]).sum::<f64>())
            .collect();

        // Initialise row and column assignments round-robin.
        let mut rng = self.seed;
        let mut row_labels = init_labels_shuffled(n_rows, n_row_clusters, &mut rng);
        let mut col_labels = init_labels_shuffled(n_cols, n_col_clusters, &mut rng);

        let mut prev_obj = f64::INFINITY;

        for _iter in 0..max_it {
            // --- Update row assignments ---
            // Cluster col distributions: q_r[rc, c] = Σ_{r: rc(r)=rc} p[r,c] / Σ_{r:rc=rc} p_r[r]
            let mut q_rc = vec![vec![0.0f64; n_cols]; n_row_clusters];
            let mut p_rc = vec![0.0f64; n_row_clusters]; // marginals of row clusters

            for r in 0..n_rows {
                let rc = row_labels[r];
                for c in 0..n_cols {
                    q_rc[rc][c] += p[[r, c]];
                }
                p_rc[rc] += p_r[r];
            }

            // Normalise to get conditional distributions.
            for rc in 0..n_row_clusters {
                let mass = p_rc[rc].max(1e-300);
                for c in 0..n_cols {
                    q_rc[rc][c] /= mass;
                }
            }

            // Reassign each row: minimise KL(p[r,·] || q_rc[rc,·]).
            for r in 0..n_rows {
                let p_r_val = p_r[r].max(1e-300);
                let mut best_rc = row_labels[r];
                let mut best_kl = f64::INFINITY;
                for rc in 0..n_row_clusters {
                    let mut kl = 0.0;
                    for c in 0..n_cols {
                        let p_val = p[[r, c]] / p_r_val;
                        let q_val = q_rc[rc][c].max(1e-300);
                        if p_val > 1e-300 {
                            kl += p_val * (p_val / q_val).ln();
                        }
                    }
                    if kl < best_kl {
                        best_kl = kl;
                        best_rc = rc;
                    }
                }
                row_labels[r] = best_rc;
            }

            // --- Update column assignments ---
            let mut q_cc = vec![vec![0.0f64; n_rows]; n_col_clusters];
            let mut p_cc = vec![0.0f64; n_col_clusters];

            for c in 0..n_cols {
                let cc = col_labels[c];
                for r in 0..n_rows {
                    q_cc[cc][r] += p[[r, c]];
                }
                p_cc[cc] += p_c[c];
            }

            for cc in 0..n_col_clusters {
                let mass = p_cc[cc].max(1e-300);
                for r in 0..n_rows {
                    q_cc[cc][r] /= mass;
                }
            }

            for c in 0..n_cols {
                let p_c_val = p_c[c].max(1e-300);
                let mut best_cc = col_labels[c];
                let mut best_kl = f64::INFINITY;
                for cc in 0..n_col_clusters {
                    let mut kl = 0.0;
                    for r in 0..n_rows {
                        let p_val = p[[r, c]] / p_c_val;
                        let q_val = q_cc[cc][r].max(1e-300);
                        if p_val > 1e-300 {
                            kl += p_val * (p_val / q_val).ln();
                        }
                    }
                    if kl < best_kl {
                        best_kl = kl;
                        best_cc = cc;
                    }
                }
                col_labels[c] = best_cc;
            }

            // Compute objective: mutual information loss approximation.
            let obj = compute_mi_loss(&p, &row_labels, &col_labels, n_row_clusters, n_col_clusters);

            if (prev_obj - obj).abs() < self.tol {
                break;
            }
            prev_obj = obj;
        }

        Ok(CoclusterResult {
            row_labels: Array1::from_vec(row_labels),
            col_labels: Array1::from_vec(col_labels),
            n_row_clusters,
            n_col_clusters,
        })
    }
}

// ---------------------------------------------------------------------------
// extract_biclusters
// ---------------------------------------------------------------------------

/// Extract individual bicluster modules from a co-clustering result.
///
/// Iterates over all (row-cluster, col-cluster) pairs and returns those whose
/// sub-matrix contains at least `min_rows` rows and `min_cols` columns.
///
/// # Arguments
/// * `data`      – Original data matrix.
/// * `result`    – A [`CoclusterResult`] (e.g. from [`SpectralCoclustering`]).
/// * `min_rows`  – Minimum number of rows a bicluster must have.
/// * `min_cols`  – Minimum number of columns a bicluster must have.
pub fn extract_biclusters(
    data: ArrayView2<f64>,
    result: &CoclusterResult,
    min_rows: usize,
    min_cols: usize,
) -> Vec<BiclusModule> {
    let n_rows = data.shape()[0];
    let n_cols = data.shape()[1];
    let mut modules = Vec::new();

    for rc in 0..result.n_row_clusters {
        let rows: Vec<usize> = (0..n_rows)
            .filter(|&r| result.row_labels[r] == rc)
            .collect();
        if rows.len() < min_rows {
            continue;
        }

        for cc in 0..result.n_col_clusters {
            let cols: Vec<usize> = (0..n_cols)
                .filter(|&c| result.col_labels[c] == cc)
                .collect();
            if cols.len() < min_cols {
                continue;
            }

            let score = {
                let sum: f64 = rows
                    .iter()
                    .flat_map(|&r| cols.iter().map(move |&c| data[[r, c]]))
                    .sum();
                sum / (rows.len() * cols.len()) as f64
            };

            modules.push(BiclusModule {
                rows: rows.clone(),
                cols: cols.clone(),
                score,
            });
        }
    }

    // Sort by descending score for usability.
    modules.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    modules
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

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

/// Normalise each row of `a` to unit L2 norm.
fn l2_normalise_rows(mut a: Array2<f64>) -> Array2<f64> {
    let (n, d) = (a.shape()[0], a.shape()[1]);
    for r in 0..n {
        let norm: f64 = (0..d).map(|c| a[[r, c]] * a[[r, c]]).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for c in 0..d {
                a[[r, c]] /= norm;
            }
        }
    }
    a
}

/// Extract columns `[from, to)` from a matrix.
fn extract_cols(mat: &Array2<f64>, from: usize, to: usize) -> Result<Array2<f64>> {
    let (n, k) = (mat.shape()[0], mat.shape()[1]);
    let to = to.min(k);
    if from >= to {
        // Return zeros with at least 1 column to avoid degenerate k-means.
        return Ok(Array2::zeros((n, 1)));
    }
    let width = to - from;
    let mut out = Array2::<f64>::zeros((n, width));
    for r in 0..n {
        for (j, c) in (from..to).enumerate() {
            out[[r, j]] = mat[[r, c]];
        }
    }
    Ok(out)
}

/// Extract rows `[from, to)` of `vt` (shape sv × n_cols) and return them as
/// an `(n_cols × width)` matrix (treating each column of Vᵀ as a feature
/// vector for the corresponding data column).
fn extract_rows_transposed(vt: &Array2<f64>, from: usize, to: usize) -> Result<Array2<f64>> {
    let (sv, n_cols) = (vt.shape()[0], vt.shape()[1]);
    let to = to.min(sv);
    if from >= to {
        return Ok(Array2::zeros((n_cols, 1)));
    }
    let width = to - from;
    let mut out = Array2::<f64>::zeros((n_cols, width));
    for c in 0..n_cols {
        for (j, row) in (from..to).enumerate() {
            out[[c, j]] = vt[[row, c]];
        }
    }
    Ok(out)
}

/// Compact randomised SVD: returns (U, s, Vt) with shapes (m,k), (k,), (k,n).
///
/// Uses a Gaussian random sketch followed by QR-orthogonalisation and
/// eigendecomposition of the small B B^T matrix.
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

    // Y = X Ω  (m × k)
    let mut y = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            let mut s = 0.0;
            for l in 0..n {
                s += x[[i, l]] * omega[[l, j]];
            }
            y[[i, j]] = s;
        }
    }

    // QR of Y → Q (m × k).
    let q = gram_schmidt(y.view())?;

    // B = Q^T X  (k × n)
    let mut b = Array2::<f64>::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            let mut s = 0.0;
            for l in 0..m {
                s += q[[l, i]] * x[[l, j]];
            }
            b[[i, j]] = s;
        }
    }

    // Eigen-decompose B B^T  (k × k).
    let mut bbt = Array2::<f64>::zeros((k, k));
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for l in 0..n {
                s += b[[i, l]] * b[[j, l]];
            }
            bbt[[i, j]] = s;
        }
    }

    let (ub, sigma) = power_iter_eig(bbt.view(), k, seed.wrapping_add(1))?;

    // Singular values.
    let s: Array1<f64> = sigma.mapv(|v| v.max(0.0).sqrt());

    // U = Q U_B  (m × k)
    let mut u = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            let mut val = 0.0;
            for l in 0..k {
                val += q[[i, l]] * ub[[l, j]];
            }
            u[[i, j]] = val;
        }
    }

    // V^T = U_B^T B / σ  (k × n)
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

/// Gram-Schmidt orthonormalisation.
fn gram_schmidt(a: ArrayView2<f64>) -> Result<Array2<f64>> {
    let (m, n) = (a.shape()[0], a.shape()[1]);
    let mut q = Array2::<f64>::zeros((m, n));

    for j in 0..n {
        let mut v = a.column(j).to_owned();
        for i in 0..j {
            let dot: f64 = v.iter().zip(q.column(i).iter()).map(|(a, b)| a * b).sum();
            for k in 0..m {
                v[k] -= dot * q[[k, i]];
            }
        }
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            if j < m {
                q[[j, j]] = 1.0;
            }
        } else {
            for k in 0..m {
                q[[k, j]] = v[k] / norm;
            }
        }
    }
    Ok(q)
}

/// Power-iteration eigen-decomposition of a symmetric matrix (top-k).
fn power_iter_eig(a: ArrayView2<f64>, k: usize, seed: u64) -> Result<(Array2<f64>, Array1<f64>)> {
    let n = a.shape()[0];
    let k = k.min(n);
    let mut rng = seed;

    let mut eigvecs = Array2::<f64>::zeros((n, k));
    let mut eigvals = Array1::<f64>::zeros(k);
    let mut deflated = a.to_owned();

    for col in 0..k {
        let mut v: Vec<f64> = (0..n).map(|_| lcg_f64(&mut rng) - 0.5).collect();
        normalize_vec(&mut v);

        for _ in 0..200 {
            let mut av: Vec<f64> = vec![0.0; n];
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
            normalize_vec(&mut av);
            v = av;
        }

        let mut av: Vec<f64> = vec![0.0; n];
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

/// Normalise a Vec<f64> in place (no-op if near zero).
fn normalize_vec(v: &mut Vec<f64>) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// K-means on row-feature matrix; returns per-row cluster indices.
fn kmeans_labels(
    features: ArrayView2<f64>,
    k: usize,
    max_iter: usize,
    rng: &mut u64,
) -> Result<Vec<usize>> {
    let (n, d) = (features.shape()[0], features.shape()[1]);
    if n == 0 || k == 0 {
        return Ok(vec![0; n]);
    }
    let k = k.min(n);

    // Shuffle-sample initial centroids.
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = lcg_usize(rng, i + 1);
        indices.swap(i, j);
    }
    let mut centroids: Vec<Vec<f64>> = indices[..k]
        .iter()
        .map(|&i| features.row(i).to_vec())
        .collect();

    let mut labels = vec![0usize; n];

    for _iter in 0..max_iter {
        let mut changed = false;

        for i in 0..n {
            let row = features.row(i).to_vec();
            let best = (0..k)
                .map(|j| {
                    let dist: f64 = row
                        .iter()
                        .zip(centroids[j].iter())
                        .map(|(a, b)| (a - b) * (a - b))
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

        // Update centroids.
        let mut sums: Vec<Vec<f64>> = vec![vec![0.0; d]; k];
        let mut counts: Vec<usize> = vec![0; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for dim in 0..d {
                sums[c][dim] += features[[i, dim]];
            }
        }
        for j in 0..k {
            if counts[j] == 0 {
                let ri = lcg_usize(rng, n);
                centroids[j] = features.row(ri).to_vec();
            } else {
                for dim in 0..d {
                    centroids[j][dim] = sums[j][dim] / counts[j] as f64;
                }
            }
        }
    }

    Ok(labels)
}

/// Bistochastise: Sinkhorn-Knopp normalisation so that row and col sums ≈ 1.
fn bistochastise(a: &mut Array2<f64>, max_iter: usize, tol: f64, eps: f64) {
    let (n_rows, n_cols) = (a.shape()[0], a.shape()[1]);

    for _ in 0..max_iter {
        // Row normalisation.
        let mut max_dev = 0.0f64;
        for r in 0..n_rows {
            let s: f64 = (0..n_cols).map(|c| a[[r, c]]).sum::<f64>().max(eps);
            for c in 0..n_cols {
                a[[r, c]] /= s;
            }
            max_dev = max_dev.max((s - 1.0).abs());
        }

        // Column normalisation.
        for c in 0..n_cols {
            let s: f64 = (0..n_rows).map(|r| a[[r, c]]).sum::<f64>().max(eps);
            for r in 0..n_rows {
                a[[r, c]] /= s;
            }
            max_dev = max_dev.max((s - 1.0).abs());
        }

        if max_dev < tol {
            break;
        }
    }
}

/// Initialise cluster labels by shuffling and assigning round-robin.
fn init_labels_shuffled(n: usize, k: usize, rng: &mut u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = lcg_usize(rng, i + 1);
        indices.swap(i, j);
    }
    let mut labels = vec![0usize; n];
    for (rank, &idx) in indices.iter().enumerate() {
        labels[idx] = rank % k;
    }
    labels
}

/// Approximate mutual information loss for the current co-clustering.
///
/// Returns −I(X̂; Ŷ) approximated as the negative log of the average cluster
/// joint probability, which monotonically increases as the partition improves.
fn compute_mi_loss(
    p: &Array2<f64>,
    row_labels: &[usize],
    col_labels: &[usize],
    n_rc: usize,
    n_cc: usize,
) -> f64 {
    let n_rows = p.shape()[0];
    let n_cols = p.shape()[1];

    // Joint distribution of cluster pairs.
    let mut p_joint = vec![vec![0.0f64; n_cc]; n_rc];
    for r in 0..n_rows {
        for c in 0..n_cols {
            p_joint[row_labels[r]][col_labels[c]] += p[[r, c]];
        }
    }

    // Cluster marginals.
    let mut p_rc_marg: Vec<f64> = vec![0.0; n_rc];
    let mut p_cc_marg: Vec<f64> = vec![0.0; n_cc];
    for rc in 0..n_rc {
        for cc in 0..n_cc {
            p_rc_marg[rc] += p_joint[rc][cc];
            p_cc_marg[cc] += p_joint[rc][cc];
        }
    }

    // Mutual information I(X̂; Ŷ) = Σ p_joint log(p_joint / (p_rc * p_cc)).
    let mut mi = 0.0f64;
    for rc in 0..n_rc {
        for cc in 0..n_cc {
            let pj = p_joint[rc][cc];
            if pj > 1e-300 {
                let denom = (p_rc_marg[rc] * p_cc_marg[cc]).max(1e-300);
                mi += pj * (pj / denom).ln();
            }
        }
    }

    // Return negative MI as "loss" (minimised when MI is maximised).
    -mi
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    /// Create a block-diagonal matrix that naturally has two row and two col clusters.
    fn block_matrix_2x2() -> Array2<f64> {
        // 8 rows × 8 cols; [0..4, 0..4] = 5.0, [4..8, 4..8] = 3.0.
        let mut m = Array2::<f64>::zeros((8, 8));
        for r in 0..4 {
            for c in 0..4 {
                m[[r, c]] = 5.0 + (r as f64) * 0.1 + (c as f64) * 0.05;
            }
        }
        for r in 4..8 {
            for c in 4..8 {
                m[[r, c]] = 3.0 + ((r - 4) as f64) * 0.1 + ((c - 4) as f64) * 0.05;
            }
        }
        m
    }

    fn block_matrix_nonneg() -> Array2<f64> {
        // Document-word count style matrix: non-negative.
        let mut m = Array2::<f64>::zeros((12, 10));
        // Block 1: rows 0..6, cols 0..5
        for r in 0..6 {
            for c in 0..5 {
                m[[r, c]] = 8.0 + (r as f64) * 0.3;
            }
        }
        // Block 2: rows 6..12, cols 5..10
        for r in 6..12 {
            for c in 5..10 {
                m[[r, c]] = 6.0 + ((r - 6) as f64) * 0.2;
            }
        }
        m
    }

    // ------------------------------------------------------------------
    // SpectralCoclustering
    // ------------------------------------------------------------------

    #[test]
    fn test_spectral_cocluster_shapes() {
        let x = block_matrix_nonneg();
        let scc = SpectralCoclustering::default();
        let result = scc.fit(x.view(), (2, 2)).expect("spectral cocluster");
        assert_eq!(result.row_labels.len(), 12);
        assert_eq!(result.col_labels.len(), 10);
        assert_eq!(result.n_row_clusters, 2);
        assert_eq!(result.n_col_clusters, 2);
    }

    #[test]
    fn test_spectral_cocluster_all_labels_valid() {
        let x = block_matrix_nonneg();
        let scc = SpectralCoclustering::default();
        let result = scc.fit(x.view(), (2, 2)).expect("spectral cocluster");
        for &l in result.row_labels.iter() {
            assert!(l < 2, "row label {l} out of range");
        }
        for &l in result.col_labels.iter() {
            assert!(l < 2, "col label {l} out of range");
        }
    }

    #[test]
    fn test_spectral_cocluster_asymmetric_clusters() {
        let x = block_matrix_nonneg();
        let scc = SpectralCoclustering::default();
        let result = scc.fit(x.view(), (3, 2)).expect("asymmetric cocluster");
        assert_eq!(result.n_row_clusters, 3);
        assert_eq!(result.n_col_clusters, 2);
        assert_eq!(result.row_labels.len(), 12);
        assert_eq!(result.col_labels.len(), 10);
    }

    #[test]
    fn test_spectral_cocluster_invalid_input() {
        let x = block_matrix_nonneg();
        let scc = SpectralCoclustering::default();
        // n_row_clusters > n_rows
        assert!(scc.fit(x.view(), (20, 2)).is_err());
        // n_col_clusters == 0
        assert!(scc.fit(x.view(), (2, 0)).is_err());
        // Empty matrix
        let empty = Array2::<f64>::zeros((0, 5));
        assert!(scc.fit(empty.view(), (2, 2)).is_err());
    }

    #[test]
    fn test_spectral_cocluster_negative_values() {
        // Should shift negative values internally and still work.
        let mut x = block_matrix_nonneg();
        x.mapv_inplace(|v| v - 10.0);
        let scc = SpectralCoclustering::default();
        let result = scc.fit(x.view(), (2, 2)).expect("negative values shifted");
        assert_eq!(result.row_labels.len(), 12);
    }

    // ------------------------------------------------------------------
    // SpectralBiclustering
    // ------------------------------------------------------------------

    #[test]
    fn test_spectral_biclustering_shapes() {
        let x = block_matrix_2x2();
        let sb = SpectralBiclustering::default();
        let result = sb.fit(x.view(), 2, 2).expect("spectral bicluster");
        assert_eq!(result.row_labels.len(), 8);
        assert_eq!(result.col_labels.len(), 8);
        assert_eq!(result.n_row_clusters, 2);
        assert_eq!(result.n_col_clusters, 2);
    }

    #[test]
    fn test_spectral_biclustering_bicluster_count() {
        let x = block_matrix_2x2();
        let sb = SpectralBiclustering::default();
        let result = sb.fit(x.view(), 2, 2).expect("spectral bicluster");
        // 2 row clusters × 2 col clusters = 4 biclusters.
        assert_eq!(result.biclusters.len(), 4);
    }

    #[test]
    fn test_spectral_biclustering_all_rows_assigned() {
        let x = block_matrix_2x2();
        let sb = SpectralBiclustering::default();
        let result = sb.fit(x.view(), 2, 2).expect("spectral bicluster");
        // In biclustering, each row appears in exactly n_col_clusters biclusters
        // (one per column-cluster combination via the Cartesian product).
        // Verify that every row appears in at least one bicluster and row
        // assignments are consistent with the flat row_labels.
        let n_rows = result.row_labels.len();
        let mut row_coverage = vec![0usize; n_rows];
        for bc in &result.biclusters {
            for &r in &bc.rows {
                row_coverage[r] += 1;
            }
        }
        // Every row must be covered (at least once).
        for (r, &cnt) in row_coverage.iter().enumerate() {
            assert!(cnt > 0, "row {r} not covered by any bicluster");
        }
        // Each row appears in exactly n_col_clusters biclusters.
        for (r, &cnt) in row_coverage.iter().enumerate() {
            assert_eq!(
                cnt, result.n_col_clusters,
                "row {r} appears in {cnt} biclusters, expected {}",
                result.n_col_clusters
            );
        }
    }

    #[test]
    fn test_spectral_biclustering_asymmetric() {
        let x = block_matrix_nonneg();
        let sb = SpectralBiclustering::default();
        let result = sb.fit(x.view(), 3, 2).expect("asymmetric bicluster");
        // 3 × 2 = 6 biclusters.
        assert_eq!(result.biclusters.len(), 6);
    }

    #[test]
    fn test_spectral_biclustering_invalid() {
        let x = block_matrix_2x2();
        let sb = SpectralBiclustering::default();
        assert!(sb.fit(x.view(), 0, 2).is_err());
        assert!(sb.fit(x.view(), 100, 2).is_err());
        let empty = Array2::<f64>::zeros((0, 4));
        assert!(sb.fit(empty.view(), 2, 2).is_err());
    }

    // ------------------------------------------------------------------
    // InformationCoclustering
    // ------------------------------------------------------------------

    #[test]
    fn test_info_cocluster_shapes() {
        let x = block_matrix_nonneg();
        let ic = InformationCoclustering::default();
        let result = ic.fit(x.view(), 2, 2, None).expect("info cocluster");
        assert_eq!(result.row_labels.len(), 12);
        assert_eq!(result.col_labels.len(), 10);
        assert_eq!(result.n_row_clusters, 2);
        assert_eq!(result.n_col_clusters, 2);
    }

    #[test]
    fn test_info_cocluster_all_labels_valid() {
        let x = block_matrix_nonneg();
        let ic = InformationCoclustering::default();
        let result = ic.fit(x.view(), 2, 2, None).expect("info cocluster");
        for &l in result.row_labels.iter() {
            assert!(l < 2);
        }
        for &l in result.col_labels.iter() {
            assert!(l < 2);
        }
    }

    #[test]
    fn test_info_cocluster_max_iter_override() {
        let x = block_matrix_nonneg();
        let ic = InformationCoclustering {
            max_iter: 1,
            ..Default::default()
        };
        // With max_iter overridden to 5, should still produce valid result.
        let result = ic
            .fit(x.view(), 2, 2, Some(5))
            .expect("info cocluster 5 iter");
        assert_eq!(result.row_labels.len(), 12);
    }

    #[test]
    fn test_info_cocluster_invalid_input() {
        let x = block_matrix_nonneg();
        let ic = InformationCoclustering::default();
        assert!(ic.fit(x.view(), 0, 2, None).is_err());
        assert!(ic.fit(x.view(), 2, 0, None).is_err());
        assert!(ic.fit(x.view(), 20, 2, None).is_err());
        let empty = Array2::<f64>::zeros((0, 4));
        assert!(ic.fit(empty.view(), 2, 2, None).is_err());
    }

    #[test]
    fn test_info_cocluster_3x3() {
        let x = {
            let mut m = Array2::<f64>::zeros((9, 9));
            for r in 0..3 {
                for c in 0..3 {
                    m[[r, c]] = 10.0;
                }
            }
            for r in 3..6 {
                for c in 3..6 {
                    m[[r, c]] = 8.0;
                }
            }
            for r in 6..9 {
                for c in 6..9 {
                    m[[r, c]] = 6.0;
                }
            }
            m
        };
        let ic = InformationCoclustering {
            max_iter: 30,
            ..Default::default()
        };
        let result = ic.fit(x.view(), 3, 3, None).expect("3x3 info cocluster");
        assert_eq!(result.n_row_clusters, 3);
        assert_eq!(result.n_col_clusters, 3);
    }

    // ------------------------------------------------------------------
    // extract_biclusters
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_biclusters_basic() {
        let x = block_matrix_nonneg();
        let scc = SpectralCoclustering::default();
        let result = scc.fit(x.view(), (2, 2)).expect("cocluster");
        let modules = extract_biclusters(x.view(), &result, 1, 1);
        // Should have at most n_row_clusters × n_col_clusters = 4 modules.
        assert!(modules.len() <= 4);
        // Every module should satisfy the minimum size constraints.
        for m in &modules {
            assert!(m.rows.len() >= 1);
            assert!(m.cols.len() >= 1);
        }
    }

    #[test]
    fn test_extract_biclusters_min_size_filter() {
        let x = block_matrix_nonneg();
        let scc = SpectralCoclustering::default();
        let result = scc.fit(x.view(), (2, 2)).expect("cocluster");
        // With very large min_rows, nothing should pass.
        let modules = extract_biclusters(x.view(), &result, 100, 1);
        assert!(modules.is_empty());
    }

    #[test]
    fn test_extract_biclusters_sorted_by_score() {
        let x = block_matrix_nonneg();
        let scc = SpectralCoclustering::default();
        let result = scc.fit(x.view(), (2, 2)).expect("cocluster");
        let modules = extract_biclusters(x.view(), &result, 1, 1);
        // Verify descending score order.
        for i in 1..modules.len() {
            assert!(
                modules[i - 1].score >= modules[i].score,
                "modules not sorted: {} < {}",
                modules[i - 1].score,
                modules[i].score
            );
        }
    }

    // ------------------------------------------------------------------
    // Helper / primitive tests
    // ------------------------------------------------------------------

    #[test]
    fn test_bistochastise_convergence() {
        let mut m =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("shape");
        bistochastise(&mut m, 100, 1e-9, 1e-12);
        // After Sinkhorn, all row sums and col sums should be ≈ 1.
        for r in 0..3 {
            let s: f64 = (0..3).map(|c| m[[r, c]]).sum();
            assert!((s - 1.0).abs() < 1e-6, "row {r} sum = {s}");
        }
        for c in 0..3 {
            let s: f64 = (0..3).map(|r| m[[r, c]]).sum();
            assert!((s - 1.0).abs() < 1e-6, "col {c} sum = {s}");
        }
    }

    #[test]
    fn test_compact_svd_reconstruction() {
        // Verify U S V^T ≈ original for a small rank-1 matrix.
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![2.0, 4.0, 6.0, 1.0, 2.0, 3.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0],
        )
        .expect("shape");
        let (u, s, vt) = compact_svd(x.view(), 2, 1).expect("svd");

        // Reconstruct: X ≈ U diag(s) V^T.
        let (m, k) = (u.shape()[0], u.shape()[1]);
        let n = vt.shape()[1];
        let mut recon = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let mut v = 0.0;
                for l in 0..k {
                    v += u[[i, l]] * s[l] * vt[[l, j]];
                }
                recon[[i, j]] = v;
            }
        }

        // The dominant singular value should capture most of the variance.
        let diff_sq: f64 = recon
            .iter()
            .zip(x.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        let orig_sq: f64 = x.iter().map(|v| v * v).sum();
        assert!(
            diff_sq / orig_sq < 0.1,
            "SVD reconstruction error too large"
        );
    }

    #[test]
    fn test_l2_normalise_rows() {
        let m = Array2::from_shape_vec((2, 3), vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0]).expect("shape");
        let n = l2_normalise_rows(m);
        let norm0: f64 = (0..3).map(|c| n[[0, c]] * n[[0, c]]).sum::<f64>().sqrt();
        assert!((norm0 - 1.0).abs() < 1e-10, "row 0 not unit: {norm0}");
        // Zero row remains zero.
        let norm1: f64 = (0..3).map(|c| n[[1, c]] * n[[1, c]]).sum::<f64>().sqrt();
        assert!(norm1 < 1e-10, "zero row should remain zero: {norm1}");
    }
}
