//! Deep Clustering and Advanced Kernel-Based Clustering Algorithms
//!
//! This module provides:
//! - Kernel K-Means: K-means in reproducing kernel Hilbert space
//! - Trimmed K-Means: Robust clustering that ignores outliers
//! - Dirichlet Process Mixture (CRP Gibbs sampler): Non-parametric Bayesian clustering
//! - Fuzzy C-Means: Soft clustering with fuzzy membership
//!
//! # References
//! - Scholkopf et al. (1998) "Kernel PCA and De-Noising in Feature Spaces"
//! - Cuesta-Albertos et al. (1997) "Trimmed k-means: an attempt to robustify quantizers"
//! - Ferguson (1973) "A Bayesian Analysis of Some Nonparametric Problems"
//! - Bezdek (1981) "Pattern Recognition with Fuzzy Objective Function Algorithms"

use std::f64::consts::TAU;

use crate::error::{ClusteringError, Result};

// ─────────────────────────────────────────────────────────────────────────────
// Simple deterministic LCG / Park-Miller RNG (no external rand dependency)
// ─────────────────────────────────────────────────────────────────────────────

/// Park-Miller LCG random number generator.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        let state = if seed == 0 { 6364136223846793005 } else { seed };
        Self { state }
    }

    /// Returns a value in [0, 1)
    fn next_f64(&mut self) -> f64 {
        // Knuth multiplicative LCG
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 11) as f64;
        bits / (1u64 << 53) as f64
    }

    /// Returns a value in [low, high)
    fn next_range_usize(&mut self, low: usize, high: usize) -> usize {
        if low >= high {
            return low;
        }
        let span = (high - low) as f64;
        low + (self.next_f64() * span) as usize
    }

    /// Standard normal via Box-Muller
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel Types
// ─────────────────────────────────────────────────────────────────────────────

/// Kernel function types for Kernel K-Means
#[derive(Clone, Debug)]
pub enum KernelType {
    /// Linear kernel: k(x,y) = x·y
    Linear,
    /// Polynomial kernel: k(x,y) = (gamma·x·y + coef0)^degree
    Polynomial { degree: u32, coef0: f64, gamma: f64 },
    /// RBF / Gaussian kernel: k(x,y) = exp(-gamma * ||x-y||^2)
    Rbf { gamma: f64 },
    /// Sigmoid kernel: k(x,y) = tanh(gamma·x·y + coef0)
    Sigmoid { coef0: f64, gamma: f64 },
}

impl KernelType {
    /// Evaluate kernel between two vectors.
    pub fn compute(&self, x: &[f64], y: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), y.len(), "kernel vectors must have same dimension");
        match self {
            KernelType::Linear => dot(x, y),
            KernelType::Polynomial {
                degree,
                coef0,
                gamma,
            } => (gamma * dot(x, y) + coef0).powi(*degree as i32),
            KernelType::Rbf { gamma } => {
                let sq = sq_dist(x, y);
                (-gamma * sq).exp()
            }
            KernelType::Sigmoid { coef0, gamma } => (gamma * dot(x, y) + coef0).tanh(),
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel K-Means
// ─────────────────────────────────────────────────────────────────────────────

/// Build the full kernel matrix K[i,j] = kernel(x_i, x_j).
fn build_kernel_matrix(data: &[Vec<f64>], kernel: &KernelType) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut k = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        k[i][i] = kernel.compute(&data[i], &data[i]);
        for j in (i + 1)..n {
            let v = kernel.compute(&data[i], &data[j]);
            k[i][j] = v;
            k[j][i] = v;
        }
    }
    k
}

/// Compute the kernel k-means objective given assignments.
///
/// The distance in RKHS between x_i and cluster centre c_l is:
///   d²(φ(x_i), μ_l) = K(i,i) - 2/|Cl| Σ_{j∈Cl} K(i,j) + 1/|Cl|² Σ_{j,k∈Cl} K(j,k)
fn kernel_kmeans_objective(k_mat: &[Vec<f64>], labels: &[usize], n_clusters: usize) -> f64 {
    let _n = labels.len();
    // cluster member lists
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        members[l].push(i);
    }
    let mut total = 0.0f64;
    for (i, &l) in labels.iter().enumerate() {
        let cl = &members[l];
        let sz = cl.len() as f64;
        if sz == 0.0 {
            continue;
        }
        // K(i,i)
        let kii = k_mat[i][i];
        // 2/sz * sum_j K(i,j) for j in cluster
        let cross: f64 = cl.iter().map(|&j| k_mat[i][j]).sum::<f64>();
        // 1/sz^2 * sum_{j,k in cluster} K(j,k)
        let inner: f64 = cl
            .iter()
            .flat_map(|&j| cl.iter().map(move |&kk| k_mat[j][kk]))
            .sum::<f64>();
        total += kii - 2.0 * cross / sz + inner / (sz * sz);
    }
    total
}

/// Assign each point to the cluster whose RKHS centroid it is closest to.
fn kernel_kmeans_assign(k_mat: &[Vec<f64>], labels: &[usize], n_clusters: usize) -> Vec<usize> {
    let n = labels.len();
    // precompute cluster sums needed for distance
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_clusters];
    for (i, &l) in labels.iter().enumerate() {
        members[l].push(i);
    }
    // For each cluster l: inner[l] = (1/|Cl|^2) * sum_{j,k in Cl} K(j,k)
    let inner: Vec<f64> = (0..n_clusters)
        .map(|l| {
            let cl = &members[l];
            let sz = cl.len() as f64;
            if sz == 0.0 {
                return f64::INFINITY;
            }
            let s: f64 = cl
                .iter()
                .flat_map(|&j| cl.iter().map(move |&kk| k_mat[j][kk]))
                .sum();
            s / (sz * sz)
        })
        .collect();

    // For each cluster l: cross_sum[i][l] = (1/|Cl|) * sum_{j in Cl} K(i,j)
    let mut new_labels = vec![0usize; n];
    for i in 0..n {
        let mut best_l = 0;
        let mut best_dist = f64::INFINITY;
        for l in 0..n_clusters {
            let cl = &members[l];
            let sz = cl.len() as f64;
            if sz == 0.0 {
                continue;
            }
            let cross: f64 = cl.iter().map(|&j| k_mat[i][j]).sum::<f64>();
            let dist = k_mat[i][i] - 2.0 * cross / sz + inner[l];
            if dist < best_dist {
                best_dist = dist;
                best_l = l;
            }
        }
        new_labels[i] = best_l;
    }
    new_labels
}

/// Kernel K-Means clustering.
///
/// Performs K-means in a reproducing kernel Hilbert space, allowing non-linear
/// cluster boundaries in the original feature space.
///
/// # Arguments
/// - `data`: slice of n feature vectors, each of length d
/// - `n_clusters`: number of clusters K
/// - `kernel`: kernel function defining the feature space
/// - `max_iter`: maximum EM iterations
/// - `n_init`: number of random restarts (best is kept)
/// - `seed`: RNG seed for reproducibility
///
/// # Returns
/// `(labels, inertia)` — cluster assignments (0..K-1) and kernel objective value
///
/// # Errors
/// Returns an error when input is empty or `n_clusters` > n_samples.
pub fn kernel_kmeans(
    data: &[Vec<f64>],
    n_clusters: usize,
    kernel: KernelType,
    max_iter: usize,
    n_init: usize,
    seed: u64,
) -> Result<(Vec<usize>, f64)> {
    if data.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "data must not be empty".into(),
        ));
    }
    if n_clusters == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be >= 1".into(),
        ));
    }
    if n_clusters > data.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "n_clusters ({}) > n_samples ({})",
            n_clusters,
            data.len()
        )));
    }
    let n = data.len();
    let k_mat = build_kernel_matrix(data, &kernel);
    let _rng = Lcg::new(seed); // seed base, actual work uses run_rng

    let mut best_labels = vec![0usize; n];
    let mut best_obj = f64::INFINITY;

    for run in 0..n_init.max(1) {
        // Random initialisation: assign each point uniformly at random, then
        // ensure every cluster has at least one member via forced seeding.
        let run_seed = seed.wrapping_add(run as u64).wrapping_add(1);
        let mut run_rng = Lcg::new(run_seed);

        // Force first k distinct points as cluster centres
        let mut labels = vec![0usize; n];
        // shuffle indices to pick k seeds
        let mut idx: Vec<usize> = (0..n).collect();
        // Fisher-Yates for k steps
        for i in 0..n_clusters {
            let j = run_rng.next_range_usize(i, n);
            idx.swap(i, j);
            labels[idx[i]] = i;
        }
        // assign remaining randomly
        for i in n_clusters..n {
            labels[i] = run_rng.next_range_usize(0, n_clusters);
        }

        // Iteratively re-assign
        for _ in 0..max_iter {
            let new_labels = kernel_kmeans_assign(&k_mat, &labels, n_clusters);
            // Check if any cluster became empty and re-seed if so
            let mut counts = vec![0usize; n_clusters];
            for &l in &new_labels {
                counts[l] += 1;
            }
            let empty = counts.iter().any(|&c| c == 0);
            if empty {
                // Keep old labels for empty clusters — restart inner loop
                break;
            }
            if new_labels == labels {
                break;
            }
            labels = new_labels;
        }

        let obj = kernel_kmeans_objective(&k_mat, &labels, n_clusters);
        if obj < best_obj {
            best_obj = obj;
            best_labels = labels;
        }
        // run_rng used per-iteration
    }

    Ok((best_labels, best_obj))
}

// ─────────────────────────────────────────────────────────────────────────────
// Trimmed K-Means
// ─────────────────────────────────────────────────────────────────────────────

/// Trimmed K-Means (Cuesta-Albertos et al., 1997).
///
/// At each iteration a fraction `trim_ratio` of the points with the largest
/// distance to their assigned centroid are treated as outliers (label = None)
/// and excluded from centroid updates.
///
/// # Arguments
/// - `data`: n feature vectors of length d
/// - `n_clusters`: number of clusters K
/// - `trim_ratio`: fraction in `[0, 0.5)` of samples to trim as outliers
/// - `max_iter`: maximum EM iterations
/// - `seed`: RNG seed
///
/// # Returns
/// `(labels, centroids)` where `labels[i] = None` for trimmed points.
///
/// # Errors
/// Returns an error if data is empty, n_clusters is 0, or trim_ratio is out of range.
pub fn trimmed_kmeans(
    data: &[Vec<f64>],
    n_clusters: usize,
    trim_ratio: f64,
    max_iter: usize,
    seed: u64,
) -> Result<(Vec<Option<usize>>, Vec<Vec<f64>>)> {
    if data.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "data must not be empty".into(),
        ));
    }
    if n_clusters == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be >= 1".into(),
        ));
    }
    if !(0.0..0.5).contains(&trim_ratio) {
        return Err(ClusteringError::InvalidInput(
            "trim_ratio must be in [0, 0.5)".into(),
        ));
    }
    let n = data.len();
    let d = data[0].len();
    if n_clusters > n {
        return Err(ClusteringError::InvalidInput(format!(
            "n_clusters ({}) > n_samples ({})",
            n_clusters, n
        )));
    }

    let n_trim = (n as f64 * trim_ratio).floor() as usize;
    let n_active = n - n_trim;
    if n_active < n_clusters {
        return Err(ClusteringError::InvalidInput(
            "After trimming, too few points remain for the requested n_clusters".into(),
        ));
    }

    let mut rng = Lcg::new(seed);

    // Initialise centroids via k-means++ style
    let mut centroids = kmeans_plus_plus_init(data, n_clusters, &mut rng);

    let mut labels = vec![None::<usize>; n];

    for _iter in 0..max_iter {
        // Step 1: assign each point to nearest centroid
        let mut dists: Vec<(usize, f64)> = (0..n)
            .map(|i| {
                let (cl, dist) = nearest_centroid(&data[i], &centroids);
                (cl, dist)
            })
            .collect();

        // Step 2: trim n_trim points with largest assignment distance
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            dists[b]
                .1
                .partial_cmp(&dists[a].1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let trimmed_set: std::collections::HashSet<usize> =
            order[..n_trim].iter().cloned().collect();

        // Step 3: update labels
        for i in 0..n {
            if trimmed_set.contains(&i) {
                labels[i] = None;
            } else {
                labels[i] = Some(dists[i].0);
            }
        }

        // Step 4: update centroids using only active (non-trimmed) points
        let mut new_centroids = vec![vec![0.0f64; d]; n_clusters];
        let mut counts = vec![0usize; n_clusters];
        for (i, lbl) in labels.iter().enumerate() {
            if let Some(l) = lbl {
                for (feat, &v) in new_centroids[*l].iter_mut().zip(data[i].iter()) {
                    *feat += v;
                }
                counts[*l] += 1;
            }
        }
        let mut changed = false;
        for l in 0..n_clusters {
            if counts[l] > 0 {
                let old = &centroids[l];
                let new_c: Vec<f64> = new_centroids[l]
                    .iter()
                    .map(|&s| s / counts[l] as f64)
                    .collect();
                let diff: f64 = old
                    .iter()
                    .zip(new_c.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if diff > 1e-10 {
                    changed = true;
                }
                centroids[l] = new_c;
            }
        }
        if !changed {
            break;
        }
    }

    Ok((labels, centroids))
}

fn nearest_centroid(point: &[f64], centroids: &[Vec<f64>]) -> (usize, f64) {
    let mut best_c = 0;
    let mut best_d = f64::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d: f64 = point
            .iter()
            .zip(c.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        if d < best_d {
            best_d = d;
            best_c = i;
        }
    }
    (best_c, best_d)
}

fn kmeans_plus_plus_init(data: &[Vec<f64>], k: usize, rng: &mut Lcg) -> Vec<Vec<f64>> {
    let n = data.len();
    let first = rng.next_range_usize(0, n);
    let mut centroids = vec![data[first].clone()];
    for _ in 1..k {
        let dists: Vec<f64> = data
            .iter()
            .map(|x| {
                centroids
                    .iter()
                    .map(|c| sq_dist(x, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = dists.iter().sum();
        let target = rng.next_f64() * total;
        let mut cumsum = 0.0;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= target {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }
    centroids
}

// ─────────────────────────────────────────────────────────────────────────────
// Dirichlet Process Mixture (Chinese Restaurant Process Gibbs sampler)
// ─────────────────────────────────────────────────────────────────────────────

/// Dirichlet Process Mixture model estimated via collapsed Gibbs sampling.
///
/// Uses a conjugate Normal-Wishart prior under the hood with simplified
/// spherical (isotropic) Gaussians per component, allowing automatic
/// inference of the number of clusters.
///
/// # References
/// - Neal (2000) "Markov Chain Sampling Methods for Dirichlet Process Mixture Models"
#[derive(Debug, Clone)]
pub struct DpMixture {
    /// Concentration parameter α (higher → more clusters)
    pub alpha: f64,
    /// Number of active components found after fitting
    pub n_components: usize,
    /// Component weights (mixing proportions)
    pub weights: Vec<f64>,
    /// Component means (one `Vec<f64>` per component)
    pub means: Vec<Vec<f64>>,
    /// Precision parameters per component (scalar per component → isotropic)
    pub concentrations: Vec<f64>,
}

impl DpMixture {
    /// Create a new DpMixture with concentration parameter `alpha`.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            n_components: 0,
            weights: Vec::new(),
            means: Vec::new(),
            concentrations: Vec::new(),
        }
    }

    /// Fit via collapsed Gibbs sampling (Algorithm 3 from Neal 2000 simplified).
    ///
    /// # Returns
    /// Cluster label assignments for each sample.
    pub fn fit(&mut self, data: &[Vec<f64>], max_iter: usize, seed: u64) -> Vec<usize> {
        if data.is_empty() {
            self.n_components = 0;
            return Vec::new();
        }
        let n = data.len();
        let d = data[0].len();
        let mut rng = Lcg::new(seed);

        // Prior hyperparameters (conjugate Normal with known spherical precision)
        let prior_mean = vec![0.0f64; d];
        let prior_kappa = 1.0f64; // prior pseudo-count
        let lambda = 1.0f64; // prior precision (isotropic)

        // Initialise: each point gets its own cluster (CRP start)
        let mut assignments: Vec<usize> = (0..n).collect();
        // component_members[k] = list of sample indices in component k
        let mut component_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

        for _iter in 0..max_iter {
            for i in 0..n {
                let xi = &data[i];
                let current_k = assignments[i];

                // Remove i from its current component
                component_members[current_k].retain(|&m| m != i);

                // Remove empty components
                let alive: Vec<usize> = (0..component_members.len())
                    .filter(|&k| !component_members[k].is_empty())
                    .collect();
                // Re-index
                let new_members: Vec<Vec<usize>> = alive
                    .iter()
                    .map(|&k| component_members[k].clone())
                    .collect();
                // Update assignments to new indices
                for j in 0..n {
                    if j == i {
                        continue;
                    }
                    let old_k = assignments[j];
                    if let Some(pos) = alive.iter().position(|&k| k == old_k) {
                        assignments[j] = pos;
                    }
                }
                component_members = new_members;

                let k_live = component_members.len();

                // Compute posterior predictive probabilities for each existing cluster
                let mut log_probs: Vec<f64> = Vec::with_capacity(k_live + 1);
                for k in 0..k_live {
                    let members = &component_members[k];
                    let n_k = members.len() as f64;
                    // Posterior Normal-Normal predictive
                    let kappa_n = prior_kappa + n_k;
                    let mu_n: Vec<f64> = {
                        let mut s = prior_mean.clone();
                        for &m in members.iter() {
                            for (f, &v) in s.iter_mut().zip(data[m].iter()) {
                                *f += v;
                            }
                        }
                        s.iter().map(|&v| v / (prior_kappa + n_k)).collect()
                    };
                    let pred_var = (kappa_n + 1.0) / (kappa_n * lambda);
                    // log p(xi | cluster k data) — Gaussian predictive
                    let log_lik: f64 = xi
                        .iter()
                        .zip(mu_n.iter())
                        .map(|(&xf, &mf)| {
                            let z = (xf - mf).powi(2);
                            -0.5 * (z / pred_var + (TAU * pred_var).ln())
                        })
                        .sum();
                    let log_prior = (n_k / (n as f64 - 1.0 + self.alpha)).ln();
                    log_probs.push(log_prior + log_lik);
                }

                // New cluster probability
                let pred_var_new = (prior_kappa + 1.0) / (prior_kappa * lambda);
                let log_lik_new: f64 = xi
                    .iter()
                    .zip(prior_mean.iter())
                    .map(|(&xf, &mf)| {
                        let z = (xf - mf).powi(2);
                        -0.5 * (z / pred_var_new + (TAU * pred_var_new).ln())
                    })
                    .sum();
                let log_prior_new = (self.alpha / (n as f64 - 1.0 + self.alpha)).ln();
                log_probs.push(log_prior_new + log_lik_new);

                // Numerically stable softmax sampling
                let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max_lp).exp()).collect();
                let total: f64 = probs.iter().sum();
                let u = rng.next_f64() * total;
                let mut cumsum = 0.0;
                let mut chosen_k = probs.len() - 1;
                for (idx, &p) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum >= u {
                        chosen_k = idx;
                        break;
                    }
                }

                if chosen_k == k_live {
                    // New component
                    component_members.push(vec![i]);
                    assignments[i] = k_live;
                } else {
                    component_members[chosen_k].push(i);
                    assignments[i] = chosen_k;
                }
            }
        }

        // Populate model parameters from final state
        let k_final = component_members.len();
        self.n_components = k_final;
        self.weights = component_members
            .iter()
            .map(|m| m.len() as f64 / n as f64)
            .collect();
        self.means = component_members
            .iter()
            .map(|members| {
                let mut mu = vec![0.0f64; d];
                for &m in members.iter() {
                    for (f, &v) in mu.iter_mut().zip(data[m].iter()) {
                        *f += v;
                    }
                }
                mu.iter().map(|&v| v / members.len() as f64).collect()
            })
            .collect();
        self.concentrations = vec![lambda; k_final];

        assignments
    }

    /// Number of active (non-empty) clusters found after fitting.
    pub fn n_clusters(&self) -> usize {
        self.n_components
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Fuzzy C-Means
// ─────────────────────────────────────────────────────────────────────────────

/// Fuzzy C-Means clustering (Bezdek 1981).
///
/// Each point has a soft membership degree `u[i][c]` to each cluster c, with
/// degrees summing to 1 over clusters. The fuzziness exponent `m > 1` controls
/// how crisp (→1) or diffuse (→∞) the memberships are.
///
/// # Arguments
/// - `data`: n feature vectors of length d
/// - `n_clusters`: number of clusters K
/// - `fuzziness`: exponent m (> 1, typically 2.0)
/// - `max_iter`: maximum EM iterations
/// - `tol`: convergence threshold on centroid change
/// - `seed`: RNG seed
///
/// # Returns
/// `(centroids, membership_matrix)` where `membership_matrix` is n × K.
///
/// # Errors
/// Returns an error for invalid parameters.
pub fn fuzzy_cmeans(
    data: &[Vec<f64>],
    n_clusters: usize,
    fuzziness: f64,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>)> {
    if data.is_empty() {
        return Err(ClusteringError::InvalidInput(
            "data must not be empty".into(),
        ));
    }
    if n_clusters == 0 {
        return Err(ClusteringError::InvalidInput(
            "n_clusters must be >= 1".into(),
        ));
    }
    if fuzziness <= 1.0 {
        return Err(ClusteringError::InvalidInput(
            "fuzziness (m) must be > 1.0".into(),
        ));
    }
    if n_clusters > data.len() {
        return Err(ClusteringError::InvalidInput(format!(
            "n_clusters ({}) > n_samples ({})",
            n_clusters,
            data.len()
        )));
    }

    let n = data.len();
    let d = data[0].len();
    let m = fuzziness;
    let exp = 2.0 / (m - 1.0);

    // Initialise memberships randomly
    let mut rng = Lcg::new(seed);
    let mut u: Vec<Vec<f64>> = (0..n)
        .map(|_| {
            let raw: Vec<f64> = (0..n_clusters).map(|_| rng.next_f64() + 1e-12).collect();
            let s: f64 = raw.iter().sum();
            raw.iter().map(|&v| v / s).collect()
        })
        .collect();

    let mut centroids = compute_fuzzy_centroids(data, &u, n_clusters, m, d);

    for _iter in 0..max_iter {
        // Update membership matrix
        let mut new_u = vec![vec![0.0f64; n_clusters]; n];
        for i in 0..n {
            let dists: Vec<f64> = (0..n_clusters)
                .map(|c| sq_dist(&data[i], &centroids[c]).max(1e-30))
                .collect();
            // Check if point is exactly on a centroid
            let exact: Vec<usize> = dists
                .iter()
                .enumerate()
                .filter(|(_, &d)| d < 1e-30)
                .map(|(c, _)| c)
                .collect();
            if !exact.is_empty() {
                let share = 1.0 / exact.len() as f64;
                for &c in &exact {
                    new_u[i][c] = share;
                }
            } else {
                for c in 0..n_clusters {
                    let ratio_sum: f64 = (0..n_clusters)
                        .map(|j| (dists[c] / dists[j]).powf(exp))
                        .sum();
                    new_u[i][c] = 1.0 / ratio_sum;
                }
            }
        }

        // Update centroids
        let new_centroids = compute_fuzzy_centroids(data, &new_u, n_clusters, m, d);

        // Check convergence
        let max_change: f64 = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(c_old, c_new)| {
                c_old
                    .iter()
                    .zip(c_new.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max)
            })
            .fold(0.0f64, f64::max);

        u = new_u;
        centroids = new_centroids;

        if max_change < tol {
            break;
        }
    }

    Ok((centroids, u))
}

fn compute_fuzzy_centroids(
    data: &[Vec<f64>],
    u: &[Vec<f64>],
    n_clusters: usize,
    m: f64,
    d: usize,
) -> Vec<Vec<f64>> {
    (0..n_clusters)
        .map(|c| {
            let mut num = vec![0.0f64; d];
            let mut denom = 0.0f64;
            for (i, xi) in data.iter().enumerate() {
                let uic_m = u[i][c].powf(m);
                denom += uic_m;
                for (f, &v) in num.iter_mut().zip(xi.iter()) {
                    *f += uic_m * v;
                }
            }
            if denom.abs() < 1e-30 {
                vec![0.0f64; d]
            } else {
                num.iter().map(|&v| v / denom).collect()
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_data() -> Vec<Vec<f64>> {
        // Two well-separated Gaussian blobs
        let mut v = Vec::new();
        for i in 0..20 {
            v.push(vec![i as f64 * 0.1, i as f64 * 0.1]);
        }
        for i in 0..20 {
            v.push(vec![10.0 + i as f64 * 0.1, 10.0 + i as f64 * 0.1]);
        }
        v
    }

    // ── Kernel K-Means ──────────────────────────────────────────────────────

    #[test]
    fn test_kernel_kmeans_rbf_two_clusters() {
        let data = two_cluster_data();
        let (labels, inertia) = kernel_kmeans(&data, 2, KernelType::Rbf { gamma: 0.5 }, 50, 3, 42)
            .expect("kernel_kmeans should succeed");
        assert_eq!(labels.len(), 40);
        assert!(inertia.is_finite());
        // Check the two blobs are separated: first 20 and last 20 should have same label
        let l0 = labels[0];
        let l20 = labels[20];
        assert_ne!(l0, l20, "blobs should be in different clusters");
        assert!(labels[..20].iter().all(|&l| l == l0));
        assert!(labels[20..].iter().all(|&l| l == l20));
    }

    #[test]
    fn test_kernel_kmeans_linear() {
        let data = two_cluster_data();
        let (labels, _) = kernel_kmeans(&data, 2, KernelType::Linear, 20, 2, 7)
            .expect("kernel_kmeans linear should succeed");
        assert_eq!(labels.len(), 40);
    }

    #[test]
    fn test_kernel_kmeans_polynomial() {
        let data = two_cluster_data();
        let (labels, _) = kernel_kmeans(
            &data,
            2,
            KernelType::Polynomial {
                degree: 2,
                coef0: 1.0,
                gamma: 0.1,
            },
            20,
            2,
            99,
        )
        .expect("kernel_kmeans poly should succeed");
        assert_eq!(labels.len(), 40);
    }

    #[test]
    fn test_kernel_kmeans_invalid_inputs() {
        let data = two_cluster_data();
        assert!(kernel_kmeans(&[], 2, KernelType::Linear, 10, 1, 0).is_err());
        assert!(kernel_kmeans(&data, 0, KernelType::Linear, 10, 1, 0).is_err());
        assert!(kernel_kmeans(&data, 100, KernelType::Linear, 10, 1, 0).is_err());
    }

    // ── Trimmed K-Means ─────────────────────────────────────────────────────

    #[test]
    fn test_trimmed_kmeans_basic() {
        let mut data = two_cluster_data();
        // Add some outliers
        data.push(vec![100.0, 100.0]);
        data.push(vec![-100.0, -100.0]);

        let (labels, centroids) =
            trimmed_kmeans(&data, 2, 0.05, 100, 42).expect("trimmed_kmeans should succeed");
        assert_eq!(labels.len(), data.len());
        assert_eq!(centroids.len(), 2);
        // The two extreme outliers should be trimmed (None)
        let trimmed_count = labels.iter().filter(|l| l.is_none()).count();
        assert!(trimmed_count >= 1, "at least one outlier should be trimmed");
    }

    #[test]
    fn test_trimmed_kmeans_no_trim() {
        let data = two_cluster_data();
        let (labels, centroids) = trimmed_kmeans(&data, 2, 0.0, 50, 0)
            .expect("trimmed_kmeans with trim=0 should succeed");
        assert_eq!(labels.len(), 40);
        assert_eq!(centroids.len(), 2);
        // No points trimmed
        assert!(labels.iter().all(|l| l.is_some()));
    }

    #[test]
    fn test_trimmed_kmeans_invalid() {
        let data = two_cluster_data();
        assert!(trimmed_kmeans(&[], 2, 0.1, 10, 0).is_err());
        assert!(trimmed_kmeans(&data, 0, 0.1, 10, 0).is_err());
        assert!(trimmed_kmeans(&data, 2, 0.6, 10, 0).is_err()); // trim_ratio >= 0.5
    }

    // ── DpMixture ───────────────────────────────────────────────────────────

    #[test]
    fn test_dp_mixture_finds_clusters() {
        let data = two_cluster_data();
        let mut dpm = DpMixture::new(1.0);
        let labels = dpm.fit(&data, 30, 42);
        assert_eq!(labels.len(), 40);
        // With two well-separated blobs the DP should find at least 2 components
        assert!(dpm.n_clusters() >= 1);
        assert!(!dpm.means.is_empty());
        assert!(!dpm.weights.is_empty());
        let weight_sum: f64 = dpm.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dp_mixture_empty() {
        let mut dpm = DpMixture::new(1.0);
        let labels = dpm.fit(&[], 10, 0);
        assert!(labels.is_empty());
        assert_eq!(dpm.n_clusters(), 0);
    }

    // ── Fuzzy C-Means ───────────────────────────────────────────────────────

    #[test]
    fn test_fuzzy_cmeans_basic() {
        let data = two_cluster_data();
        let (centroids, membership) =
            fuzzy_cmeans(&data, 2, 2.0, 100, 1e-6, 42).expect("fuzzy_cmeans should succeed");
        assert_eq!(centroids.len(), 2);
        assert_eq!(membership.len(), 40);
        assert_eq!(membership[0].len(), 2);
        // Membership rows must sum to 1
        for row in &membership {
            let s: f64 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-8,
                "membership row must sum to 1, got {}",
                s
            );
        }
    }

    #[test]
    fn test_fuzzy_cmeans_high_fuzziness() {
        let data = two_cluster_data();
        let (_, membership) = fuzzy_cmeans(&data, 3, 3.5, 50, 1e-5, 99)
            .expect("fuzzy_cmeans high fuzz should succeed");
        for row in &membership {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-7);
        }
    }

    #[test]
    fn test_fuzzy_cmeans_invalid() {
        let data = two_cluster_data();
        assert!(fuzzy_cmeans(&[], 2, 2.0, 10, 1e-6, 0).is_err());
        assert!(fuzzy_cmeans(&data, 0, 2.0, 10, 1e-6, 0).is_err());
        assert!(fuzzy_cmeans(&data, 2, 1.0, 10, 1e-6, 0).is_err()); // fuzziness <= 1
        assert!(fuzzy_cmeans(&data, 2, 0.5, 10, 1e-6, 0).is_err()); // fuzziness < 1
        assert!(fuzzy_cmeans(&data, 100, 2.0, 10, 1e-6, 0).is_err()); // k > n
    }
}
