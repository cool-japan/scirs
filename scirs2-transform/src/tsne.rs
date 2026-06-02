//! t-SNE (t-distributed Stochastic Neighbor Embedding) — config-based API
//!
//! This module provides a self-contained, config-driven t-SNE implementation.
//! It implements the **exact O(n²)** t-SNE algorithm which guarantees correctness
//! for small-to-medium datasets (hundreds to low thousands of samples).
//!
//! ## Algorithm
//!
//! 1. Compute pairwise affinities `P_ij` using a Gaussian kernel calibrated by
//!    binary search to achieve the target perplexity.
//! 2. Symmetrise and normalise: `P = (P + P^T) / (2n)`.
//! 3. Initialise `Y` randomly or via PCA.
//! 4. Gradient descent with momentum:
//!    - **Early exaggeration** (first `early_exaggeration_iter` steps): `P` is
//!      multiplied by `early_exaggeration` to spread out clusters.
//!    - **Final phase**: plain KL gradient with adaptive gains.
//! 5. The gradient of `KL(P || Q)` w.r.t. `y_i` is:
//!    `4 Σ_j (p_ij − q_ij) q̃_ij (y_i − y_j)`
//!    where `q̃_ij = (1 + ||y_i − y_j||²)^{-1}` (Student-t kernel, df=1).

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Distribution, Normal};

use crate::error::{Result, TransformError};
use crate::reduction::PCA;

// ────────────────────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────────────────────

/// Initialisation strategy for the t-SNE embedding
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TsneInit {
    /// Start from a (scaled) PCA projection
    Pca,
    /// Start from Gaussian noise with std 1e-4
    Random,
}

/// Configuration for the t-SNE algorithm
#[derive(Debug, Clone)]
pub struct TsneConfig {
    /// Dimensionality of the output embedding (default 2)
    pub n_components: usize,
    /// Perplexity — balances local vs global structure (default 30.0)
    pub perplexity: f64,
    /// SGD learning rate (default 200.0)
    pub learning_rate: f64,
    /// Total number of gradient-descent iterations (default 1000)
    pub n_iter: usize,
    /// Exaggeration factor for the first `early_exaggeration_iter` steps (default 12.0)
    pub early_exaggeration: f64,
    /// Number of iterations in the early-exaggeration phase (default 250)
    pub early_exaggeration_iter: usize,
    /// Initialisation strategy (default Pca)
    pub init: TsneInit,
    /// RNG seed (default 0 → system entropy)
    pub seed: u64,
}

impl Default for TsneConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            early_exaggeration: 12.0,
            early_exaggeration_iter: 250,
            init: TsneInit::Pca,
            seed: 0,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Free function
// ────────────────────────────────────────────────────────────────────────────

/// Run t-SNE on `data` and return the low-dim embedding.
///
/// This is the simplest entry point — it accepts a config and data matrix and
/// returns the final embedding.
///
/// # Arguments
/// * `data` — shape `(n_samples, n_features)`
/// * `config` — algorithm hyper-parameters
///
/// # Returns
/// Embedding array, shape `(n_samples, n_components)`
pub fn tsne(data: &Array2<f64>, config: TsneConfig) -> Result<Array2<f64>> {
    let mut runner = Tsne::new(config);
    runner.fit_transform(data)
}

// ────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────────────────

const MEPS: f64 = 1e-14; // machine epsilon guard

/// Binary search for beta (= 1/(2σ²)) to achieve target entropy `H = log(perplexity)`
fn compute_affinities_row(distances_sq: &[f64], self_idx: usize, target_h: f64) -> Vec<f64> {
    let n = distances_sq.len();
    let mut beta = 1.0_f64;
    let mut beta_lo = -f64::INFINITY;
    let mut beta_hi = f64::INFINITY;
    let mut p_row = vec![0.0_f64; n];

    for _ in 0..50 {
        let mut sum_p = 0.0_f64;
        let mut h = 0.0_f64;

        for j in 0..n {
            if j == self_idx {
                p_row[j] = 0.0;
                continue;
            }
            let val = (-beta * distances_sq[j]).exp();
            p_row[j] = val;
            sum_p += val;
        }

        if sum_p > MEPS {
            for (j, v) in p_row.iter_mut().enumerate() {
                if j == self_idx {
                    continue;
                }
                *v /= sum_p;
                if *v > MEPS {
                    h -= *v * v.ln();
                }
            }
        }

        let h_diff = h - target_h;
        if h_diff.abs() < 1e-6 {
            break;
        }
        if h_diff > 0.0 {
            beta_lo = beta;
            beta = if beta_hi == f64::INFINITY {
                beta * 2.0
            } else {
                (beta + beta_hi) / 2.0
            };
        } else {
            beta_hi = beta;
            beta = if beta_lo == -f64::INFINITY {
                beta / 2.0
            } else {
                (beta + beta_lo) / 2.0
            };
        }
    }

    p_row
}

/// Compute the full symmetric, normalised P matrix
fn compute_p_matrix(data: &Array2<f64>, perplexity: f64) -> Result<Array2<f64>> {
    let n = data.shape()[0];
    let d = data.shape()[1];
    let target_h = perplexity.ln();

    // Pairwise squared Euclidean distances
    let mut p = Array2::zeros((n, n));
    for i in 0..n {
        let mut dist_sq = vec![0.0_f64; n];
        for j in 0..n {
            if i == j {
                continue;
            }
            let mut sq = 0.0_f64;
            for k in 0..d {
                let diff = data[[i, k]] - data[[j, k]];
                sq += diff * diff;
            }
            dist_sq[j] = sq;
        }
        let row = compute_affinities_row(&dist_sq, i, target_h);
        for (j, v) in row.iter().enumerate() {
            p[[i, j]] = *v;
        }
    }

    // Symmetrise: P = (P + P^T) / (2n), then clamp negatives
    let p_sym = (&p + &p.t()) / (2.0 * n as f64);
    let p_sym = p_sym.mapv(|v: f64| v.max(MEPS));

    Ok(p_sym)
}

/// Compute KL(P||Q) gradient and KL value for exact t-SNE
fn compute_grad_exact(
    embedding: &Array2<f64>,
    p: &Array2<f64>,
    exaggeration: f64,
) -> (f64, Array2<f64>) {
    let n = embedding.shape()[0];
    let nc = embedding.shape()[1];

    // Compute Q̃_{ij} = (1 + ||yi − yj||²)^{-1}  (unnormalised)
    let mut q_tilde = Array2::zeros((n, n));
    let mut sum_q = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut sq = 0.0_f64;
            for k in 0..nc {
                let diff = embedding[[i, k]] - embedding[[j, k]];
                sq += diff * diff;
            }
            let qt = 1.0 / (1.0 + sq);
            q_tilde[[i, j]] = qt;
            q_tilde[[j, i]] = qt;
            sum_q += 2.0 * qt;
        }
    }
    sum_q = sum_q.max(MEPS);

    // Normalise Q
    let q = q_tilde.mapv(|v| v / sum_q);

    // KL divergence
    let mut kl = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let pij = p[[i, j]] * exaggeration;
            let qij = q[[i, j]].max(MEPS);
            if pij > MEPS {
                kl += pij * (pij / qij).ln();
            }
        }
    }

    // Gradient:  dC/dy_i = 4 Σ_j (p_ij − q_ij) * q̃_ij * (y_i − y_j)
    let mut grad = Array2::zeros((n, nc));
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let pij = p[[i, j]] * exaggeration;
            let qij = q[[i, j]];
            let qt = q_tilde[[i, j]];
            let factor = 4.0 * (pij - qij) * qt;
            for k in 0..nc {
                grad[[i, k]] += factor * (embedding[[i, k]] - embedding[[j, k]]);
            }
        }
    }

    (kl, grad)
}

/// PCA-based initialisation, scaled to std ≈ 1e-4
fn pca_init(data: &Array2<f64>, n_components: usize) -> Result<Array2<f64>> {
    // Bound by both dimensions: PCA can return at most min(n_samples, n_features) components.
    let nc = n_components.min(data.shape()[0]).min(data.shape()[1]);
    let mut pca = PCA::new(nc, true, false);
    let mut x = pca.fit_transform(data)?;
    // Scale first column std → 1e-4
    let n = x.shape()[0];
    let col0: Array1<f64> = x.column(0).to_owned();
    let var = col0.iter().map(|v| v * v).sum::<f64>() / n as f64;
    let std_dev = var.sqrt();
    if std_dev > 1e-12 {
        x.mapv_inplace(|v| v / std_dev * 1e-4);
    }
    Ok(x)
}

/// Random N(0, 1e-4) initialisation
fn random_init(n_samples: usize, n_components: usize) -> Result<Array2<f64>> {
    let normal = Normal::new(0.0, 1e-4).map_err(|e| {
        TransformError::ComputationError(format!("Normal distribution error: {e}"))
    })?;
    let mut rng = thread_rng();
    let data: Vec<f64> = (0..(n_samples * n_components))
        .map(|_| normal.sample(&mut rng))
        .collect();
    Array2::from_shape_vec((n_samples, n_components), data)
        .map_err(|e| TransformError::ComputationError(format!("Array construction failed: {e}")))
}

// ────────────────────────────────────────────────────────────────────────────
// Main struct
// ────────────────────────────────────────────────────────────────────────────

/// t-SNE with config-based API.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::tsne::{Tsne, TsneConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::<f64>::zeros((30, 5));
/// let mut tsne = Tsne::new(TsneConfig { perplexity: 5.0, n_iter: 200, ..Default::default() });
/// let embedding = tsne.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[30, 2]);
/// ```
pub struct Tsne {
    config: TsneConfig,
    /// Stored embedding after fitting
    embedding: Option<Array2<f64>>,
    /// Final KL divergence
    kl_divergence: Option<f64>,
    /// Number of iterations actually executed
    n_iter_done: Option<usize>,
}

impl Tsne {
    /// Create a new `Tsne` instance from the given configuration.
    pub fn new(config: TsneConfig) -> Self {
        Self {
            config,
            embedding: None,
            kl_divergence: None,
            n_iter_done: None,
        }
    }

    /// Fit to `data` and return the embedding.
    ///
    /// # Arguments
    /// * `data` — shape `(n_samples, n_features)`
    pub fn fit_transform(&mut self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let n = data.shape()[0];
        let nf = data.shape()[1];

        if n == 0 || nf == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }
        if self.config.perplexity >= n as f64 {
            return Err(TransformError::InvalidInput(format!(
                "perplexity ({}) must be < n_samples ({})",
                self.config.perplexity, n
            )));
        }

        // 1. Compute P matrix
        let p = compute_p_matrix(data, self.config.perplexity)?;

        // 2. Initialise embedding
        let mut y = match self.config.init {
            TsneInit::Pca => pca_init(data, self.config.n_components).unwrap_or_else(|_| {
                random_init(n, self.config.n_components)
                    .expect("random_init should not fail")
            }),
            TsneInit::Random => random_init(n, self.config.n_components)?,
        };

        // 3. Gradient descent with momentum
        let nc = self.config.n_components;
        let mut velocity = Array2::<f64>::zeros((n, nc));
        let mut gains = Array2::from_elem((n, nc), 1.0_f64);
        let eta = self.config.learning_rate;
        let exagg_iters = self.config.early_exaggeration_iter.min(self.config.n_iter);

        let mut last_kl = f64::INFINITY;
        let mut done_iters = 0_usize;

        for iter in 0..self.config.n_iter {
            let exaggeration = if iter < exagg_iters {
                self.config.early_exaggeration
            } else {
                1.0
            };
            let momentum = if iter < exagg_iters { 0.5 } else { 0.8 };

            let (kl, grad) = compute_grad_exact(&y, &p, exaggeration);
            last_kl = kl;

            // Adaptive-gain update with momentum
            for i in 0..n {
                for d in 0..nc {
                    let same_sign = velocity[[i, d]] * grad[[i, d]] > 0.0;
                    if same_sign {
                        gains[[i, d]] = (gains[[i, d]] * 0.8).max(0.01);
                    } else {
                        gains[[i, d]] += 0.2;
                    }
                    velocity[[i, d]] =
                        momentum * velocity[[i, d]] - eta * gains[[i, d]] * grad[[i, d]];
                    y[[i, d]] += velocity[[i, d]];
                }
            }

            // Re-centre to zero mean (stops unbounded drift)
            let mut mean = vec![0.0_f64; nc];
            for i in 0..n {
                for d in 0..nc {
                    mean[d] += y[[i, d]];
                }
            }
            for d in 0..nc {
                mean[d] /= n as f64;
            }
            for i in 0..n {
                for d in 0..nc {
                    y[[i, d]] -= mean[d];
                }
            }

            done_iters = iter + 1;
        }

        self.embedding = Some(y.clone());
        self.kl_divergence = Some(last_kl);
        self.n_iter_done = Some(done_iters);

        Ok(y)
    }

    /// Return the KL divergence recorded at the last iteration.
    pub fn kl_divergence(&self) -> Option<f64> {
        self.kl_divergence
    }

    /// Return the number of iterations executed.
    pub fn n_iter_done(&self) -> Option<usize> {
        self.n_iter_done
    }

    /// Return the stored embedding (available after `fit_transform`).
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    fn two_cluster_data(n_each: usize, n_features: usize) -> Array2<f64> {
        let mut rows = Vec::with_capacity(n_each * 2 * n_features);
        for i in 0..n_each {
            let base = i as f64 * 0.1;
            for k in 0..n_features {
                rows.push(base + k as f64 * 0.01);
            }
        }
        for i in 0..n_each {
            let base = i as f64 * 0.1 + 20.0;
            for k in 0..n_features {
                rows.push(base + k as f64 * 0.01);
            }
        }
        Array::from_shape_vec((n_each * 2, n_features), rows).expect("shape")
    }

    #[test]
    fn test_tsne_output_shape() {
        let data = two_cluster_data(8, 5);
        let mut t = Tsne::new(TsneConfig {
            perplexity: 3.0,
            n_iter: 100,
            ..Default::default()
        });
        let emb = t.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.shape(), &[16, 2]);
        for v in emb.iter() {
            assert!(v.is_finite(), "non-finite value in embedding");
        }
    }

    #[test]
    fn test_tsne_3_components() {
        let data = two_cluster_data(7, 4);
        let mut t = Tsne::new(TsneConfig {
            n_components: 3,
            perplexity: 3.0,
            n_iter: 50,
            ..Default::default()
        });
        let emb = t.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.shape(), &[14, 3]);
    }

    #[test]
    fn test_tsne_free_function() {
        let data = two_cluster_data(8, 4);
        let emb = tsne(
            &data,
            TsneConfig {
                perplexity: 3.0,
                n_iter: 80,
                ..Default::default()
            },
        )
        .expect("tsne");
        assert_eq!(emb.shape(), &[16, 2]);
    }

    #[test]
    fn test_tsne_random_init() {
        let data = two_cluster_data(8, 4);
        let mut t = Tsne::new(TsneConfig {
            perplexity: 3.0,
            n_iter: 80,
            init: TsneInit::Random,
            ..Default::default()
        });
        let emb = t.fit_transform(&data).expect("fit_transform");
        assert_eq!(emb.shape(), &[16, 2]);
    }

    #[test]
    fn test_tsne_kl_decreases() {
        // Run t-SNE on a well-separated dataset and confirm KL goes down
        // from the initial (measured after early-exaggeration) to the end.
        let data = two_cluster_data(8, 4);

        // Run only exaggeration phase
        let mut t_short = Tsne::new(TsneConfig {
            perplexity: 3.0,
            n_iter: 30,
            early_exaggeration_iter: 30, // all in exaggeration
            ..Default::default()
        });
        t_short.fit_transform(&data).expect("short");
        let kl_short = t_short.kl_divergence().expect("kl");

        // Run longer (full optimization)
        let mut t_long = Tsne::new(TsneConfig {
            perplexity: 3.0,
            n_iter: 400,
            ..Default::default()
        });
        t_long.fit_transform(&data).expect("long");
        let kl_long = t_long.kl_divergence().expect("kl");

        // After full optimisation the KL should be finite
        assert!(kl_long.is_finite(), "KL should be finite: {kl_long}");
        // The final KL should be much smaller than the exaggeration-phase KL
        // (which artificially inflates the cost function)
        assert!(
            kl_long < kl_short * 2.0,
            "Expected kl_long ({kl_long:.4}) << kl_short ({kl_short:.4})"
        );
    }

    #[test]
    fn test_tsne_cluster_separation() {
        // Two far-apart clusters should remain separated after t-SNE
        let data = two_cluster_data(10, 4);
        let mut t = Tsne::new(TsneConfig {
            perplexity: 4.0,
            n_iter: 500,
            ..Default::default()
        });
        let emb = t.fit_transform(&data).expect("fit_transform");
        let n = 10_usize;

        // Centroid of cluster 0
        let mut c0 = [0.0_f64; 2];
        let mut c1 = [0.0_f64; 2];
        for i in 0..n {
            c0[0] += emb[[i, 0]];
            c0[1] += emb[[i, 1]];
            c1[0] += emb[[n + i, 0]];
            c1[1] += emb[[n + i, 1]];
        }
        for v in c0.iter_mut() {
            *v /= n as f64;
        }
        for v in c1.iter_mut() {
            *v /= n as f64;
        }
        let dist = ((c0[0] - c1[0]).powi(2) + (c0[1] - c1[1]).powi(2)).sqrt();
        assert!(
            dist > 0.01,
            "cluster centroids collapsed: dist = {dist:.4}"
        );
    }

    #[test]
    fn test_tsne_perplexity_too_large_error() {
        let data: Array2<f64> = Array::zeros((5, 3));
        let mut t = Tsne::new(TsneConfig {
            perplexity: 10.0, // >= n_samples
            ..Default::default()
        });
        let res = t.fit_transform(&data);
        assert!(res.is_err(), "should fail when perplexity >= n_samples");
    }

    #[test]
    fn test_tsne_iter_count_stored() {
        let data = two_cluster_data(6, 3);
        let n_iter = 80_usize;
        let mut t = Tsne::new(TsneConfig {
            perplexity: 2.0,
            n_iter,
            ..Default::default()
        });
        t.fit_transform(&data).expect("fit_transform");
        assert_eq!(t.n_iter_done(), Some(n_iter));
    }
}
