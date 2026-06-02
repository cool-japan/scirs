//! t-SNE (t-distributed Stochastic Neighbor Embedding) implementation
//!
//! This module provides an implementation of t-SNE, a technique for dimensionality
//! reduction particularly well-suited for visualization of high-dimensional data.
//!
//! t-SNE converts similarities between data points to joint probabilities and tries
//! to minimize the Kullback-Leibler divergence between the joint probabilities of
//! the low-dimensional embedding and the high-dimensional data.
//!
//! ## Features
//!
//! - **Barnes-Hut approximation** for O(N log N) complexity via spatial trees
//! - **Perplexity-based bandwidth selection** using binary search for sigma
//! - **Early exaggeration phase** for better global structure preservation
//! - **Momentum-based gradient descent** with adaptive gains
//! - **Multiple distance metrics**: euclidean, manhattan, cosine, chebyshev
//! - **Sparse kNN affinity** for memory-efficient computation on large datasets
//! - **Multicore support** via rayon parallel iterators

mod spatial_tree;

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::parallel_ops::*;
use scirs2_core::random::Normal;
use scirs2_core::random::RandomExt;

use crate::error::{Result, TransformError};
use crate::reduction::PCA;

use spatial_tree::SpatialTree;

// Constants for numerical stability
const MACHINE_EPSILON: f64 = 1e-14;
const EPSILON: f64 = 1e-7;

/// t-SNE (t-distributed Stochastic Neighbor Embedding) for dimensionality reduction
///
/// t-SNE is a nonlinear dimensionality reduction technique well-suited for
/// embedding high-dimensional data for visualization in a low-dimensional space
/// (typically 2D or 3D). It models each high-dimensional object by a two- or
/// three-dimensional point in such a way that similar objects are modeled by
/// nearby points and dissimilar objects are modeled by distant points with
/// high probability.
///
/// # Example
///
/// ```rust,no_run
/// use scirs2_transform::TSNE;
/// use scirs2_core::ndarray::arr2;
///
/// let data = arr2(&[
///     [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0],
///     [5.0, 5.0], [6.0, 5.0], [5.0, 6.0], [6.0, 6.0],
/// ]);
///
/// let mut tsne = TSNE::new()
///     .with_n_components(2)
///     .with_perplexity(2.0)
///     .with_max_iter(500);
///
/// let embedding = tsne.fit_transform(&data).expect("should succeed");
/// assert_eq!(embedding.shape(), &[8, 2]);
/// ```
pub struct TSNE {
    /// Number of components in the embedded space
    n_components: usize,
    /// Perplexity parameter that balances attention between local and global structure
    perplexity: f64,
    /// Weight of early exaggeration phase
    early_exaggeration: f64,
    /// Learning rate for optimization
    learning_rate: f64,
    /// Maximum number of iterations
    max_iter: usize,
    /// Maximum iterations without progress before early stopping
    n_iter_without_progress: usize,
    /// Minimum gradient norm for convergence
    min_grad_norm: f64,
    /// Method to compute pairwise distances
    metric: String,
    /// Method to perform dimensionality reduction ("exact" or "barnes_hut")
    method: String,
    /// Initialization method ("pca" or "random")
    init: String,
    /// Angle for Barnes-Hut approximation (trade-off between speed and accuracy)
    angle: f64,
    /// Whether to use multicore processing (-1 = all cores, 1 = single)
    n_jobs: i32,
    /// Verbosity level
    verbose: bool,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Degrees of freedom for the t-distribution (default 1.0 = standard Cauchy)
    degrees_of_freedom: Option<f64>,
    /// The embedding vectors
    embedding_: Option<Array2<f64>>,
    /// KL divergence after optimization
    kl_divergence_: Option<f64>,
    /// Total number of iterations run
    n_iter_: Option<usize>,
    /// Effective learning rate used
    learning_rate_: Option<f64>,
}

impl Default for TSNE {
    fn default() -> Self {
        Self::new()
    }
}

impl TSNE {
    /// Creates a new t-SNE instance with default parameters
    pub fn new() -> Self {
        TSNE {
            n_components: 2,
            perplexity: 30.0,
            early_exaggeration: 12.0,
            learning_rate: 200.0,
            max_iter: 1000,
            n_iter_without_progress: 300,
            min_grad_norm: 1e-7,
            metric: "euclidean".to_string(),
            method: "barnes_hut".to_string(),
            init: "pca".to_string(),
            angle: 0.5,
            n_jobs: -1,
            verbose: false,
            random_state: None,
            degrees_of_freedom: None,
            embedding_: None,
            kl_divergence_: None,
            n_iter_: None,
            learning_rate_: None,
        }
    }

    /// Sets the number of components in the embedded space
    pub fn with_n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Sets the perplexity parameter
    pub fn with_perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Sets the early exaggeration factor
    pub fn with_early_exaggeration(mut self, early_exaggeration: f64) -> Self {
        self.early_exaggeration = early_exaggeration;
        self
    }

    /// Sets the learning rate for gradient descent
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the number of iterations without progress before early stopping
    pub fn with_n_iter_without_progress(mut self, n_iter_without_progress: usize) -> Self {
        self.n_iter_without_progress = n_iter_without_progress;
        self
    }

    /// Sets the minimum gradient norm for convergence
    pub fn with_min_grad_norm(mut self, min_grad_norm: f64) -> Self {
        self.min_grad_norm = min_grad_norm;
        self
    }

    /// Sets the metric for pairwise distance computation
    ///
    /// Supported metrics:
    /// - "euclidean": Euclidean distance (L2 norm) - default
    /// - "manhattan": Manhattan distance (L1 norm)
    /// - "cosine": Cosine distance (1 - cosine similarity)
    /// - "chebyshev": Chebyshev distance (maximum coordinate difference)
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Sets the method for dimensionality reduction ("exact" or "barnes_hut")
    pub fn with_method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Sets the initialization method ("pca" or "random")
    pub fn with_init(mut self, init: &str) -> Self {
        self.init = init.to_string();
        self
    }

    /// Sets the angle for Barnes-Hut approximation (0.0 = exact, 1.0 = fast but approximate)
    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angle = angle;
        self
    }

    /// Sets the number of parallel jobs to run
    /// * n_jobs = -1: Use all available cores
    /// * n_jobs = 1: Use single-core (disable multicore)
    /// * n_jobs > 1: Use specific number of cores
    pub fn with_n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Sets the verbosity level
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Sets the random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Sets the degrees of freedom for the Student-t distribution
    ///
    /// Default is n_components - 1 (or 1 if n_components <= 1).
    /// Setting this to a larger value produces heavier tails, which can help
    /// with crowding in higher-dimensional embeddings.
    pub fn with_degrees_of_freedom(mut self, dof: f64) -> Self {
        self.degrees_of_freedom = Some(dof);
        self
    }

    /// Fit t-SNE to input data and transform it to the embedded space
    ///
    /// # Arguments
    /// * `x` - Input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - Embedding of the training data, shape (n_samples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        let n_samples = x_f64.shape()[0];
        let n_features = x_f64.shape()[1];

        // Input validation
        if n_samples == 0 || n_features == 0 {
            return Err(TransformError::InvalidInput("Empty input data".to_string()));
        }

        if self.perplexity >= n_samples as f64 {
            return Err(TransformError::InvalidInput(format!(
                "perplexity ({}) must be less than n_samples ({})",
                self.perplexity, n_samples
            )));
        }

        if self.method == "barnes_hut" && self.n_components > 3 {
            return Err(TransformError::InvalidInput(
                "'n_components' should be <= 3 for barnes_hut algorithm".to_string(),
            ));
        }

        self.learning_rate_ = Some(self.learning_rate);

        // Initialize embedding
        let x_embedded = self.initialize_embedding(&x_f64)?;

        // Compute pairwise affinities (P)
        let p = self.compute_pairwise_affinities(&x_f64)?;

        // Run t-SNE optimization
        let (embedding, kl_divergence, n_iter) =
            self.tsne_optimization(p, x_embedded, n_samples)?;

        self.embedding_ = Some(embedding.clone());
        self.kl_divergence_ = Some(kl_divergence);
        self.n_iter_ = Some(n_iter);

        Ok(embedding)
    }

    /// Initialize embedding either with PCA or random
    fn initialize_embedding(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];

        if self.init == "pca" {
            // Bound by both dimensions: PCA can return at most min(n_samples, n_features) components.
            let n_components = self.n_components.min(x.shape()[0]).min(x.shape()[1]);
            let mut pca = PCA::new(n_components, true, false);
            let mut x_embedded = pca.fit_transform(x)?;

            // Scale PCA initialization
            let col_var = x_embedded.column(0).map(|&v| v * v).sum() / (n_samples as f64);
            let std_dev = col_var.sqrt();
            if std_dev > 0.0 {
                x_embedded.mapv_inplace(|v| v / std_dev * 1e-4);
            }

            Ok(x_embedded)
        } else if self.init == "random" {
            use scirs2_core::random::{thread_rng, Distribution};
            let normal = Normal::new(0.0, 1e-4).map_err(|e| {
                TransformError::ComputationError(format!(
                    "Failed to create normal distribution: {e}"
                ))
            })?;
            let mut rng = thread_rng();

            let data: Vec<f64> = (0..(n_samples * self.n_components))
                .map(|_| normal.sample(&mut rng))
                .collect();
            Array2::from_shape_vec((n_samples, self.n_components), data).map_err(|e| {
                TransformError::ComputationError(format!("Failed to create embedding array: {e}"))
            })
        } else {
            Err(TransformError::InvalidInput(format!(
                "Initialization method '{}' not recognized. Use 'pca' or 'random'.",
                self.init
            )))
        }
    }

    /// Compute pairwise affinities with perplexity-based normalization
    fn compute_pairwise_affinities(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        // Compute pairwise distances
        let distances = self.compute_pairwise_distances(x)?;

        // Convert distances to affinities using binary search for sigma
        let p = self.distances_to_affinities(&distances)?;

        // Symmetrize and normalize the affinity matrix
        let mut p_symmetric = &p + &p.t();

        let p_sum = p_symmetric.sum();
        if p_sum > 0.0 {
            p_symmetric.mapv_inplace(|v| v.max(MACHINE_EPSILON) / p_sum);
        }

        Ok(p_symmetric)
    }

    /// Compute pairwise distances with optional multicore support
    fn compute_pairwise_distances(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let mut distances = Array2::zeros((n_samples, n_samples));

        match self.metric.as_str() {
            "euclidean" => {
                if self.n_jobs == 1 {
                    for i in 0..n_samples {
                        for j in i + 1..n_samples {
                            let mut dist_squared = 0.0;
                            for k in 0..n_features {
                                let diff = x[[i, k]] - x[[j, k]];
                                dist_squared += diff * diff;
                            }
                            distances[[i, j]] = dist_squared;
                            distances[[j, i]] = dist_squared;
                        }
                    }
                } else {
                    let upper_triangle_indices: Vec<(usize, usize)> = (0..n_samples)
                        .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
                        .collect();

                    let squared_distances: Vec<f64> = upper_triangle_indices
                        .par_iter()
                        .map(|&(i, j)| {
                            let mut dist_squared = 0.0;
                            for k in 0..n_features {
                                let diff = x[[i, k]] - x[[j, k]];
                                dist_squared += diff * diff;
                            }
                            dist_squared
                        })
                        .collect();

                    for (idx, &(i, j)) in upper_triangle_indices.iter().enumerate() {
                        distances[[i, j]] = squared_distances[idx];
                        distances[[j, i]] = squared_distances[idx];
                    }
                }
            }
            "manhattan" => {
                let compute_manhattan = |i: usize, j: usize| -> f64 {
                    let mut dist = 0.0;
                    for k in 0..n_features {
                        dist += (x[[i, k]] - x[[j, k]]).abs();
                    }
                    dist
                };

                if self.n_jobs == 1 {
                    for i in 0..n_samples {
                        for j in i + 1..n_samples {
                            let dist = compute_manhattan(i, j);
                            distances[[i, j]] = dist;
                            distances[[j, i]] = dist;
                        }
                    }
                } else {
                    let upper: Vec<(usize, usize)> = (0..n_samples)
                        .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
                        .collect();
                    let dists: Vec<f64> = upper
                        .par_iter()
                        .map(|&(i, j)| {
                            let mut dist = 0.0;
                            for k in 0..n_features {
                                dist += (x[[i, k]] - x[[j, k]]).abs();
                            }
                            dist
                        })
                        .collect();
                    for (idx, &(i, j)) in upper.iter().enumerate() {
                        distances[[i, j]] = dists[idx];
                        distances[[j, i]] = dists[idx];
                    }
                }
            }
            "cosine" => {
                let mut normalized_x = Array2::zeros((n_samples, n_features));
                for i in 0..n_samples {
                    let row = x.row(i);
                    let norm = row.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if norm > EPSILON {
                        for j in 0..n_features {
                            normalized_x[[i, j]] = x[[i, j]] / norm;
                        }
                    }
                }

                let upper: Vec<(usize, usize)> = (0..n_samples)
                    .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
                    .collect();

                let compute_fn = |i: usize, j: usize| -> f64 {
                    let mut dot_product = 0.0;
                    for k in 0..n_features {
                        dot_product += normalized_x[[i, k]] * normalized_x[[j, k]];
                    }
                    1.0 - dot_product.clamp(-1.0, 1.0)
                };

                if self.n_jobs == 1 {
                    for &(i, j) in &upper {
                        let d = compute_fn(i, j);
                        distances[[i, j]] = d;
                        distances[[j, i]] = d;
                    }
                } else {
                    let dists: Vec<f64> = upper
                        .par_iter()
                        .map(|&(i, j)| {
                            let mut dp = 0.0;
                            for k in 0..n_features {
                                dp += normalized_x[[i, k]] * normalized_x[[j, k]];
                            }
                            1.0 - dp.clamp(-1.0, 1.0)
                        })
                        .collect();
                    for (idx, &(i, j)) in upper.iter().enumerate() {
                        distances[[i, j]] = dists[idx];
                        distances[[j, i]] = dists[idx];
                    }
                }
            }
            "chebyshev" => {
                let upper: Vec<(usize, usize)> = (0..n_samples)
                    .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
                    .collect();

                if self.n_jobs == 1 {
                    for &(i, j) in &upper {
                        let mut max_dist = 0.0;
                        for k in 0..n_features {
                            let diff = (x[[i, k]] - x[[j, k]]).abs();
                            max_dist = max_dist.max(diff);
                        }
                        distances[[i, j]] = max_dist;
                        distances[[j, i]] = max_dist;
                    }
                } else {
                    let dists: Vec<f64> = upper
                        .par_iter()
                        .map(|&(i, j)| {
                            let mut max_dist = 0.0;
                            for k in 0..n_features {
                                let diff = (x[[i, k]] - x[[j, k]]).abs();
                                max_dist = max_dist.max(diff);
                            }
                            max_dist
                        })
                        .collect();
                    for (idx, &(i, j)) in upper.iter().enumerate() {
                        distances[[i, j]] = dists[idx];
                        distances[[j, i]] = dists[idx];
                    }
                }
            }
            _ => {
                return Err(TransformError::InvalidInput(format!(
                    "Metric '{}' not supported. Use: 'euclidean', 'manhattan', 'cosine', 'chebyshev'",
                    self.metric
                )));
            }
        }

        Ok(distances)
    }

    /// Convert distances to affinities using perplexity-based normalization
    fn distances_to_affinities(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = distances.shape()[0];
        let target = (2.0f64).ln() * self.perplexity;

        if self.n_jobs == 1 {
            let mut p = Array2::zeros((n_samples, n_samples));
            for i in 0..n_samples {
                self.binary_search_sigma(i, distances, target, &mut p)?;
            }
            Ok(p)
        } else {
            let prob_rows: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let distances_i: Vec<f64> = (0..n_samples).map(|j| distances[[i, j]]).collect();
                    Self::binary_search_sigma_row(i, &distances_i, n_samples, target)
                })
                .collect();

            let mut p = Array2::zeros((n_samples, n_samples));
            for (i, row) in prob_rows.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    p[[i, j]] = val;
                }
            }
            Ok(p)
        }
    }

    /// Binary search for the optimal sigma (bandwidth) for a single row
    fn binary_search_sigma(
        &self,
        i: usize,
        distances: &Array2<f64>,
        target: f64,
        p: &mut Array2<f64>,
    ) -> Result<()> {
        let n_samples = distances.shape()[0];
        let mut beta_min = -f64::INFINITY;
        let mut beta_max = f64::INFINITY;
        let mut beta = 1.0;

        for _ in 0..50 {
            let mut sum_pi = 0.0;
            let mut h = 0.0;

            for j in 0..n_samples {
                if i == j {
                    p[[i, j]] = 0.0;
                    continue;
                }
                let p_ij = (-beta * distances[[i, j]]).exp();
                p[[i, j]] = p_ij;
                sum_pi += p_ij;
            }

            if sum_pi > 0.0 {
                for j in 0..n_samples {
                    if i == j {
                        continue;
                    }
                    p[[i, j]] /= sum_pi;
                    if p[[i, j]] > MACHINE_EPSILON {
                        h -= p[[i, j]] * p[[i, j]].ln();
                    }
                }
            }

            let h_diff = h - target;
            if h_diff.abs() < EPSILON {
                break;
            }

            if h_diff > 0.0 {
                beta_min = beta;
                beta = if beta_max == f64::INFINITY {
                    beta * 2.0
                } else {
                    (beta + beta_max) / 2.0
                };
            } else {
                beta_max = beta;
                beta = if beta_min == -f64::INFINITY {
                    beta / 2.0
                } else {
                    (beta + beta_min) / 2.0
                };
            }
        }

        Ok(())
    }

    /// Parallel-safe binary search for a single row (returns Vec)
    fn binary_search_sigma_row(
        i: usize,
        distances_i: &[f64],
        n_samples: usize,
        target: f64,
    ) -> Vec<f64> {
        let mut beta_min = -f64::INFINITY;
        let mut beta_max = f64::INFINITY;
        let mut beta = 1.0;
        let mut p_row = vec![0.0; n_samples];

        for _ in 0..50 {
            let mut sum_pi = 0.0;
            let mut h = 0.0;

            for j in 0..n_samples {
                if i == j {
                    p_row[j] = 0.0;
                    continue;
                }
                let p_ij = (-beta * distances_i[j]).exp();
                p_row[j] = p_ij;
                sum_pi += p_ij;
            }

            if sum_pi > 0.0 {
                for (j, prob) in p_row.iter_mut().enumerate().take(n_samples) {
                    if i == j {
                        continue;
                    }
                    *prob /= sum_pi;
                    if *prob > MACHINE_EPSILON {
                        h -= *prob * prob.ln();
                    }
                }
            }

            let h_diff = h - target;
            if h_diff.abs() < EPSILON {
                break;
            }

            if h_diff > 0.0 {
                beta_min = beta;
                beta = if beta_max == f64::INFINITY {
                    beta * 2.0
                } else {
                    (beta + beta_max) / 2.0
                };
            } else {
                beta_max = beta;
                beta = if beta_min == -f64::INFINITY {
                    beta / 2.0
                } else {
                    (beta + beta_min) / 2.0
                };
            }
        }

        p_row
    }

    /// Get the effective degrees of freedom for the t-distribution
    fn effective_dof(&self) -> f64 {
        if let Some(dof) = self.degrees_of_freedom {
            dof
        } else {
            (self.n_components - 1).max(1) as f64
        }
    }

    /// Main t-SNE optimization loop using gradient descent
    fn tsne_optimization(
        &self,
        p: Array2<f64>,
        initial_embedding: Array2<f64>,
        n_samples: usize,
    ) -> Result<(Array2<f64>, f64, usize)> {
        let n_components = self.n_components;
        let degrees_of_freedom = self.effective_dof();

        let mut embedding = initial_embedding;
        let mut update = Array2::zeros((n_samples, n_components));
        let mut gains = Array2::ones((n_samples, n_components));
        let mut error = f64::INFINITY;
        let mut best_error = f64::INFINITY;
        let mut best_iter = 0;
        let mut iter = 0;

        let exploration_n_iter = 250;
        let n_iter_check = 50;

        // Apply early exaggeration
        let p_early = &p * self.early_exaggeration;

        if self.verbose {
            println!("[t-SNE] Starting optimization with early exaggeration phase...");
        }

        // Early exaggeration phase
        for i in 0..exploration_n_iter {
            let (curr_error, grad) = if self.method == "barnes_hut" {
                self.compute_gradient_barnes_hut(&embedding, &p_early, degrees_of_freedom)?
            } else {
                self.compute_gradient_exact(&embedding, &p_early, degrees_of_freedom)?
            };

            self.gradient_update(
                &mut embedding,
                &mut update,
                &mut gains,
                &grad,
                0.5,
                self.learning_rate_,
            )?;

            if (i + 1) % n_iter_check == 0 {
                if self.verbose {
                    println!("[t-SNE] Iteration {}: error = {:.7}", i + 1, curr_error);
                }

                if curr_error < best_error {
                    best_error = curr_error;
                    best_iter = i;
                } else if i - best_iter > self.n_iter_without_progress {
                    if self.verbose {
                        println!("[t-SNE] Early convergence at iteration {}", i + 1);
                    }
                    break;
                }

                let grad_norm = grad.mapv(|v| v * v).sum().sqrt();
                if grad_norm < self.min_grad_norm {
                    if self.verbose {
                        println!(
                            "[t-SNE] Gradient norm {} below threshold at iteration {}",
                            grad_norm,
                            i + 1
                        );
                    }
                    break;
                }
            }

            iter = i;
        }

        if self.verbose {
            println!("[t-SNE] Completed early exaggeration, starting final optimization...");
        }

        // Final optimization phase without early exaggeration
        for i in iter + 1..self.max_iter {
            let (curr_error, grad) = if self.method == "barnes_hut" {
                self.compute_gradient_barnes_hut(&embedding, &p, degrees_of_freedom)?
            } else {
                self.compute_gradient_exact(&embedding, &p, degrees_of_freedom)?
            };
            error = curr_error;

            self.gradient_update(
                &mut embedding,
                &mut update,
                &mut gains,
                &grad,
                0.8,
                self.learning_rate_,
            )?;

            if (i + 1) % n_iter_check == 0 {
                if self.verbose {
                    println!("[t-SNE] Iteration {}: error = {:.7}", i + 1, curr_error);
                }

                if curr_error < best_error {
                    best_error = curr_error;
                    best_iter = i;
                } else if i - best_iter > self.n_iter_without_progress {
                    if self.verbose {
                        println!("[t-SNE] Stopping optimization at iteration {}", i + 1);
                    }
                    break;
                }

                let grad_norm = grad.mapv(|v| v * v).sum().sqrt();
                if grad_norm < self.min_grad_norm {
                    if self.verbose {
                        println!(
                            "[t-SNE] Gradient norm {} below threshold at iteration {}",
                            grad_norm,
                            i + 1
                        );
                    }
                    break;
                }
            }

            iter = i;
        }

        if self.verbose {
            println!(
                "[t-SNE] Optimization finished after {} iterations with error {:.7}",
                iter + 1,
                error
            );
        }

        Ok((embedding, error, iter + 1))
    }

    /// Compute gradient and error for exact t-SNE
    fn compute_gradient_exact(
        &self,
        embedding: &Array2<f64>,
        p: &Array2<f64>,
        degrees_of_freedom: f64,
    ) -> Result<(f64, Array2<f64>)> {
        let n_samples = embedding.shape()[0];
        let n_components = embedding.shape()[1];

        // Compute Q matrix
        let mut dist = Array2::zeros((n_samples, n_samples));
        let upper: Vec<(usize, usize)> = (0..n_samples)
            .flat_map(|i| ((i + 1)..n_samples).map(move |j| (i, j)))
            .collect();

        if self.n_jobs == 1 {
            for &(i, j) in &upper {
                let mut d_squared = 0.0;
                for k in 0..n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    d_squared += diff * diff;
                }
                let q_ij =
                    (1.0 + d_squared / degrees_of_freedom).powf(-(degrees_of_freedom + 1.0) / 2.0);
                dist[[i, j]] = q_ij;
                dist[[j, i]] = q_ij;
            }
        } else {
            let q_values: Vec<f64> = upper
                .par_iter()
                .map(|&(i, j)| {
                    let mut d_squared = 0.0;
                    for k in 0..n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        d_squared += diff * diff;
                    }
                    (1.0 + d_squared / degrees_of_freedom).powf(-(degrees_of_freedom + 1.0) / 2.0)
                })
                .collect();

            for (idx, &(i, j)) in upper.iter().enumerate() {
                dist[[i, j]] = q_values[idx];
                dist[[j, i]] = q_values[idx];
            }
        }

        for i in 0..n_samples {
            dist[[i, i]] = 0.0;
        }

        let sum_q = dist.sum().max(MACHINE_EPSILON);
        let q = &dist / sum_q;

        // Compute KL divergence
        let kl_divergence: f64 = if self.n_jobs == 1 {
            let mut kl = 0.0;
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if p[[i, j]] > MACHINE_EPSILON && q[[i, j]] > MACHINE_EPSILON {
                        kl += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                    }
                }
            }
            kl
        } else {
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut local_kl = 0.0;
                    for j in 0..n_samples {
                        if p[[i, j]] > MACHINE_EPSILON && q[[i, j]] > MACHINE_EPSILON {
                            local_kl += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                        }
                    }
                    local_kl
                })
                .sum()
        };

        // Compute gradient
        let factor = 4.0 * (degrees_of_freedom + 1.0) / (degrees_of_freedom * sum_q * sum_q);

        let grad = if self.n_jobs == 1 {
            let mut g = Array2::zeros((n_samples, n_components));
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let p_q_diff = p[[i, j]] - q[[i, j]];
                        for k in 0..n_components {
                            g[[i, k]] += factor
                                * p_q_diff
                                * dist[[i, j]]
                                * (embedding[[i, k]] - embedding[[j, k]]);
                        }
                    }
                }
            }
            g
        } else {
            let grad_rows: Vec<Vec<f64>> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut grad_row = vec![0.0; n_components];
                    for j in 0..n_samples {
                        if i != j {
                            let p_q_diff = p[[i, j]] - q[[i, j]];
                            for k in 0..n_components {
                                grad_row[k] += factor
                                    * p_q_diff
                                    * dist[[i, j]]
                                    * (embedding[[i, k]] - embedding[[j, k]]);
                            }
                        }
                    }
                    grad_row
                })
                .collect();

            let mut g = Array2::zeros((n_samples, n_components));
            for (i, row) in grad_rows.iter().enumerate() {
                for (k, &val) in row.iter().enumerate() {
                    g[[i, k]] = val;
                }
            }
            g
        };

        Ok((kl_divergence, grad))
    }

    /// Compute gradient and error using Barnes-Hut approximation
    fn compute_gradient_barnes_hut(
        &self,
        embedding: &Array2<f64>,
        p: &Array2<f64>,
        degrees_of_freedom: f64,
    ) -> Result<(f64, Array2<f64>)> {
        let n_samples = embedding.shape()[0];
        let n_components = embedding.shape()[1];

        // Build spatial tree
        let tree = if n_components == 2 {
            SpatialTree::new_quadtree(embedding)?
        } else if n_components == 3 {
            SpatialTree::new_octree(embedding)?
        } else {
            return Err(TransformError::InvalidInput(
                "Barnes-Hut only supports 2D and 3D embeddings".to_string(),
            ));
        };

        let mut q = Array2::zeros((n_samples, n_samples));
        let mut grad = Array2::zeros((n_samples, n_components));
        let mut sum_q = 0.0;

        // Compute repulsive forces using Barnes-Hut
        for i in 0..n_samples {
            let point = embedding.row(i).to_owned();
            let (repulsive_force, q_sum) =
                tree.compute_forces(&point, i, self.angle, degrees_of_freedom)?;

            sum_q += q_sum;

            for j in 0..n_components {
                grad[[i, j]] += repulsive_force[j];
            }

            // Compute Q matrix entries for KL divergence
            for j in 0..n_samples {
                if i != j {
                    let mut dist_squared = 0.0;
                    for k in 0..n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        dist_squared += diff * diff;
                    }
                    let q_ij = (1.0 + dist_squared / degrees_of_freedom)
                        .powf(-(degrees_of_freedom + 1.0) / 2.0);
                    q[[i, j]] = q_ij;
                }
            }
        }

        sum_q = sum_q.max(MACHINE_EPSILON);
        q.mapv_inplace(|v| v / sum_q);

        // Add attractive forces
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j && p[[i, j]] > MACHINE_EPSILON {
                    let mut dist_squared = 0.0;
                    for k in 0..n_components {
                        let diff = embedding[[i, k]] - embedding[[j, k]];
                        dist_squared += diff * diff;
                    }

                    let q_ij = (1.0 + dist_squared / degrees_of_freedom)
                        .powf(-(degrees_of_freedom + 1.0) / 2.0);
                    let attraction = 4.0 * p[[i, j]] * q_ij;

                    for k in 0..n_components {
                        grad[[i, k]] -= attraction * (embedding[[i, k]] - embedding[[j, k]]);
                    }
                }
            }
        }

        // Compute KL divergence
        let mut kl_divergence = 0.0;
        for i in 0..n_samples {
            for j in 0..n_samples {
                if p[[i, j]] > MACHINE_EPSILON && q[[i, j]] > MACHINE_EPSILON {
                    kl_divergence += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                }
            }
        }

        Ok((kl_divergence, grad))
    }

    /// Update embedding using gradient descent with momentum and adaptive gains
    fn gradient_update(
        &self,
        embedding: &mut Array2<f64>,
        update: &mut Array2<f64>,
        gains: &mut Array2<f64>,
        grad: &Array2<f64>,
        momentum: f64,
        learning_rate: Option<f64>,
    ) -> Result<()> {
        let n_samples = embedding.shape()[0];
        let n_components = embedding.shape()[1];
        let eta = learning_rate.unwrap_or(self.learning_rate);

        for i in 0..n_samples {
            for j in 0..n_components {
                let same_sign = update[[i, j]] * grad[[i, j]] > 0.0;

                if same_sign {
                    gains[[i, j]] *= 0.8;
                } else {
                    gains[[i, j]] += 0.2;
                }

                gains[[i, j]] = gains[[i, j]].max(0.01);
                update[[i, j]] = momentum * update[[i, j]] - eta * gains[[i, j]] * grad[[i, j]];
                embedding[[i, j]] += update[[i, j]];
            }
        }

        Ok(())
    }

    /// Returns the embedding after fitting
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding_.as_ref()
    }

    /// Returns the KL divergence after optimization
    pub fn kl_divergence(&self) -> Option<f64> {
        self.kl_divergence_
    }

    /// Returns the number of iterations run
    pub fn n_iter(&self) -> Option<usize> {
        self.n_iter_
    }
}

/// Calculate trustworthiness score for a dimensionality reduction
///
/// Trustworthiness measures to what extent the local structure is retained when
/// projecting data from the original space to the embedding space.
///
/// A trustworthiness of 1.0 means all local neighborhoods are perfectly preserved.
///
/// # Arguments
/// * `x` - Original data, shape (n_samples, n_features)
/// * `x_embedded` - Embedded data, shape (n_samples, n_components)
/// * `n_neighbors` - Number of neighbors to consider
/// * `metric` - Metric to use (currently only 'euclidean' is supported)
///
/// # Returns
/// * `Result<f64>` - Trustworthiness score between 0.0 and 1.0
pub fn trustworthiness<S1, S2>(
    x: &ArrayBase<S1, Ix2>,
    x_embedded: &ArrayBase<S2, Ix2>,
    n_neighbors: usize,
    metric: &str,
) -> Result<f64>
where
    S1: Data,
    S2: Data,
    S1::Elem: Float + NumCast,
    S2::Elem: Float + NumCast,
{
    let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));
    let x_embedded_f64 = x_embedded.mapv(|v| NumCast::from(v).unwrap_or(0.0));

    let n_samples = x_f64.shape()[0];

    if n_neighbors >= n_samples / 2 {
        return Err(TransformError::InvalidInput(format!(
            "n_neighbors ({}) should be less than n_samples / 2 ({})",
            n_neighbors,
            n_samples / 2
        )));
    }

    if metric != "euclidean" {
        return Err(TransformError::InvalidInput(format!(
            "Metric '{metric}' not supported. Currently only 'euclidean' is implemented."
        )));
    }

    // Compute pairwise distances in original space
    let mut dist_x = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i == j {
                dist_x[[i, j]] = f64::INFINITY;
                continue;
            }
            let mut d_squared = 0.0;
            for k in 0..x_f64.shape()[1] {
                let diff = x_f64[[i, k]] - x_f64[[j, k]];
                d_squared += diff * diff;
            }
            dist_x[[i, j]] = d_squared.sqrt();
        }
    }

    // Compute pairwise distances in embedded space
    let mut dist_embedded = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i == j {
                dist_embedded[[i, j]] = f64::INFINITY;
                continue;
            }
            let mut d_squared = 0.0;
            for k in 0..x_embedded_f64.shape()[1] {
                let diff = x_embedded_f64[[i, k]] - x_embedded_f64[[j, k]];
                d_squared += diff * diff;
            }
            dist_embedded[[i, j]] = d_squared.sqrt();
        }
    }

    // For each point, find the n_neighbors nearest neighbors in the original space
    let mut nn_orig = Array2::<usize>::zeros((n_samples, n_neighbors));
    for i in 0..n_samples {
        let row = dist_x.row(i).to_owned();
        let mut pairs: Vec<(usize, f64)> = row.iter().enumerate().map(|(j, &d)| (j, d)).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (j, &(idx, _)) in pairs.iter().enumerate().take(n_neighbors) {
            nn_orig[[i, j]] = idx;
        }
    }

    // For each point, find n_neighbors nearest neighbors in embedded space
    let mut nn_embedded = Array2::<usize>::zeros((n_samples, n_neighbors));
    for i in 0..n_samples {
        let row = dist_embedded.row(i).to_owned();
        let mut pairs: Vec<(usize, f64)> = row.iter().enumerate().map(|(j, &d)| (j, d)).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (j, &(idx, _)) in pairs.iter().skip(1).take(n_neighbors).enumerate() {
            nn_embedded[[i, j]] = idx;
        }
    }

    // Calculate the trustworthiness score
    let mut t = 0.0;
    for i in 0..n_samples {
        for &j in nn_embedded.row(i).iter() {
            let is_not_neighbor = !nn_orig.row(i).iter().any(|&nn| nn == j);

            if is_not_neighbor {
                let row = dist_x.row(i).to_owned();
                let mut pairs: Vec<(usize, f64)> =
                    row.iter().enumerate().map(|(idx, &d)| (idx, d)).collect();
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let rank = pairs
                    .iter()
                    .position(|&(idx, _)| idx == j)
                    .unwrap_or(n_neighbors)
                    .saturating_sub(n_neighbors);

                t += rank as f64;
            }
        }
    }

    // Normalize
    let n = n_samples as f64;
    let k = n_neighbors as f64;
    let normalizer = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0));
    let trustworthiness_val = 1.0 - normalizer * t;

    Ok(trustworthiness_val)
}

#[cfg(test)]
#[path = "../tsne_tests.rs"]
mod tests;
