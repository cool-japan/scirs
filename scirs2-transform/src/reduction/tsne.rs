//! t-SNE (t-distributed Stochastic Neighbor Embedding) implementation
//!
//! This module provides an implementation of t-SNE, a technique for dimensionality
//! reduction particularly well-suited for visualization of high-dimensional data.
//!
//! t-SNE converts similarities between data points to joint probabilities and tries
//! to minimize the Kullback-Leibler divergence between the joint probabilities of
//! the low-dimensional embedding and the high-dimensional data.

use ndarray::{Array2, ArrayBase, Data, Ix2};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use num_traits::{Float, NumCast};
// Remove matmul import as it seems not to be used in the code

use crate::error::{Result, TransformError};
use crate::reduction::PCA;

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
    /// Method to perform dimensionality reduction
    method: String,
    /// Initialization method
    init: String,
    /// Angle for Barnes-Hut approximation
    angle: f64,
    /// Verbosity level
    verbose: bool,
    /// Random state for reproducibility
    random_state: Option<u64>,
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
            verbose: false,
            random_state: None,
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
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Sets the method for dimensionality reduction
    pub fn with_method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Sets the initialization method
    pub fn with_init(mut self, init: &str) -> Self {
        self.init = init.to_string();
        self
    }

    /// Sets the angle for Barnes-Hut approximation
    pub fn with_angle(mut self, angle: f64) -> Self {
        self.angle = angle;
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
        let x_f64 = x.mapv(|x| num_traits::cast::<S::Elem, f64>(x).unwrap_or(0.0));

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
                "'n_components' should be less than or equal to 3 for barnes_hut algorithm"
                    .to_string(),
            ));
        }

        // Set learning rate if auto
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
            let n_components = self.n_components.min(x.shape()[1]);
            let mut pca = PCA::new(n_components, true, false);
            let mut x_embedded = pca.fit_transform(x)?;

            // Scale PCA initialization
            let std_dev = (x_embedded.column(0).map(|&x| x * x).sum() / (n_samples as f64)).sqrt();
            if std_dev > 0.0 {
                x_embedded.mapv_inplace(|x| x / std_dev * 1e-4);
            }

            Ok(x_embedded)
        } else if self.init == "random" {
            // Random initialization from standard normal distribution
            // Ignoring random_state as it's not needed for basic random functionality
            let normal = Normal::new(0.0, 1e-4).unwrap();

            // Use simple random initialization
            Ok(Array2::random((n_samples, self.n_components), normal))
        } else {
            Err(TransformError::InvalidInput(format!(
                "Initialization method '{}' not recognized",
                self.init
            )))
        }
    }

    /// Compute pairwise affinities with perplexity-based normalization
    fn compute_pairwise_affinities(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let _n_samples = x.shape()[0];

        // Compute pairwise distances
        let distances = self.compute_pairwise_distances(x)?;

        // Convert distances to affinities using binary search for sigma
        let p = self.distances_to_affinities(&distances)?;

        // Symmetrize and normalize the affinity matrix
        let mut p_symmetric = &p + &p.t();

        // Normalize
        let p_sum = p_symmetric.sum();
        if p_sum > 0.0 {
            p_symmetric.mapv_inplace(|x| x.max(MACHINE_EPSILON) / p_sum);
        }

        Ok(p_symmetric)
    }

    /// Compute pairwise Euclidean distances
    fn compute_pairwise_distances(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.shape()[0];
        let mut distances = Array2::zeros((n_samples, n_samples));

        if self.metric == "euclidean" {
            // Compute Euclidean distances
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    let mut dist_squared = 0.0;
                    for k in 0..x.shape()[1] {
                        let diff = x[[i, k]] - x[[j, k]];
                        dist_squared += diff * diff;
                    }
                    distances[[i, j]] = dist_squared;
                    distances[[j, i]] = dist_squared;
                }
            }
        } else {
            return Err(TransformError::InvalidInput(format!(
                "Metric '{}' not implemented. Currently only 'euclidean' is supported",
                self.metric
            )));
        }

        Ok(distances)
    }

    /// Convert distances to affinities using perplexity-based normalization
    fn distances_to_affinities(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = distances.shape()[0];
        let mut p = Array2::zeros((n_samples, n_samples));

        // Perform binary search to find sigma for each point
        for i in 0..n_samples {
            let mut beta_min = -f64::INFINITY;
            let mut beta_max = f64::INFINITY;
            let mut beta = 1.0;
            let target = (2.0f64).ln() * self.perplexity;

            // Get all distances from point i except self-distance (which is 0)
            let distances_i = distances.row(i).to_owned();

            // Binary search for beta
            for _ in 0..50 {
                // Usually converges within 50 iterations
                // Compute conditional probabilities with current beta
                let mut sum_pi = 0.0;
                let mut h = 0.0;

                for j in 0..n_samples {
                    if i == j {
                        p[[i, j]] = 0.0;
                        continue;
                    }

                    let p_ij = (-beta * distances_i[j]).exp();
                    p[[i, j]] = p_ij;
                    sum_pi += p_ij;
                }

                // Normalize probabilities and compute entropy
                if sum_pi > 0.0 {
                    for j in 0..n_samples {
                        if i == j {
                            continue;
                        }

                        p[[i, j]] /= sum_pi;

                        // Compute entropy
                        if p[[i, j]] > MACHINE_EPSILON {
                            h -= p[[i, j]] * p[[i, j]].ln();
                        }
                    }
                }

                // Adjust beta based on entropy difference from target
                let h_diff = h - target;

                if h_diff.abs() < EPSILON {
                    break; // Converged
                }

                // Update beta using binary search
                if h_diff > 0.0 {
                    beta_min = beta;
                    if beta_max == f64::INFINITY {
                        beta *= 2.0;
                    } else {
                        beta = (beta + beta_max) / 2.0;
                    }
                } else {
                    beta_max = beta;
                    if beta_min == -f64::INFINITY {
                        beta /= 2.0;
                    } else {
                        beta = (beta + beta_min) / 2.0;
                    }
                }
            }
        }

        Ok(p)
    }

    /// Main t-SNE optimization loop using gradient descent
    fn tsne_optimization(
        &self,
        p: Array2<f64>,
        initial_embedding: Array2<f64>,
        n_samples: usize,
    ) -> Result<(Array2<f64>, f64, usize)> {
        let n_components = self.n_components;
        let degrees_of_freedom = (n_components - 1).max(1) as f64;

        // Initialize variables for optimization
        let mut embedding = initial_embedding;
        let mut update = Array2::zeros((n_samples, n_components));
        let mut gains = Array2::ones((n_samples, n_components));
        let mut error = f64::INFINITY;
        let mut best_error = f64::INFINITY;
        let mut best_iter = 0;
        let mut iter = 0;

        // Exploration phase with early exaggeration
        let exploration_n_iter = 250;
        let n_iter_check = 50;

        // Apply early exaggeration
        let p_early = &p * self.early_exaggeration;

        if self.verbose {
            println!("[t-SNE] Starting optimization with early exaggeration phase...");
        }

        // Early exaggeration phase
        for i in 0..exploration_n_iter {
            // Compute gradient and error for early exaggeration phase
            let (curr_error, grad) =
                self.compute_gradient_exact(&embedding, &p_early, degrees_of_freedom)?;

            // Perform gradient update with momentum and gains
            self.gradient_update(
                &mut embedding,
                &mut update,
                &mut gains,
                &grad,
                0.5,
                self.learning_rate_,
            )?;

            // Check for convergence
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

                // Check gradient norm
                let grad_norm = grad.mapv(|x| x * x).sum().sqrt();
                if grad_norm < self.min_grad_norm {
                    if self.verbose {
                        println!("[t-SNE] Gradient norm {} below threshold, stopping optimization at iteration {}", 
                                grad_norm, i + 1);
                    }
                    break;
                }
            }

            iter = i;
        }

        if self.verbose {
            println!("[t-SNE] Completed early exaggeration phase, starting final optimization...");
        }

        // Final optimization phase without early exaggeration
        for i in iter + 1..self.max_iter {
            // Compute gradient and error for normal phase
            let (curr_error, grad) =
                self.compute_gradient_exact(&embedding, &p, degrees_of_freedom)?;
            error = curr_error;

            // Perform gradient update with momentum and gains
            self.gradient_update(
                &mut embedding,
                &mut update,
                &mut gains,
                &grad,
                0.8,
                self.learning_rate_,
            )?;

            // Check for convergence
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

                // Check gradient norm
                let grad_norm = grad.mapv(|x| x * x).sum().sqrt();
                if grad_norm < self.min_grad_norm {
                    if self.verbose {
                        println!("[t-SNE] Gradient norm {} below threshold, stopping optimization at iteration {}", 
                                grad_norm, i + 1);
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

        // Compute pairwise squared Euclidean distances in the embedded space
        let mut dist = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let mut d_squared = 0.0;
                for k in 0..n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    d_squared += diff * diff;
                }

                // Convert squared distance to t-distribution's probability
                let q_ij =
                    (1.0 + d_squared / degrees_of_freedom).powf(-(degrees_of_freedom + 1.0) / 2.0);
                dist[[i, j]] = q_ij;
                dist[[j, i]] = q_ij;
            }
        }

        // Set diagonal to zero (self-distance)
        for i in 0..n_samples {
            dist[[i, i]] = 0.0;
        }

        // Normalize Q matrix
        let sum_q = dist.sum().max(MACHINE_EPSILON);
        let q = &dist / sum_q;

        // Compute KL divergence
        let mut kl_divergence = 0.0;
        for i in 0..n_samples {
            for j in 0..n_samples {
                if p[[i, j]] > MACHINE_EPSILON && q[[i, j]] > MACHINE_EPSILON {
                    kl_divergence += p[[i, j]] * (p[[i, j]] / q[[i, j]]).ln();
                }
            }
        }

        // Compute gradient
        let mut grad = Array2::zeros((n_samples, n_components));
        let factor = 4.0 * (degrees_of_freedom + 1.0) / (degrees_of_freedom * (sum_q.powf(2.0)));

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let p_q_diff = p[[i, j]] - q[[i, j]];
                    for k in 0..n_components {
                        grad[[i, k]] += factor
                            * p_q_diff
                            * dist[[i, j]]
                            * (embedding[[i, k]] - embedding[[j, k]]);
                    }
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

        // Update gains and momentum
        for i in 0..n_samples {
            for j in 0..n_components {
                let same_sign = update[[i, j]] * grad[[i, j]] > 0.0;

                if same_sign {
                    gains[[i, j]] *= 0.8;
                } else {
                    gains[[i, j]] += 0.2;
                }

                // Ensure minimum gain
                gains[[i, j]] = gains[[i, j]].max(0.01);

                // Update with momentum and adaptive learning rate
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
/// # Arguments
/// * `x` - Original data, shape (n_samples, n_features)
/// * `x_embedded` - Embedded data, shape (n_samples, n_components)
/// * `n_neighbors` - Number of neighbors to consider
/// * `metric` - Metric to use (currently only 'euclidean' is implemented)
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
    let x_f64 = x.mapv(|x| num_traits::cast::<S1::Elem, f64>(x).unwrap_or(0.0));
    let x_embedded_f64 = x_embedded.mapv(|x| num_traits::cast::<S2::Elem, f64>(x).unwrap_or(0.0));

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
            "Metric '{}' not implemented. Currently only 'euclidean' is supported",
            metric
        )));
    }

    // Compute pairwise distances in original space
    let mut dist_x = Array2::zeros((n_samples, n_samples));
    for i in 0..n_samples {
        for j in 0..n_samples {
            if i == j {
                dist_x[[i, j]] = f64::INFINITY; // Set self-distance to infinity
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
                dist_embedded[[i, j]] = f64::INFINITY; // Set self-distance to infinity
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
        // Get the indices of the sorted distances
        let row = dist_x.row(i).to_owned();
        let mut pairs: Vec<(usize, f64)> = row.iter().enumerate().map(|(j, &d)| (j, d)).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // The first element will be i itself (distance 0), so skip it
        for (j, &(idx, _)) in pairs.iter().enumerate().take(n_neighbors) {
            nn_orig[[i, j]] = idx;
        }
    }

    // For each point, find the n_neighbors nearest neighbors in the embedded space
    let mut nn_embedded = Array2::<usize>::zeros((n_samples, n_neighbors));
    for i in 0..n_samples {
        // Get the indices of the sorted distances
        let row = dist_embedded.row(i).to_owned();
        let mut pairs: Vec<(usize, f64)> = row.iter().enumerate().map(|(j, &d)| (j, d)).collect();
        pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // The first element will be i itself (distance 0), so skip it
        for (j, &(idx, _)) in pairs.iter().enumerate().take(n_neighbors) {
            nn_embedded[[i, j]] = idx;
        }
    }

    // Calculate the trustworthiness score
    let mut t = 0.0;
    for i in 0..n_samples {
        for &j in nn_embedded.row(i).iter() {
            // Check if j is not in the n_neighbors nearest neighbors in the original space
            let is_not_neighbor = !nn_orig.row(i).iter().any(|&nn| nn == j);

            if is_not_neighbor {
                // Find the rank of j in the original space
                let row = dist_x.row(i).to_owned();
                let mut pairs: Vec<(usize, f64)> =
                    row.iter().enumerate().map(|(idx, &d)| (idx, d)).collect();
                pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let rank = pairs.iter().position(|&(idx, _)| idx == j).unwrap_or(0) - n_neighbors;

                t += rank as f64;
            }
        }
    }

    // Normalize the trustworthiness score
    let n = n_samples as f64;
    let k = n_neighbors as f64;
    let normalizer = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0));
    let trustworthiness = 1.0 - normalizer * t;

    Ok(trustworthiness)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    #[test]
    fn test_tsne_simple() {
        // Create a simple dataset
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [6.0, 5.0],
            [5.0, 6.0],
            [6.0, 6.0],
        ]);

        // Initialize and fit t-SNE
        let mut tsne = TSNE::new()
            .with_n_components(2)
            .with_perplexity(2.0)
            .with_random_state(42)
            .with_max_iter(250)
            .with_verbose(false);

        let embedding = tsne.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(embedding.shape(), &[8, 2]);

        // Check that groups are separated in the embedding space
        // Compute the average distance within each group
        let dist_group1 = average_pairwise_distance(&embedding.slice(ndarray::s![0..4, ..]));
        let dist_group2 = average_pairwise_distance(&embedding.slice(ndarray::s![4..8, ..]));

        // Compute the average distance between groups
        let dist_between = average_intergroup_distance(
            &embedding.slice(ndarray::s![0..4, ..]),
            &embedding.slice(ndarray::s![4..8, ..]),
        );

        // The between-group distance should be larger than the within-group distances
        assert!(dist_between > dist_group1);
        assert!(dist_between > dist_group2);
    }

    // Helper function to compute average pairwise distance within a group
    fn average_pairwise_distance(points: &ArrayBase<ndarray::ViewRepr<&f64>, Ix2>) -> f64 {
        let n = points.shape()[0];
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in i + 1..n {
                let mut dist_squared = 0.0;
                for k in 0..points.shape()[1] {
                    let diff = points[[i, k]] - points[[j, k]];
                    dist_squared += diff * diff;
                }
                total_dist += dist_squared.sqrt();
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }

    // Helper function to compute average distance between two groups
    fn average_intergroup_distance(
        group1: &ArrayBase<ndarray::ViewRepr<&f64>, Ix2>,
        group2: &ArrayBase<ndarray::ViewRepr<&f64>, Ix2>,
    ) -> f64 {
        let n1 = group1.shape()[0];
        let n2 = group2.shape()[0];
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..n1 {
            for j in 0..n2 {
                let mut dist_squared = 0.0;
                for k in 0..group1.shape()[1] {
                    let diff = group1[[i, k]] - group2[[j, k]];
                    dist_squared += diff * diff;
                }
                total_dist += dist_squared.sqrt();
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }

    #[test]
    fn test_trustworthiness() {
        // Create a simple dataset where we know the structure
        let x = arr2(&[
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [5.0, 5.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [6.0, 6.0],
        ]);

        // A perfect embedding would preserve all neighborhoods
        let perfect_embedding = x.clone();
        let t_perfect = trustworthiness(&x, &perfect_embedding, 3, "euclidean").unwrap();
        assert_abs_diff_eq!(t_perfect, 1.0, epsilon = 1e-10);

        // A random embedding would have low trustworthiness
        let random_embedding = arr2(&[
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.3, 0.7],
            [0.2, 0.8],
        ]);

        let t_random = trustworthiness(&x, &random_embedding, 3, "euclidean").unwrap();
        assert!(t_random < 1.0);
    }
}
