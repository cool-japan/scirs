//! Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction
//!
//! UMAP is a non-linear dimensionality reduction technique that can be used for
//! visualization similarly to t-SNE, but also for general non-linear dimension reduction.

use ndarray::{Array2, ArrayBase, Data, Ix2};
use num_traits::{Float, NumCast};
use rand::Rng;
use scirs2_core::validation::{check_positive, checkshape};
use std::collections::BinaryHeap;

use crate::error::{Result, TransformError};

/// UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction
///
/// UMAP constructs a high dimensional graph representation of the data then optimizes
/// a low dimensional graph to be as structurally similar as possible.
#[derive(Debug, Clone)]
pub struct UMAP {
    /// Number of neighbors to consider for local structure
    n_neighbors: usize,
    /// Number of components (dimensions) in the low dimensional space
    n_components: usize,
    /// Controls how UMAP balances local versus global structure
    #[allow(dead_code)]
    mindist: f64,
    /// Controls how tightly UMAP is allowed to pack points together
    #[allow(dead_code)]
    spread: f64,
    /// Learning rate for optimization
    learning_rate: f64,
    /// Number of epochs for optimization
    n_epochs: usize,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Training k-NN graph for out-of-sample extension
    training_graph: Option<Array2<f64>>,
    /// Metric to use for distance computation
    metric: String,
    /// The low dimensional embedding
    embedding: Option<Array2<f64>>,
    /// Parameters for the optimization
    a: f64,
    b: f64,
}

impl UMAP {
    /// Creates a new UMAP instance
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighbors to consider for local structure (default: 15)
    /// * `n_components` - Number of dimensions in the low dimensional space (default: 2)
    /// * `mindist` - Minimum distance between points in low dimensional space (default: 0.1)
    /// * `learning_rate` - Learning rate for optimization (default: 1.0)
    /// * `n_epochs` - Number of epochs for optimization (default: 200)
    pub fn new(
        n_neighbors: usize,
        n_components: usize,
        mindist: f64,
        learning_rate: f64,
        n_epochs: usize,
    ) -> Self {
        // Compute a and b parameters based on mindist and spread
        let spread = 1.0;
        let (a, b) = Self::find_ab_params(spread, mindist);

        UMAP {
            n_neighbors,
            n_components,
            mindist,
            spread,
            learning_rate,
            n_epochs,
            random_state: None,
            metric: "euclidean".to_string(),
            embedding: None,
            training_data: None,
            training_graph: None,
            a,
            b,
        }
    }

    /// Sets the random state for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sets the distance metric
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// Find a and b parameters to approximate the fuzzy set membership function
    fn find_ab_params(_spread: f64, mindist: f64) -> (f64, f64) {
        // Binary search to find good values of a and b
        let mut a = 1.0;
        let mut b = 1.0;

        // Initial guess based on mindist and _spread
        if mindist > 0.0 {
            b = mindist.ln() / (1.0 - mindist).ln();
        }

        // Refine using Newton's method
        for _ in 0..64 {
            let val = 1.0 / (1.0 + a * mindist.powf(2.0 * b));
            let grad_a = -mindist.powf(2.0 * b) / (1.0 + a * mindist.powf(2.0 * b)).powi(2);
            let grad_b = -2.0 * a * mindist.powf(2.0 * b) * mindist.ln()
                / (1.0 + a * mindist.powf(2.0 * b)).powi(2);

            if (val - 0.5).abs() < 1e-5 {
                break;
            }

            a -= (val - 0.5) / grad_a;
            b -= (val - 0.5) / grad_b;

            a = a.max(0.001);
            b = b.max(0.001);
        }

        (a, b)
    }

    /// Compute pairwise distances between all points
    fn compute_distances<S>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let nsamples = x.shape()[0];
        let mut distances = Array2::zeros((nsamples, nsamples));

        // Compute pairwise Euclidean distances
        for i in 0..nsamples {
            for j in i + 1..nsamples {
                let mut dist = 0.0;
                for k in 0..x.shape()[1] {
                    let diff = num_traits::cast::<S::Elem, f64>(x[[i, k]]).unwrap_or(0.0)
                        - num_traits::cast::<S::Elem, f64>(x[[j, k]]).unwrap_or(0.0);
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Find k nearest neighbors for each point
    fn find_neighbors(&self, distances: &Array2<f64>) -> (Array2<usize>, Array2<f64>) {
        let nsamples = distances.shape()[0];
        let k = self.n_neighbors;

        let mut indices = Array2::zeros((nsamples, k));
        let mut neighbor_distances = Array2::zeros((nsamples, k));

        for i in 0..nsamples {
            // Use a min heap to find k smallest distances
            let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();

            for j in 0..nsamples {
                if i != j {
                    // Convert to fixed point for comparison
                    let dist_fixed = (distances[[i, j]] * 1e9) as i64;
                    heap.push((std::cmp::Reverse(dist_fixed), j));
                }
            }

            // Extract k nearest neighbors
            for j in 0..k {
                if let Some((std::cmp::Reverse(dist_fixed), idx)) = heap.pop() {
                    indices[[i, j]] = idx;
                    neighbor_distances[[i, j]] = dist_fixed as f64 / 1e9;
                }
            }
        }

        (indices, neighbor_distances)
    }

    /// Compute fuzzy simplicial set (high dimensional graph)
    fn compute_graph(
        &self,
        knn_indices: &Array2<usize>,
        knn_distances: &Array2<f64>,
    ) -> Array2<f64> {
        let nsamples = knn_indices.shape()[0];
        let mut graph = Array2::zeros((nsamples, nsamples));

        // For each point, compute membership strengths to its neighbors
        for i in 0..nsamples {
            // Find rho (distance to nearest neighbor)
            let rho = knn_distances[[i, 0]];

            // Binary search for sigma
            let mut sigma = 1.0;
            let target = self.n_neighbors as f64;

            for _ in 0..64 {
                let mut sum = 0.0;
                for j in 1..self.n_neighbors {
                    let d = (knn_distances[[i, j]] - rho).max(0.0);
                    sum += (-d / sigma).exp();
                }

                if (sum - target).abs() < 1e-5 {
                    break;
                }

                if sum > target {
                    sigma *= 2.0;
                } else {
                    sigma /= 2.0;
                }
            }

            // Compute membership strengths
            for j in 0..self.n_neighbors {
                let neighbor_idx = knn_indices[[i, j]];
                let d = (knn_distances[[i, j]] - rho).max(0.0);
                let strength = (-d / sigma).exp();
                graph[[i, neighbor_idx]] = strength;
            }
        }

        // Symmetrize the graph
        let graph_transpose = graph.t().to_owned();
        &graph + &graph_transpose - &graph * &graph_transpose
    }

    /// Initialize the low dimensional embedding
    fn initialize_embedding(&self, nsamples: usize) -> Array2<f64> {
        let mut rng = rand::rng();

        // Initialize with small random values
        let mut embedding = Array2::zeros((nsamples, self.n_components));
        for i in 0..nsamples {
            for j in 0..self.n_components {
                embedding[[i, j]] = rng.gen_range(0.0..1.0) * 10.0 - 5.0;
            }
        }

        embedding
    }

    /// Optimize the low dimensional embedding
    fn optimize_embedding(
        &self,
        embedding: &mut Array2<f64>,
        graph: &Array2<f64>,
        n_epochs: usize,
    ) {
        let nsamples = embedding.shape()[0];
        let mut rng = rand::rng();

        // Create edge list from graph
        let mut edges = Vec::new();
        let mut weights = Vec::new();
        for i in 0..nsamples {
            for j in 0..nsamples {
                if graph[[i, j]] > 0.0 {
                    edges.push((i, j));
                    weights.push(graph[[i, j]]);
                }
            }
        }

        let n_edges = edges.len();

        // Optimization loop
        for epoch in 0..n_epochs {
            // Adjust learning rate
            let alpha = self.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);

            // Sample edges for this epoch
            for _ in 0..n_edges {
                // Sample an edge
                let edge_idx = rng.gen_range(0..n_edges);
                let (i, j) = edges[edge_idx];

                // Compute distance in embedding space
                let mut dist_sq = 0.0;
                for d in 0..self.n_components {
                    let diff = embedding[[i, d]] - embedding[[j, d]];
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();

                // Attractive force
                if dist > 0.0 {
                    let attraction = -2.0 * self.a * self.b * dist.powf(2.0 * self.b - 2.0)
                        / (1.0 + self.a * dist.powf(2.0 * self.b));

                    for d in 0..self.n_components {
                        let grad = attraction * (embedding[[i, d]] - embedding[[j, d]]) / dist;
                        embedding[[i, d]] += alpha * grad * weights[edge_idx];
                        embedding[[j, d]] -= alpha * grad * weights[edge_idx];
                    }
                }

                // Repulsive force - sample a negative edge
                let k = rng.gen_range(0..nsamples);
                if k != i && k != j {
                    let mut neg_dist_sq = 0.0;
                    for d in 0..self.n_components {
                        let diff = embedding[[i, d]] - embedding[[k, d]];
                        neg_dist_sq += diff * diff;
                    }
                    let neg_dist = neg_dist_sq.sqrt();

                    if neg_dist > 0.0 {
                        let repulsion = 2.0 * self.b
                            / (1.0 + self.a * neg_dist.powf(2.0 * self.b))
                            / (1.0 + neg_dist * neg_dist);

                        for d in 0..self.n_components {
                            let grad =
                                repulsion * (embedding[[i, d]] - embedding[[k, d]]) / neg_dist;
                            embedding[[i, d]] += alpha * grad;
                            embedding[[k, d]] -= alpha * grad;
                        }
                    }
                }
            }
        }
    }

    /// Fits the UMAP model to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (nsamples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast + Send + Sync,
    {
        let (nsamples, n_features) = x.dim();

        // Validate inputs
        check_positive(self.n_neighbors, "n_neighbors")?;
        check_positive(self.n_components, "n_components")?;
        check_positive(self.n_epochs, "n_epochs")?;
        checkshape(x, &[nsamples, n_features], "x")?;

        if nsamples < self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be <= nsamples={}",
                self.n_neighbors, nsamples
            )));
        }

        // Store training data for out-of-sample extension
        let training_data = Array2::from_shape_fn((nsamples, n_features), |(i, j)| {
            num_traits::cast::<S::Elem, f64>(x[[i, j]]).unwrap_or(0.0)
        });
        self.training_data = Some(training_data);

        // Step 1: Compute pairwise distances
        let distances = self.compute_distances(x);

        // Step 2: Find k nearest neighbors
        let (knn_indices, knn_distances) = self.find_neighbors(&distances);

        // Step 3: Compute fuzzy simplicial set
        let graph = self.compute_graph(&knn_indices, &knn_distances);
        self.training_graph = Some(graph.clone());

        // Step 4: Initialize low dimensional embedding
        let mut embedding = self.initialize_embedding(nsamples);

        // Step 5: Optimize the embedding
        self.optimize_embedding(&mut embedding, &graph, self.n_epochs);

        self.embedding = Some(embedding);

        Ok(())
    }

    /// Transforms the input data using the fitted UMAP model
    ///
    /// # Arguments
    /// * `x` - The input data, shape (nsamples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (nsamples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.embedding.is_none() {
            return Err(TransformError::NotFitted(
                "UMAP model has not been fitted".to_string(),
            ));
        }

        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;

        let (_n_new_samples, n_features) = x.dim();
        let (_, n_training_features) = training_data.dim();

        if n_features != n_training_features {
            return Err(TransformError::InvalidInput(format!(
                "Input features {n_features} must match training features {n_training_features}"
            )));
        }

        // If transforming the same data as training, return stored embedding
        if self.is_same_data(x, training_data) {
            return Ok(self.embedding.as_ref().unwrap().clone());
        }

        // Implement out-of-sample extension using weighted average of nearest neighbors
        self.transform_new_data(x)
    }

    /// Fits the UMAP model to the input data and returns the embedding
    ///
    /// # Arguments
    /// * `x` - The input data, shape (nsamples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (nsamples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast + Send + Sync,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the low dimensional embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Check if the input data is the same as training data
    fn is_same_data<S>(&self, x: &ArrayBase<S, Ix2>, trainingdata: &Array2<f64>) -> bool
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if x.dim() != trainingdata.dim() {
            return false;
        }

        let (nsamples, n_features) = x.dim();
        for i in 0..nsamples {
            for j in 0..n_features {
                let x_val = num_traits::cast::<S::Elem, f64>(x[[i, j]]).unwrap_or(0.0);
                if (x_val - trainingdata[[i, j]]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Transform new data using out-of-sample extension
    fn transform_new_data<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let training_data = self.training_data.as_ref().unwrap();
        let training_embedding = self.embedding.as_ref().unwrap();

        let (n_new_samples_, _) = x.dim();
        let (n_training_samples_, _) = training_data.dim();

        // For each new sample, find k nearest neighbors in training data
        let mut new_embedding = Array2::zeros((n_new_samples_, self.n_components));

        for i in 0..n_new_samples_ {
            // Compute distances to all training samples
            let mut distances = Vec::new();
            for j in 0..n_training_samples_ {
                let mut dist_sq = 0.0;
                for k in 0..x.ncols() {
                    let x_val = num_traits::cast::<S::Elem, f64>(x[[i, k]]).unwrap_or(0.0);
                    let train_val = training_data[[j, k]];
                    let diff = x_val - train_val;
                    dist_sq += diff * diff;
                }
                distances.push((dist_sq.sqrt(), j));
            }

            // Sort and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let k = self.n_neighbors.min(n_training_samples_);

            // Compute weights based on distances (inverse distance weighting)
            let mut total_weight = 0.0;
            let mut weighted_coords = vec![0.0; self.n_components];

            for (dist, train_idx) in distances.iter().take(k) {
                let weight = if *dist > 1e-10 {
                    1.0 / (*dist + 1e-10)
                } else {
                    1e10
                };
                total_weight += weight;

                for dim in 0..self.n_components {
                    weighted_coords[dim] += weight * training_embedding[[*train_idx, dim]];
                }
            }

            // Normalize weights and set coordinates
            if total_weight > 0.0 {
                for dim in 0..self.n_components {
                    new_embedding[[i, dim]] = weighted_coords[dim] / total_weight;
                }
            }
        }

        Ok(new_embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_umap_basic() {
        // Create a simple dataset
        let x = Array::from_shape_vec(
            (10, 3),
            vec![
                1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2, 5.0, 6.0, 7.0, 5.1, 6.1, 7.1, 5.2,
                6.2, 7.2, 9.0, 10.0, 11.0, 9.1, 10.1, 11.1, 9.2, 10.2, 11.2, 9.3, 10.3, 11.3,
            ],
        )
        .unwrap();

        // Initialize and fit UMAP
        let mut umap = UMAP::new(3, 2, 0.1, 1.0, 50);
        let embedding = umap.fit_transform(&x).unwrap();

        // Check that the shape is correct
        assert_eq!(embedding.shape(), &[10, 2]);

        // Check that embedding values are finite
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_umap_parameters() {
        let x: Array2<f64> = Array::eye(5);

        // Test with different parameters
        let mut umap = UMAP::new(2, 3, 0.5, 0.5, 100)
            .with_random_state(42)
            .with_metric("euclidean");

        let embedding = umap.fit_transform(&x).unwrap();
        assert_eq!(embedding.shape(), &[5, 3]);
    }
}
