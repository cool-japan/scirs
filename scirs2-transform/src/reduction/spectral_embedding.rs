//! Spectral Embedding for non-linear dimensionality reduction
//!
//! Spectral embedding is a non-linear dimensionality reduction technique that uses
//! the eigenvectors of a graph Laplacian to embed data points in a lower-dimensional space.
//! It's particularly effective for data that lies on a non-linear manifold.

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::eigh;

use crate::error::{Result, TransformError};

/// Affinity matrix construction methods
#[derive(Debug, Clone, PartialEq)]
pub enum AffinityMethod {
    /// Gaussian (RBF) kernel with automatic bandwidth selection
    Gaussian,
    /// K-nearest neighbors graph
    KNN,
    /// Epsilon-ball graph
    Epsilon,
}

/// Spectral Embedding dimensionality reduction
///
/// Spectral embedding uses the eigenvectors of the graph Laplacian matrix
/// to find a low-dimensional representation that preserves local neighborhood structure.
#[derive(Debug, Clone)]
pub struct SpectralEmbedding {
    /// Number of components (dimensions) in the embedding
    n_components: usize,
    /// Method for constructing affinity matrix
    affinity_method: AffinityMethod,
    /// Number of neighbors for KNN graph
    n_neighbors: usize,
    /// Bandwidth parameter for Gaussian kernel (auto if None)
    gamma: Option<f64>,
    /// Epsilon parameter for epsilon-ball graph
    epsilon: f64,
    /// Whether to use normalized Laplacian
    normalized: bool,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// The embedding vectors
    embedding: Option<Array2<f64>>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Affinity matrix computed during fitting
    affinity_matrix: Option<Array2<f64>>,
    /// Eigenvectors of the Laplacian
    eigenvectors: Option<Array2<f64>>,
    /// Eigenvalues of the Laplacian
    eigenvalues: Option<Array1<f64>>,
}

impl SpectralEmbedding {
    /// Creates a new SpectralEmbedding instance
    ///
    /// # Arguments
    /// * `n_components` - Number of dimensions in the embedding space
    /// * `affinity_method` - Method for constructing the affinity matrix
    pub fn new(n_components: usize, affinitymethod: AffinityMethod) -> Self {
        SpectralEmbedding {
            n_components,
            affinity_method: affinitymethod,
            n_neighbors: 10,
            gamma: None,
            epsilon: 1.0,
            normalized: true,
            random_state: None,
            embedding: None,
            training_data: None,
            affinity_matrix: None,
            eigenvectors: None,
            eigenvalues: None,
        }
    }

    /// Set the number of neighbors for KNN graph construction
    pub fn with_n_neighbors(mut self, nneighbors: usize) -> Self {
        self.n_neighbors = nneighbors;
        self
    }

    /// Set the gamma parameter for Gaussian kernel
    pub fn with_gamma(mut self, gamma: f64) -> Self {
        self.gamma = Some(gamma);
        self
    }

    /// Set the epsilon parameter for epsilon-ball graph
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set whether to use normalized Laplacian
    pub fn with_normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Compute pairwise distances between data points
    fn compute_distances<S>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let mut dist_sq = 0.0;
                for k in 0..x.ncols() {
                    let diff = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                        - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Construct affinity matrix based on the specified method
    fn construct_affinity_matrix(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = distances.nrows();
        let mut affinity = Array2::zeros((n_samples, n_samples));

        match &self.affinity_method {
            AffinityMethod::Gaussian => {
                // Automatic bandwidth selection if not specified
                let gamma = if let Some(g) = self.gamma {
                    g
                } else {
                    // Use median heuristic: gamma = 1 / (2 * median_distance^2)
                    let mut all_distances: Vec<f64> = Vec::new();
                    for i in 0..n_samples {
                        for j in i + 1..n_samples {
                            all_distances.push(distances[[i, j]]);
                        }
                    }
                    all_distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median_dist = all_distances[all_distances.len() / 2];
                    1.0 / (2.0 * median_dist * median_dist)
                };

                // Gaussian kernel: exp(-gamma * ||x_i - x_j||^2)
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        if i != j {
                            let dist_sq = distances[[i, j]] * distances[[i, j]];
                            affinity[[i, j]] = (-gamma * dist_sq).exp();
                        }
                    }
                }
            }
            AffinityMethod::KNN => {
                // K-nearest neighbors graph
                for i in 0..n_samples {
                    // Find k nearest neighbors
                    let mut neighbors_with_dist: Vec<(f64, usize)> = Vec::new();
                    for j in 0..n_samples {
                        if i != j {
                            neighbors_with_dist.push((distances[[i, j]], j));
                        }
                    }
                    neighbors_with_dist.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    // Connect to k nearest neighbors
                    #[allow(clippy::needless_range_loop)]
                    for k in 0..self.n_neighbors.min(neighbors_with_dist.len()) {
                        let neighbor = neighbors_with_dist[k].1;
                        let weight = 1.0; // Binary weights for KNN
                        affinity[[i, neighbor]] = weight;
                        affinity[[neighbor, i]] = weight; // Make symmetric
                    }
                }
            }
            AffinityMethod::Epsilon => {
                // Epsilon-ball graph
                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        if distances[[i, j]] <= self.epsilon {
                            let weight = 1.0; // Binary weights for epsilon-ball
                            affinity[[i, j]] = weight;
                            affinity[[j, i]] = weight;
                        }
                    }
                }
            }
        }

        Ok(affinity)
    }

    /// Compute the graph Laplacian matrix
    fn compute_laplacian(&self, affinity: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = affinity.nrows();

        // Compute degree matrix
        let mut degree = Array1::zeros(n_samples);
        for i in 0..n_samples {
            degree[i] = affinity.row(i).sum();
        }

        // Check for isolated nodes
        for i in 0..n_samples {
            if degree[i] < 1e-10 {
                return Err(TransformError::ComputationError(
                    "Graph has isolated nodes. Try increasing epsilon or n_neighbors.".to_string(),
                ));
            }
        }

        let mut laplacian = Array2::zeros((n_samples, n_samples));

        if self.normalized {
            // Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i == j {
                        laplacian[[i, j]] = 1.0;
                    } else {
                        let normalized_affinity = affinity[[i, j]] / (degree[i] * degree[j]).sqrt();
                        laplacian[[i, j]] = -normalized_affinity;
                    }
                }
            }
        } else {
            // Unnormalized Laplacian: L = D - A
            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i == j {
                        laplacian[[i, j]] = degree[i];
                    } else {
                        laplacian[[i, j]] = -affinity[[i, j]];
                    }
                }
            }
        }

        Ok(laplacian)
    }

    /// Fits the SpectralEmbedding model to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast + Send + Sync,
    {
        let (n_samples, n_features) = x.dim();

        // Validate inputs
        check_positive(self.n_components, "n_components")?;
        checkshape(x, &[n_samples, n_features], "x")?;

        if self.n_components >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be < n_samples={}",
                self.n_components, n_samples
            )));
        }

        if matches!(self.affinity_method, AffinityMethod::KNN) && n_samples <= self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be < n_samples={}",
                self.n_neighbors, n_samples
            )));
        }

        // Convert input to f64
        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Step 1: Compute pairwise distances
        let distances = self.compute_distances(&x_f64.view());

        // Step 2: Construct affinity matrix
        let affinity = self.construct_affinity_matrix(&distances)?;

        // Step 3: Compute Laplacian matrix
        let laplacian = self.compute_laplacian(&affinity)?;

        // Step 4: Eigendecomposition of Laplacian
        let (eigenvalues, eigenvectors) = match eigh(&laplacian.view(), None) {
            Ok(result) => result,
            Err(e) => return Err(TransformError::LinalgError(e)),
        };

        // Step 5: Sort eigenvalues and eigenvectors in ascending order
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

        // Step 6: Select eigenvectors for embedding (skip the first if normalized)
        let start_idx = if self.normalized { 1 } else { 0 }; // Skip constant eigenvector for normalized Laplacian
        let mut embedding = Array2::zeros((n_samples, self.n_components));

        for j in 0..self.n_components {
            let idx = indices[start_idx + j];
            for i in 0..n_samples {
                embedding[[i, j]] = eigenvectors[[i, idx]];
            }
        }

        // Store results
        self.embedding = Some(embedding);
        self.training_data = Some(x_f64);
        self.affinity_matrix = Some(affinity);
        self.eigenvectors = Some(eigenvectors);
        self.eigenvalues = Some(eigenvalues);

        Ok(())
    }

    /// Transforms the input data using the fitted SpectralEmbedding model
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn transform<S>(&self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        if self.embedding.is_none() {
            return Err(TransformError::NotFitted(
                "SpectralEmbedding model has not been fitted".to_string(),
            ));
        }

        let training_data = self
            .training_data
            .as_ref()
            .ok_or_else(|| TransformError::NotFitted("Training data not available".to_string()))?;

        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Check if this is the training data
        if self.is_same_data(&x_f64, training_data) {
            return Ok(self.embedding.as_ref().unwrap().clone());
        }

        // For new data, use Nyström method for out-of-sample extension
        self.nystrom_extension(&x_f64)
    }

    /// Fits the model and transforms the data in one step
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast + Send + Sync,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Returns the affinity matrix
    pub fn affinity_matrix(&self) -> Option<&Array2<f64>> {
        self.affinity_matrix.as_ref()
    }

    /// Returns the eigenvalues of the Laplacian
    pub fn eigenvalues(&self) -> Option<&Array1<f64>> {
        self.eigenvalues.as_ref()
    }

    /// Check if the input data is the same as training data
    fn is_same_data(&self, x: &Array2<f64>, trainingdata: &Array2<f64>) -> bool {
        if x.dim() != trainingdata.dim() {
            return false;
        }

        let (n_samples, n_features) = x.dim();
        for i in 0..n_samples {
            for j in 0..n_features {
                if (x[[i, j]] - trainingdata[[i, j]]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Nyström method for out-of-sample extension
    fn nystrom_extension(&self, xnew: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self.training_data.as_ref().unwrap();
        let training_embedding = self.embedding.as_ref().unwrap();
        let eigenvalues = self.eigenvalues.as_ref().unwrap();

        let (n_new, n_features) = xnew.dim();
        let (n_training_, _) = training_data.dim();

        if n_features != training_data.ncols() {
            return Err(TransformError::InvalidInput(format!(
                "Input features {} must match training features {}",
                n_features,
                training_data.ncols()
            )));
        }

        // Compute affinity between _new points and training points
        let mut new_to_training_affinity = Array2::zeros((n_new, n_training_));

        for i in 0..n_new {
            for j in 0..n_training_ {
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = xnew[[i, k]] - training_data[[j, k]];
                    dist_sq += diff * diff;
                }

                let affinity_value = match &self.affinity_method {
                    AffinityMethod::Gaussian => {
                        let gamma = self.gamma.unwrap_or(1.0); // Use stored gamma or default
                        (-gamma * dist_sq).exp()
                    }
                    AffinityMethod::KNN | AffinityMethod::Epsilon => {
                        // For discrete methods, use distance-based weighting
                        let dist = dist_sq.sqrt();
                        if dist > 0.0 {
                            1.0 / (1.0 + dist)
                        } else {
                            1.0
                        }
                    }
                };

                new_to_training_affinity[[i, j]] = affinity_value;
            }
        }

        // Nyström approximation: Y_new = K_new_train * V * Λ^(-1)
        // where V are the eigenvectors and Λ are the eigenvalues
        let mut new_embedding = Array2::zeros((n_new, self.n_components));

        let start_idx = if self.normalized { 1 } else { 0 };

        for i in 0..n_new {
            for j in 0..self.n_components {
                let eigenvalue = eigenvalues[start_idx + j];
                if eigenvalue.abs() > 1e-10 {
                    let mut coord = 0.0;
                    for k in 0..n_training_ {
                        coord += new_to_training_affinity[[i, k]] * training_embedding[[k, j]];
                    }
                    new_embedding[[i, j]] = coord / eigenvalue.sqrt();
                }
            }
        }

        Ok(new_embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_spectral_embedding_gaussian() {
        // Create a simple 2D dataset with two clusters
        let data = vec![1.0, 1.0, 1.1, 1.1, 1.2, 0.9, 5.0, 5.0, 5.1, 5.1, 4.9, 5.2];
        let x = Array::from_shape_vec((6, 2), data).unwrap();

        let mut spectral = SpectralEmbedding::new(2, AffinityMethod::Gaussian);
        let embedding = spectral.fit_transform(&x).unwrap();

        assert_eq!(embedding.shape(), &[6, 2]);

        // Check that values are finite
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_spectral_embedding_knn() {
        let x: Array2<f64> = Array::eye(8);

        let mut spectral = SpectralEmbedding::new(3, AffinityMethod::KNN).with_n_neighbors(3);
        let embedding = spectral.fit_transform(&x).unwrap();

        assert_eq!(embedding.shape(), &[8, 3]);

        // Check that values are finite
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_spectral_embedding_out_of_sample() {
        let x_train: Array2<f64> = Array::eye(5);
        let x_test = Array::from_shape_vec(
            (2, 5),
            vec![0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0],
        )
        .unwrap();

        let mut spectral = SpectralEmbedding::new(2, AffinityMethod::Gaussian);
        spectral.fit(&x_train).unwrap();
        let test_embedding = spectral.transform(&x_test).unwrap();

        assert_eq!(test_embedding.shape(), &[2, 2]);

        // Check that values are finite
        for val in test_embedding.iter() {
            assert!(val.is_finite());
        }
    }
}
