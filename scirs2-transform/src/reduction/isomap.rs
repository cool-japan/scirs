//! Isomap (Isometric Feature Mapping) for non-linear dimensionality reduction
//!
//! Isomap is a non-linear dimensionality reduction method that preserves geodesic
//! distances between all points. It extends MDS by using geodesic distances instead
//! of Euclidean distances.

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::eigh;
use std::collections::BinaryHeap;
use std::f64;

use crate::error::{Result, TransformError};
// use statrs::statistics::Statistics; // TODO: Add statrs dependency - needs generic type fixes

/// Isomap (Isometric Feature Mapping) dimensionality reduction
///
/// Isomap seeks a lower-dimensional embedding that maintains geodesic distances
/// between all points. It uses graph distances to approximate geodesic distances
/// on the manifold.
#[derive(Debug, Clone)]
pub struct Isomap {
    /// Number of neighbors to use for graph construction
    n_neighbors: usize,
    /// Number of components for dimensionality reduction
    n_components: usize,
    /// Whether to use k-neighbors or epsilon-ball for graph construction
    neighbor_mode: String,
    /// Epsilon for epsilon-ball graph construction
    epsilon: f64,
    /// The embedding vectors
    embedding: Option<Array2<f64>>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
    /// Geodesic distances from training data
    geodesic_distances: Option<Array2<f64>>,
}

impl Isomap {
    /// Creates a new Isomap instance
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighbors for graph construction
    /// * `n_components` - Number of dimensions in the embedding space
    pub fn new(n_neighbors: usize, ncomponents: usize) -> Self {
        Isomap {
            n_neighbors,
            n_components: ncomponents,
            neighbor_mode: "knn".to_string(),
            epsilon: 0.0,
            embedding: None,
            training_data: None,
            geodesic_distances: None,
        }
    }

    /// Use epsilon-ball instead of k-nearest neighbors
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.neighbor_mode = "epsilon".to_string();
        self.epsilon = epsilon;
        self
    }

    /// Compute pairwise Euclidean distances
    fn compute_distances<S>(&self, x: &ArrayBase<S, Ix2>) -> Array2<f64>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let mut dist = 0.0;
                for k in 0..x.shape()[1] {
                    let diff = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                        - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                    dist += diff * diff;
                }
                dist = dist.sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        distances
    }

    /// Construct the neighborhood graph
    fn construct_graph(&self, distances: &Array2<f64>) -> Array2<f64> {
        let n_samples = distances.shape()[0];
        let mut graph = Array2::from_elem((n_samples, n_samples), f64::INFINITY);

        // Set diagonal to 0
        for i in 0..n_samples {
            graph[[i, i]] = 0.0;
        }

        if self.neighbor_mode == "knn" {
            // K-nearest neighbors graph
            for i in 0..n_samples {
                // Find k nearest neighbors
                let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();

                for j in 0..n_samples {
                    if i != j {
                        let dist_fixed = (distances[[i, j]] * 1e9) as i64;
                        heap.push((std::cmp::Reverse(dist_fixed), j));
                    }
                }

                // Connect to k nearest neighbors
                for _ in 0..self.n_neighbors {
                    if let Some((_, j)) = heap.pop() {
                        graph[[i, j]] = distances[[i, j]];
                        graph[[j, i]] = distances[[j, i]]; // Make symmetric
                    }
                }
            }
        } else {
            // Epsilon-ball graph
            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    if distances[[i, j]] <= self.epsilon {
                        graph[[i, j]] = distances[[i, j]];
                        graph[[j, i]] = distances[[j, i]];
                    }
                }
            }
        }

        graph
    }

    /// Compute shortest paths using Floyd-Warshall algorithm
    fn compute_shortest_paths(&self, graph: &Array2<f64>) -> Result<Array2<f64>> {
        let n = graph.shape()[0];
        let mut dist = graph.clone();

        // Floyd-Warshall algorithm
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if dist[[i, k]] + dist[[k, j]] < dist[[i, j]] {
                        dist[[i, j]] = dist[[i, k]] + dist[[k, j]];
                    }
                }
            }
        }

        // Check if graph is connected
        for i in 0..n {
            for j in 0..n {
                if dist[[i, j]].is_infinite() {
                    return Err(TransformError::InvalidInput(
                        "Graph is not connected. Try increasing n_neighbors or epsilon."
                            .to_string(),
                    ));
                }
            }
        }

        Ok(dist)
    }

    /// Apply classical MDS to the geodesic distance matrix
    fn classical_mds(&self, distances: &Array2<f64>) -> Result<Array2<f64>> {
        let n = distances.shape()[0];

        // Double center the squared distance matrix
        let squared_distances = distances.mapv(|d| d * d);

        // Row means
        let row_means = squared_distances.mean_axis(Axis(1)).unwrap();

        // Column means
        let col_means = squared_distances.mean_axis(Axis(0)).unwrap();

        // Grand mean
        let grand_mean = row_means.mean().unwrap();

        // Double centering
        let mut gram = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                gram[[i, j]] =
                    -0.5 * (squared_distances[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        // Ensure symmetry by averaging with transpose (fixes floating point errors)
        let gram_symmetric = 0.5 * (&gram + &gram.t());

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = match eigh(&gram_symmetric.view(), None) {
            Ok(result) => result,
            Err(e) => return Err(TransformError::LinalgError(e)),
        };

        // Sort eigenvalues and eigenvectors in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Extract the top n_components eigenvectors
        let mut embedding = Array2::zeros((n, self.n_components));
        for j in 0..self.n_components {
            let idx = indices[j];
            let scale = eigenvalues[idx].max(0.0).sqrt();

            for i in 0..n {
                embedding[[i, j]] = eigenvectors[[i, idx]] * scale;
            }
        }

        Ok(embedding)
    }

    /// Fits the Isomap model to the input data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Err otherwise
    pub fn fit<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let (n_samples, n_features) = x.dim();

        // Validate inputs
        check_positive(self.n_neighbors, "n_neighbors")?;
        check_positive(self.n_components, "n_components")?;
        checkshape(x, &[n_samples, n_features], "x")?;

        if n_samples < self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be <= n_samples={}",
                self.n_neighbors, n_samples
            )));
        }

        if self.n_components >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be < n_samples={}",
                self.n_components, n_samples
            )));
        }

        // Convert input to f64
        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Step 1: Compute pairwise distances
        let distances = self.compute_distances(&x_f64.view());

        // Step 2: Construct neighborhood graph
        let graph = self.construct_graph(&distances);

        // Step 3: Compute shortest paths (geodesic distances)
        let geodesic_distances = self.compute_shortest_paths(&graph)?;

        // Step 4: Apply classical MDS
        let embedding = self.classical_mds(&geodesic_distances)?;

        self.embedding = Some(embedding);
        self.training_data = Some(x_f64);
        self.geodesic_distances = Some(geodesic_distances);

        Ok(())
    }

    /// Transforms the input data using the fitted Isomap model
    ///
    /// For new points, this uses the Landmark MDS approach
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
                "Isomap model has not been fitted".to_string(),
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

        // Implement Landmark MDS for out-of-sample extension
        self.landmark_mds(&x_f64)
    }

    /// Fits the Isomap model and transforms the data
    ///
    /// # Arguments
    /// * `x` - The input data, shape (n_samples, n_features)
    ///
    /// # Returns
    /// * `Result<Array2<f64>>` - The transformed data, shape (n_samples, n_components)
    pub fn fit_transform<S>(&mut self, x: &ArrayBase<S, Ix2>) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        self.fit(x)?;
        self.transform(x)
    }

    /// Returns the embedding
    pub fn embedding(&self) -> Option<&Array2<f64>> {
        self.embedding.as_ref()
    }

    /// Returns the geodesic distances computed during fitting
    pub fn geodesic_distances(&self) -> Option<&Array2<f64>> {
        self.geodesic_distances.as_ref()
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

    /// Implement Landmark MDS for out-of-sample extension
    fn landmark_mds(&self, xnew: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self.training_data.as_ref().unwrap();
        let training_embedding = self.embedding.as_ref().unwrap();
        let geodesic_distances = self.geodesic_distances.as_ref().unwrap();

        let (n_new, n_features) = xnew.dim();
        let (n_training_, _) = training_data.dim();

        if n_features != training_data.ncols() {
            return Err(TransformError::InvalidInput(format!(
                "Input features {} must match training features {}",
                n_features,
                training_data.ncols()
            )));
        }

        // Step 1: Compute distances from _new points to all training points
        let mut distances_to_training = Array2::zeros((n_new, n_training_));
        for i in 0..n_new {
            for j in 0..n_training_ {
                let mut dist_sq = 0.0;
                for k in 0..n_features {
                    let diff = xnew[[i, k]] - training_data[[j, k]];
                    dist_sq += diff * diff;
                }
                distances_to_training[[i, j]] = dist_sq.sqrt();
            }
        }

        // Step 2: Apply Landmark MDS algorithm
        // For each _new point, find its coordinates that minimize stress
        // with respect to the known training points
        let mut new_embedding = Array2::zeros((n_new, self.n_components));

        for i in 0..n_new {
            // Use weighted least squares to find optimal coordinates
            let coords = self.solve_landmark_coordinates(
                &distances_to_training.row(i),
                training_embedding,
                geodesic_distances,
            )?;

            for j in 0..self.n_components {
                new_embedding[[i, j]] = coords[j];
            }
        }

        Ok(new_embedding)
    }

    /// Solve for landmark coordinates using weighted least squares
    fn solve_landmark_coordinates(
        &self,
        distances_to_landmarks: &scirs2_core::ndarray::ArrayView1<f64>,
        landmark_embedding: &Array2<f64>,
        _geodesic_distances: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let n_landmarks = landmark_embedding.nrows();

        // Use a subset of _landmarks for efficiency (select k nearest)
        let k_landmarks = (n_landmarks / 2)
            .max(self.n_components + 1)
            .min(n_landmarks);

        // Find k nearest _landmarks
        let mut landmark_dists: Vec<(f64, usize)> = distances_to_landmarks
            .indexed_iter()
            .map(|(idx, &dist)| (dist, idx))
            .collect();
        landmark_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Use the k nearest _landmarks
        let selected_landmarks: Vec<usize> = landmark_dists
            .into_iter()
            .take(k_landmarks)
            .map(|(_, idx)| idx)
            .collect();

        // Build system: A * x = b where x are the coordinates
        // Using the constraint that _distances in _embedding space should
        // approximate geodesic _distances
        let mut a = Array2::zeros((k_landmarks, self.n_components));
        let mut b = Array1::zeros(k_landmarks);
        let mut weights = Array1::zeros(k_landmarks);

        // For each selected landmark, create a constraint equation
        for (row_idx, &landmark_idx) in selected_landmarks.iter().enumerate() {
            let dist_to_landmark = distances_to_landmarks[landmark_idx];
            let weight = if dist_to_landmark > 1e-10 {
                1.0 / (dist_to_landmark + 1e-10)
            } else {
                1e10
            };
            weights[row_idx] = weight;

            // Target distance is the distance from new point to this landmark
            b[row_idx] = dist_to_landmark * weight;

            // Coefficients are the landmark coordinates (weighted)
            for dim in 0..self.n_components {
                a[[row_idx, dim]] = landmark_embedding[[landmark_idx, dim]] * weight;
            }
        }

        // Solve weighted least squares: A^T W A x = A^T W b
        // where W is the diagonal weight matrix
        let mut at_wa = Array2::zeros((self.n_components, self.n_components));
        let mut at_wb = Array1::zeros(self.n_components);

        for i in 0..self.n_components {
            for j in 0..self.n_components {
                for k in 0..k_landmarks {
                    at_wa[[i, j]] += a[[k, i]] * weights[k] * a[[k, j]];
                }
            }
            for k in 0..k_landmarks {
                at_wb[i] += a[[k, i]] * weights[k] * b[k];
            }
        }

        // Add regularization to prevent singular matrix
        for i in 0..self.n_components {
            at_wa[[i, i]] += 1e-10;
        }

        // Solve using simple Gaussian elimination for small systems
        self.solve_linear_system(&at_wa, &at_wb)
    }

    /// Simple linear system solver for small matrices
    fn solve_linear_system(&self, a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
        let n = a.nrows();
        let mut a_copy = a.clone();
        let mut b_copy = b.clone();

        // Gaussian elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if a_copy[[k, i]].abs() > a_copy[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..n {
                    let temp = a_copy[[i, j]];
                    a_copy[[i, j]] = a_copy[[max_row, j]];
                    a_copy[[max_row, j]] = temp;
                }
                let temp = b_copy[i];
                b_copy[i] = b_copy[max_row];
                b_copy[max_row] = temp;
            }

            // Check for singular matrix
            if a_copy[[i, i]].abs() < 1e-12 {
                return Err(TransformError::ComputationError(
                    "Singular matrix in landmark MDS".to_string(),
                ));
            }

            // Eliminate
            for k in i + 1..n {
                let factor = a_copy[[k, i]] / a_copy[[i, i]];
                for j in i..n {
                    a_copy[[k, j]] -= factor * a_copy[[i, j]];
                }
                b_copy[k] -= factor * b_copy[i];
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = b_copy[i];
            for j in i + 1..n {
                x[i] -= a_copy[[i, j]] * x[j];
            }
            x[i] /= a_copy[[i, i]];
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_isomap_basic() {
        // Create a simple S-curve dataset
        let n_points = 20;
        let mut data = Vec::new();

        for i in 0..n_points {
            let t = i as f64 / n_points as f64 * 3.0 * std::f64::consts::PI;
            let x = t.sin();
            let y = 2.0 * (i as f64 / n_points as f64);
            let z = t.cos();
            data.extend_from_slice(&[x, y, z]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).unwrap();

        // Fit Isomap
        let mut isomap = Isomap::new(5, 2);
        let embedding = isomap.fit_transform(&x).unwrap();

        // Check shape
        assert_eq!(embedding.shape(), &[n_points, 2]);

        // Check that values are finite
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_isomap_epsilon_ball() {
        let x: Array2<f64> = Array::eye(5);

        let mut isomap = Isomap::new(3, 2).with_epsilon(1.5);
        let result = isomap.fit_transform(&x);

        // This should work as the identity matrix forms a connected graph with epsilon=1.5
        assert!(result.is_ok());

        let embedding = result.unwrap();
        assert_eq!(embedding.shape(), &[5, 2]);
    }

    #[test]
    fn test_isomap_disconnected_graph() {
        // Create clearly disconnected data: two separate clusters
        let x = scirs2_core::ndarray::array![
            [0.0, 0.0],   // Cluster 1
            [0.1, 0.1],   // Cluster 1
            [10.0, 10.0], // Cluster 2 (far away)
            [10.1, 10.1], // Cluster 2
        ];

        // With only 1 neighbor, the two clusters won't connect
        let mut isomap = Isomap::new(1, 2);
        let result = isomap.fit(&x);

        // Should fail due to disconnected graph
        assert!(result.is_err());
        if let Err(e) = result {
            // Verify it's specifically a connectivity error
            match e {
                TransformError::InvalidInput(msg) => {
                    assert!(msg.contains("Graph is not connected"));
                }
                _ => panic!("Expected InvalidInput error for disconnected graph"),
            }
        }
    }
}
