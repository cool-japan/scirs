//! Locally Linear Embedding (LLE) for non-linear dimensionality reduction
//!
//! LLE is a non-linear dimensionality reduction method that assumes the data lies
//! on a low-dimensional manifold that is locally linear. It preserves local
//! neighborhood structure in the embedding space.

use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use scirs2_core::numeric::{Float, NumCast};
use scirs2_core::validation::{check_positive, checkshape};
use scirs2_linalg::{eigh, solve};
use std::collections::BinaryHeap;

use crate::error::{Result, TransformError};

/// Locally Linear Embedding (LLE) dimensionality reduction
///
/// LLE finds a low-dimensional embedding that preserves local linear structure.
/// Each point is reconstructed from its neighbors with fixed weights, and the
/// embedding preserves these reconstruction weights.
#[derive(Debug, Clone)]
pub struct LLE {
    /// Number of neighbors to use
    n_neighbors: usize,
    /// Number of components in the embedding
    n_components: usize,
    /// Regularization parameter
    reg: f64,
    /// Method variant: 'standard', 'modified', 'hessian', 'ltsa'
    method: String,
    /// The embedding
    embedding: Option<Array2<f64>>,
    /// Reconstruction weights
    weights: Option<Array2<f64>>,
    /// Training data for out-of-sample extension
    training_data: Option<Array2<f64>>,
}

impl LLE {
    /// Creates a new LLE instance
    ///
    /// # Arguments
    /// * `n_neighbors` - Number of neighbors to use
    /// * `n_components` - Number of dimensions in the embedding
    pub fn new(n_neighbors: usize, ncomponents: usize) -> Self {
        LLE {
            n_neighbors,
            n_components: ncomponents,
            reg: 1e-3,
            method: "standard".to_string(),
            embedding: None,
            weights: None,
            training_data: None,
        }
    }

    /// Set the regularization parameter
    pub fn with_regularization(mut self, reg: f64) -> Self {
        self.reg = reg;
        self
    }

    /// Set the LLE method variant
    pub fn with_method(mut self, method: &str) -> Self {
        self.method = method.to_string();
        self
    }

    /// Find k nearest neighbors for each point
    fn find_neighbors<S>(&self, x: &ArrayBase<S, Ix2>) -> (Array2<usize>, Array2<f64>)
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let mut indices = Array2::zeros((n_samples, self.n_neighbors));
        let mut distances = Array2::zeros((n_samples, self.n_neighbors));

        for i in 0..n_samples {
            // Compute distances to all other points
            let mut heap: BinaryHeap<(std::cmp::Reverse<i64>, usize)> = BinaryHeap::new();

            for j in 0..n_samples {
                if i != j {
                    let mut dist = 0.0;
                    for k in 0..x.shape()[1] {
                        let diff = NumCast::from(x[[i, k]]).unwrap_or(0.0)
                            - NumCast::from(x[[j, k]]).unwrap_or(0.0);
                        dist += diff * diff;
                    }
                    dist = dist.sqrt();

                    let dist_fixed = (dist * 1e9) as i64;
                    heap.push((std::cmp::Reverse(dist_fixed), j));
                }
            }

            // Extract k nearest neighbors
            for j in 0..self.n_neighbors {
                if let Some((std::cmp::Reverse(dist_fixed), idx)) = heap.pop() {
                    indices[[i, j]] = idx;
                    distances[[i, j]] = dist_fixed as f64 / 1e9;
                }
            }
        }

        (indices, distances)
    }

    /// Compute reconstruction weights
    fn compute_weights<S>(
        &self,
        x: &ArrayBase<S, Ix2>,
        neighbors: &Array2<usize>,
    ) -> Result<Array2<f64>>
    where
        S: Data,
        S::Elem: Float + NumCast,
    {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let k = self.n_neighbors;

        let mut weights = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            // Create local covariance matrix
            let mut c = Array2::zeros((k, k));
            let xi = x.index_axis(Axis(0), i);

            // Build the local covariance matrix
            for j in 0..k {
                let neighbor_j = neighbors[[i, j]];
                let xj = x.index_axis(Axis(0), neighbor_j);

                for l in 0..k {
                    let neighbor_l = neighbors[[i, l]];
                    let xl = x.index_axis(Axis(0), neighbor_l);

                    let mut dot = 0.0;
                    for m in 0..n_features {
                        let diff_j = NumCast::from(xi[m] - xj[m]).unwrap_or(0.0);
                        let diff_l = NumCast::from(xi[m] - xl[m]).unwrap_or(0.0);
                        dot += diff_j * diff_l;
                    }
                    c[[j, l]] = dot;
                }
            }

            // Add regularization to diagonal
            let trace = (0..k).map(|j| c[[j, j]]).sum::<f64>();
            let reg_value = self.reg * trace / k as f64;
            for j in 0..k {
                c[[j, j]] += reg_value;
            }

            // Solve C * w = 1 for weights
            let ones = Array1::ones(k);
            let w = match solve(&c.view(), &ones.view(), None) {
                Ok(solution) => solution,
                Err(_) => {
                    // If solving fails, use uniform weights
                    Array1::from_elem(k, 1.0 / k as f64)
                }
            };

            // Normalize weights to sum to 1
            let w_sum = w.sum();
            let w_normalized = if w_sum.abs() > 1e-10 {
                w / w_sum
            } else {
                Array1::from_elem(k, 1.0 / k as f64)
            };

            // Store weights in sparse format
            for j in 0..k {
                let neighbor = neighbors[[i, j]];
                weights[[i, neighbor]] = w_normalized[j];
            }
        }

        Ok(weights)
    }

    /// Compute the embedding from reconstruction weights
    fn compute_embedding(&self, weights: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = weights.shape()[0];

        // Construct the cost matrix M = (I - W)^T (I - W)
        let mut m = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut sum = 0.0;

                if i == j {
                    // Diagonal element
                    sum += 1.0 - 2.0 * weights[[i, j]] + weights.column(j).dot(&weights.column(j));
                } else {
                    // Off-diagonal element
                    sum += -weights[[i, j]] - weights[[j, i]]
                        + weights.column(i).dot(&weights.column(j));
                }

                m[[i, j]] = sum;
            }
        }

        // Find the eigenvectors corresponding to the smallest eigenvalues
        let (eigenvalues, eigenvectors) = match eigh(&m.view(), None) {
            Ok(result) => result,
            Err(e) => return Err(TransformError::LinalgError(e)),
        };

        // Sort eigenvalues and eigenvectors
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap());

        // Skip the first eigenvector (corresponding to eigenvalue 0)
        // and take the next n_components eigenvectors
        let mut embedding = Array2::zeros((n_samples, self.n_components));
        for j in 0..self.n_components {
            let idx = indices[j + 1]; // Skip first eigenvector
            for i in 0..n_samples {
                embedding[[i, j]] = eigenvectors[[i, idx]];
            }
        }

        Ok(embedding)
    }

    /// Fits the LLE model to the input data
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

        if n_samples <= self.n_neighbors {
            return Err(TransformError::InvalidInput(format!(
                "n_neighbors={} must be < n_samples={}",
                self.n_neighbors, n_samples
            )));
        }

        if self.n_components >= n_samples {
            return Err(TransformError::InvalidInput(format!(
                "n_components={} must be < n_samples={}",
                self.n_components, n_samples
            )));
        }

        // Convert to f64
        let x_f64 = x.mapv(|v| NumCast::from(v).unwrap_or(0.0));

        // Step 1: Find k nearest neighbors
        let (neighbors, distances) = self.find_neighbors(&x_f64.view());

        // Step 2: Compute reconstruction weights
        let weights = self.compute_weights(&x_f64.view(), &neighbors)?;

        // Step 3: Compute embedding from weights
        let embedding = self.compute_embedding(&weights)?;

        self.embedding = Some(embedding);
        self.weights = Some(weights);
        self.training_data = Some(x_f64);

        Ok(())
    }

    /// Transforms the input data using the fitted LLE model
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
                "LLE model has not been fitted".to_string(),
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

        // Implement out-of-sample extension
        self.transform_new_data(&x_f64)
    }

    /// Fits the LLE model and transforms the data
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

    /// Returns the reconstruction weights
    pub fn reconstruction_weights(&self) -> Option<&Array2<f64>> {
        self.weights.as_ref()
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

    /// Transform new data using out-of-sample extension
    fn transform_new_data(&self, xnew: &Array2<f64>) -> Result<Array2<f64>> {
        let training_data = self.training_data.as_ref().unwrap();
        let training_embedding = self.embedding.as_ref().unwrap();

        let (n_new, n_features) = xnew.dim();
        let (_n_training_, _) = training_data.dim();

        if n_features != training_data.ncols() {
            return Err(TransformError::InvalidInput(format!(
                "Input features {} must match training features {}",
                n_features,
                training_data.ncols()
            )));
        }

        let mut new_embedding = Array2::zeros((n_new, self.n_components));

        // For each _new point, find its reconstruction weights w.r.t. training data
        // and use these weights to compute its embedding coordinates
        for i in 0..n_new {
            let new_coords =
                self.compute_new_point_embedding(&xnew.row(i), training_data, training_embedding)?;

            for j in 0..self.n_components {
                new_embedding[[i, j]] = new_coords[j];
            }
        }

        Ok(new_embedding)
    }

    /// Compute embedding coordinates for a single new point
    fn compute_new_point_embedding(
        &self,
        x_new: &scirs2_core::ndarray::ArrayView1<f64>,
        training_data: &Array2<f64>,
        training_embedding: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let n_training = training_data.nrows();
        let n_features = training_data.ncols();

        // Step 1: Find k nearest neighbors in training _data
        let mut distances: Vec<(f64, usize)> = Vec::new();
        for j in 0..n_training {
            let mut dist_sq = 0.0;
            for k in 0..n_features {
                let diff = x_new[k] - training_data[[j, k]];
                dist_sq += diff * diff;
            }
            distances.push((dist_sq.sqrt(), j));
        }

        // Sort by distance and take k nearest
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let k = self.n_neighbors.min(n_training);
        let neighbor_indices: Vec<usize> =
            distances.into_iter().take(k).map(|(_, idx)| idx).collect();

        // Step 2: Compute reconstruction weights
        let weights =
            self.compute_reconstruction_weights_for_point(x_new, training_data, &neighbor_indices)?;

        // Step 3: Compute _embedding as weighted combination of neighbor embeddings
        let mut new_coords = Array1::zeros(self.n_components);
        for (i, &neighbor_idx) in neighbor_indices.iter().enumerate() {
            for dim in 0..self.n_components {
                new_coords[dim] += weights[i] * training_embedding[[neighbor_idx, dim]];
            }
        }

        Ok(new_coords)
    }

    /// Compute reconstruction weights for a single point given its neighbors
    fn compute_reconstruction_weights_for_point(
        &self,
        x_point: &scirs2_core::ndarray::ArrayView1<f64>,
        training_data: &Array2<f64>,
        neighbor_indices: &[usize],
    ) -> Result<Array1<f64>> {
        let k = neighbor_indices.len();
        let n_features = training_data.ncols();

        // Build local covariance matrix C
        let mut c = Array2::zeros((k, k));

        for i in 0..k {
            let neighbor_i = neighbor_indices[i];
            for j in 0..k {
                let neighbor_j = neighbor_indices[j];

                let mut dot = 0.0;
                for m in 0..n_features {
                    let diff_i = x_point[m] - training_data[[neighbor_i, m]];
                    let diff_j = x_point[m] - training_data[[neighbor_j, m]];
                    dot += diff_i * diff_j;
                }
                c[[i, j]] = dot;
            }
        }

        // Add regularization to diagonal
        let trace = (0..k).map(|i| c[[i, i]]).sum::<f64>();
        let reg_value = self.reg * trace / k as f64;
        for i in 0..k {
            c[[i, i]] += reg_value;
        }

        // Solve C * w = 1 for weights
        let ones = Array1::ones(k);
        let w = match solve(&c.view(), &ones.view(), None) {
            Ok(solution) => solution,
            Err(_) => {
                // If solving fails, use uniform weights
                Array1::from_elem(k, 1.0 / k as f64)
            }
        };

        // Normalize weights to sum to 1
        let w_sum = w.sum();
        let w_normalized = if w_sum.abs() > 1e-10 {
            w / w_sum
        } else {
            Array1::from_elem(k, 1.0 / k as f64)
        };

        Ok(w_normalized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_lle_basic() {
        // Create a simple Swiss roll dataset
        let n_points = 30;
        let mut data = Vec::new();

        for i in 0..n_points {
            let t = 1.5 * std::f64::consts::PI * (1.0 + 2.0 * i as f64 / n_points as f64);
            let x = t * t.cos();
            let y = 10.0 * i as f64 / n_points as f64;
            let z = t * t.sin();
            data.extend_from_slice(&[x, y, z]);
        }

        let x = Array::from_shape_vec((n_points, 3), data).unwrap();

        // Fit LLE
        let mut lle = LLE::new(10, 2);
        let embedding = lle.fit_transform(&x).unwrap();

        // Check shape
        assert_eq!(embedding.shape(), &[n_points, 2]);

        // Check that values are finite
        for val in embedding.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_lle_regularization() {
        let x: Array2<f64> = Array::eye(10) * 2.0; // Scale to avoid numerical issues

        let mut lle = LLE::new(3, 2).with_regularization(0.01);
        let result = lle.fit_transform(&x);

        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.shape(), &[10, 2]);
    }

    #[test]
    fn test_lle_invalid_params() {
        let x: Array2<f64> = Array::eye(5);

        // Too many neighbors
        let mut lle = LLE::new(10, 2);
        assert!(lle.fit(&x).is_err());

        // Too many components
        let mut lle = LLE::new(2, 10);
        assert!(lle.fit(&x).is_err());
    }
}
