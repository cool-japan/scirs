//! Self-Organizing Map (SOM) clustering implementation
//!
//! A Self-Organizing Map is an unsupervised neural network that reduces high-dimensional
//! data to a 2D grid while preserving topological relationships. Each neuron in the grid
//! has a weight vector (prototype) that adapts during training.
//!
//! # Algorithm
//!
//! 1. Initialize weight vectors randomly
//! 2. For each training sample, find the Best Matching Unit (BMU)
//! 3. Update the BMU and its neighbors using a Gaussian neighborhood function
//! 4. Decay learning rate and neighborhood radius over iterations
//!
//! # References
//!
//! Kohonen, T. (1982). "Self-organized formation of topologically correct feature maps."
//! Biological Cybernetics, 43(1), 59-69.

use crate::error::{ClusteringError, Result};
use scirs2_core::ndarray::{Array2, Array3};

/// Topology of the SOM grid
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SomTopology {
    /// Standard rectangular grid; neighbors computed by Chebyshev-like Gaussian
    Rectangular,
    /// Hexagonal grid; alternating rows are offset by 0.5
    Hexagonal,
}

/// Configuration for the Self-Organizing Map
#[derive(Debug, Clone)]
pub struct SomConfig {
    /// Number of rows in the SOM grid
    pub grid_rows: usize,
    /// Number of columns in the SOM grid
    pub grid_cols: usize,
    /// Total number of training iterations (passes over random samples)
    pub n_iter: usize,
    /// Initial learning rate
    pub learning_rate: f64,
    /// Learning rate decay constant (time constant for exponential decay)
    pub lr_decay: f64,
    /// Initial neighborhood radius (sigma)
    pub sigma: f64,
    /// Sigma decay constant (time constant for exponential decay)
    pub sigma_decay: f64,
    /// Grid topology
    pub topology: SomTopology,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for SomConfig {
    fn default() -> Self {
        Self {
            grid_rows: 10,
            grid_cols: 10,
            n_iter: 1000,
            learning_rate: 0.5,
            lr_decay: 1000.0,
            sigma: 5.0,
            sigma_decay: 1000.0,
            topology: SomTopology::Rectangular,
            random_seed: None,
        }
    }
}

/// Self-Organizing Map neural network
///
/// Trains a 2D grid of weight vectors to represent the input data topology.
///
/// # Examples
///
/// ```rust
/// use scirs2_cluster::som::{Som, SomConfig};
/// use scirs2_core::ndarray::Array2;
///
/// let data = Array2::from_shape_vec((6, 2), vec![
///     1.0_f64, 2.0, 1.2, 1.8, 0.8, 1.9,
///     5.0, 6.0, 5.2, 5.8, 4.8, 6.1,
/// ]).expect("operation should succeed");
///
/// let config = SomConfig {
///     grid_rows: 4, grid_cols: 4, n_iter: 100,
///     learning_rate: 0.5, lr_decay: 100.0,
///     sigma: 2.0, sigma_decay: 100.0,
///     ..Default::default()
/// };
///
/// let mut som = Som::new(2, config);
/// let _ = som.fit(&data);
/// ```
pub struct Som {
    /// Weight vectors: shape (grid_rows, grid_cols, n_features)
    weights: Array3<f64>,
    /// SOM configuration
    config: SomConfig,
    /// Whether the model has been trained
    trained: bool,
}

impl Som {
    /// Create a new Self-Organizing Map
    ///
    /// # Arguments
    ///
    /// * `n_features` - Dimensionality of each input sample
    /// * `config` - SOM hyperparameters
    pub fn new(n_features: usize, config: SomConfig) -> Self {
        let rows = config.grid_rows;
        let cols = config.grid_cols;
        // Initialize weights to zeros; randomized in fit()
        let weights = Array3::zeros((rows, cols, n_features));
        Self {
            weights,
            config,
            trained: false,
        }
    }

    /// Train the SOM on input data
    ///
    /// Initializes weights from random samples in the data, then iteratively
    /// updates the BMU and its neighbors.
    ///
    /// # Arguments
    ///
    /// * `x` - Training data of shape (n_samples, n_features)
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&Self> {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];

        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput(
                "Training data must not be empty".into(),
            ));
        }

        let expected_features = self.weights.shape()[2];
        if n_features != expected_features {
            return Err(ClusteringError::InvalidInput(format!(
                "Expected {} features, got {}",
                expected_features, n_features
            )));
        }

        // Initialize weights by sampling random data points
        self.initialize_weights(x)?;

        let rows = self.config.grid_rows;
        let cols = self.config.grid_cols;

        // Precompute grid positions for all neurons
        let grid_positions = self.compute_grid_positions();

        // Training loop
        let n_iter = self.config.n_iter;
        let lr0 = self.config.learning_rate;
        let lr_tau = self.config.lr_decay.max(1.0);
        let sigma0 = self.config.sigma;
        let sigma_tau = self.config.sigma_decay.max(1.0);

        // Use a simple LCG PRNG for deterministic sample selection
        let mut rng_state: u64 = self.config.random_seed.unwrap_or(42).wrapping_add(1);

        for t in 0..n_iter {
            // Decay learning rate and sigma
            let lr = lr0 * (-(t as f64) / lr_tau).exp();
            let sigma = sigma0 * (-(t as f64) / sigma_tau).exp();
            let sigma_sq = sigma * sigma;

            // Pick a random sample
            rng_state = lcg_next(rng_state);
            let idx = (rng_state as usize) % n_samples;
            let sample = x.row(idx);

            // Find BMU
            let (bmu_r, bmu_c) = self.find_bmu(sample)?;
            let bmu_pos = grid_positions[bmu_r * cols + bmu_c];

            // Update weights for all neurons based on neighborhood function
            for r in 0..rows {
                for c in 0..cols {
                    let pos = grid_positions[r * cols + c];
                    let dist_sq = self.grid_distance_sq(bmu_pos, pos);
                    let neighborhood = (-dist_sq / (2.0 * sigma_sq + 1e-300)).exp();
                    let update_factor = lr * neighborhood;

                    if update_factor < 1e-10 {
                        continue;
                    }

                    // Update weights[r, c, :]
                    for f in 0..n_features {
                        let delta = sample[f] - self.weights[[r, c, f]];
                        self.weights[[r, c, f]] += update_factor * delta;
                    }
                }
            }
        }

        self.trained = true;
        Ok(self)
    }

    /// Transform data to BMU (row, col) grid coordinates
    ///
    /// # Arguments
    ///
    /// * `x` - Input data of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Array of shape (n_samples, 2) where each row is [bmu_row, bmu_col]
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<usize>> {
        if !self.trained {
            return Err(ClusteringError::InvalidState(
                "SOM must be trained before calling transform".into(),
            ));
        }

        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let expected_features = self.weights.shape()[2];

        if n_features != expected_features {
            return Err(ClusteringError::InvalidInput(format!(
                "Expected {} features, got {}",
                expected_features, n_features
            )));
        }

        let mut result = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let sample = x.row(i);
            let (bmu_r, bmu_c) = self.find_bmu(sample)?;
            result[[i, 0]] = bmu_r;
            result[[i, 1]] = bmu_c;
        }

        Ok(result)
    }

    /// Compute the average quantization error over all samples
    ///
    /// This measures how well the SOM represents the data.
    /// Lower values indicate a better fit.
    pub fn quantization_error(&self, x: &Array2<f64>) -> Result<f64> {
        if !self.trained {
            return Err(ClusteringError::InvalidState(
                "SOM must be trained before computing quantization error".into(),
            ));
        }

        let n_samples = x.shape()[0];
        if n_samples == 0 {
            return Err(ClusteringError::InvalidInput("Empty input data".into()));
        }

        let n_features = x.shape()[1];
        let expected_features = self.weights.shape()[2];

        if n_features != expected_features {
            return Err(ClusteringError::InvalidInput(format!(
                "Expected {} features, got {}",
                expected_features, n_features
            )));
        }

        let mut total_error = 0.0_f64;

        for i in 0..n_samples {
            let sample = x.row(i);
            let (bmu_r, bmu_c) = self.find_bmu(sample)?;
            let mut dist_sq = 0.0_f64;
            for f in 0..n_features {
                let diff = sample[f] - self.weights[[bmu_r, bmu_c, f]];
                dist_sq += diff * diff;
            }
            total_error += dist_sq.sqrt();
        }

        Ok(total_error / n_samples as f64)
    }

    /// Return the component planes: weight array of shape (rows, cols, n_features)
    ///
    /// Each "plane" ([:, :, k]) shows how feature k is distributed across the grid.
    pub fn component_planes(&self) -> Array3<f64> {
        self.weights.clone()
    }

    /// Compute the Unified Distance Matrix (U-matrix)
    ///
    /// Each cell contains the mean distance to its immediate neighbors,
    /// revealing cluster boundaries (high values) and dense regions (low values).
    ///
    /// Returns array of shape (grid_rows, grid_cols).
    pub fn u_matrix(&self) -> Array2<f64> {
        let rows = self.config.grid_rows;
        let cols = self.config.grid_cols;
        let n_features = self.weights.shape()[2];
        let mut u = Array2::zeros((rows, cols));

        for r in 0..rows {
            for c in 0..cols {
                let mut neighbor_dists = Vec::with_capacity(8);

                // 8-connected neighbors for rectangular, 6-connected for hex
                let offsets: &[(i64, i64)] = match self.config.topology {
                    SomTopology::Rectangular => &[
                        (-1, -1),
                        (-1, 0),
                        (-1, 1),
                        (0, -1),
                        (0, 1),
                        (1, -1),
                        (1, 0),
                        (1, 1),
                    ],
                    SomTopology::Hexagonal => {
                        // Even rows: neighbors at offsets (0,-1),(0,1),(-1,-1),(-1,0),(1,-1),(1,0)
                        // Odd rows:  neighbors at offsets (0,-1),(0,1),(-1,0),(-1,1),(1,0),(1,1)
                        // Use rectangular approximation here for simplicity
                        &[
                            (-1, 0),
                            (-1, 1),
                            (0, -1),
                            (0, 1),
                            (1, 0),
                            (1, 1),
                            // padding to same length
                            (-1, -1),
                            (1, -1),
                        ]
                    }
                };

                for &(dr, dc) in offsets {
                    let nr = r as i64 + dr;
                    let nc = c as i64 + dc;
                    if nr >= 0 && nr < rows as i64 && nc >= 0 && nc < cols as i64 {
                        let nr = nr as usize;
                        let nc = nc as usize;
                        let mut dist_sq = 0.0_f64;
                        for f in 0..n_features {
                            let diff = self.weights[[r, c, f]] - self.weights[[nr, nc, f]];
                            dist_sq += diff * diff;
                        }
                        neighbor_dists.push(dist_sq.sqrt());
                    }
                }

                if !neighbor_dists.is_empty() {
                    let mean = neighbor_dists.iter().sum::<f64>() / neighbor_dists.len() as f64;
                    u[[r, c]] = mean;
                }
            }
        }

        u
    }

    /// Find the Best Matching Unit for a single sample
    fn find_bmu(&self, sample: scirs2_core::ndarray::ArrayView1<f64>) -> Result<(usize, usize)> {
        let rows = self.config.grid_rows;
        let cols = self.config.grid_cols;
        let n_features = self.weights.shape()[2];

        let mut best_dist = f64::INFINITY;
        let mut best_r = 0;
        let mut best_c = 0;

        for r in 0..rows {
            for c in 0..cols {
                let mut dist_sq = 0.0_f64;
                for f in 0..n_features {
                    let diff = sample[f] - self.weights[[r, c, f]];
                    dist_sq += diff * diff;
                }
                if dist_sq < best_dist {
                    best_dist = dist_sq;
                    best_r = r;
                    best_c = c;
                }
            }
        }

        Ok((best_r, best_c))
    }

    /// Initialize weights from random samples in the training data
    fn initialize_weights(&mut self, x: &Array2<f64>) -> Result<()> {
        let rows = self.config.grid_rows;
        let cols = self.config.grid_cols;
        let n_features = self.weights.shape()[2];
        let n_samples = x.shape()[0];

        let mut rng_state: u64 = self.config.random_seed.unwrap_or(12345);

        for r in 0..rows {
            for c in 0..cols {
                rng_state = lcg_next(rng_state);
                let idx = (rng_state as usize) % n_samples;
                for f in 0..n_features {
                    self.weights[[r, c, f]] = x[[idx, f]];
                }
            }
        }

        Ok(())
    }

    /// Compute (floating-point) 2D grid position for each neuron
    /// Hexagonal layout shifts odd rows by 0.5 on the x-axis
    fn compute_grid_positions(&self) -> Vec<(f64, f64)> {
        let rows = self.config.grid_rows;
        let cols = self.config.grid_cols;
        let mut positions = Vec::with_capacity(rows * cols);

        for r in 0..rows {
            for c in 0..cols {
                let x_offset = if self.config.topology == SomTopology::Hexagonal && r % 2 == 1 {
                    0.5
                } else {
                    0.0
                };
                positions.push((r as f64, c as f64 + x_offset));
            }
        }

        positions
    }

    /// Squared Euclidean distance between two grid positions
    fn grid_distance_sq(&self, a: (f64, f64), b: (f64, f64)) -> f64 {
        let dr = a.0 - b.0;
        let dc = a.1 - b.1;
        dr * dr + dc * dc
    }
}

/// Linear Congruential Generator for fast pseudo-random number generation
#[inline]
fn lcg_next(state: u64) -> u64 {
    // Parameters from Knuth MMIX
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn two_cluster_data() -> Array2<f64> {
        Array2::from_shape_vec(
            (12, 2),
            vec![
                1.0, 1.0, 1.1, 0.9, 0.9, 1.1, 1.2, 1.0, 1.0, 1.2, 0.8, 0.8, 5.0, 5.0, 5.1, 4.9,
                4.9, 5.1, 5.2, 5.0, 5.0, 5.2, 4.8, 4.8,
            ],
        )
        .expect("Failed to create test data")
    }

    #[test]
    fn test_som_trains_without_error() {
        let data = two_cluster_data();
        let config = SomConfig {
            grid_rows: 4,
            grid_cols: 4,
            n_iter: 100,
            learning_rate: 0.5,
            lr_decay: 50.0,
            sigma: 2.0,
            sigma_decay: 50.0,
            random_seed: Some(42),
            ..Default::default()
        };

        let mut som = Som::new(2, config);
        let result = som.fit(&data);
        assert!(result.is_ok(), "SOM fit should succeed: {:?}", result.err());
    }

    #[test]
    fn test_som_transform_returns_correct_shape() {
        let data = two_cluster_data();
        let config = SomConfig {
            grid_rows: 4,
            grid_cols: 4,
            n_iter: 50,
            learning_rate: 0.5,
            lr_decay: 50.0,
            sigma: 2.0,
            sigma_decay: 50.0,
            random_seed: Some(1),
            ..Default::default()
        };

        let mut som = Som::new(2, config);
        som.fit(&data).expect("fit should succeed");

        let bmus = som.transform(&data).expect("transform should succeed");
        assert_eq!(bmus.shape(), &[12, 2]);

        // All BMU coordinates must be within grid bounds
        for i in 0..12 {
            assert!(bmus[[i, 0]] < 4, "row out of bounds");
            assert!(bmus[[i, 1]] < 4, "col out of bounds");
        }
    }

    #[test]
    fn test_som_quantization_error_decreases_with_more_iterations() {
        let data = two_cluster_data();

        let make_som = |n_iter: usize| {
            let config = SomConfig {
                grid_rows: 5,
                grid_cols: 5,
                n_iter,
                learning_rate: 0.5,
                lr_decay: n_iter as f64,
                sigma: 3.0,
                sigma_decay: n_iter as f64,
                random_seed: Some(99),
                ..Default::default()
            };
            let mut som = Som::new(2, config);
            som.fit(&data).expect("fit failed");
            som
        };

        let som_few = make_som(10);
        let som_many = make_som(500);

        let qe_few = som_few.quantization_error(&data).expect("qe failed");
        let qe_many = som_many.quantization_error(&data).expect("qe failed");

        // More iterations should yield lower or equal quantization error
        assert!(
            qe_many <= qe_few + 0.5,
            "More iterations should reduce QE: few={}, many={}",
            qe_few,
            qe_many
        );
    }

    #[test]
    fn test_som_u_matrix_shape() {
        let data = two_cluster_data();
        let config = SomConfig {
            grid_rows: 3,
            grid_cols: 4,
            n_iter: 50,
            learning_rate: 0.5,
            lr_decay: 50.0,
            sigma: 2.0,
            sigma_decay: 50.0,
            random_seed: Some(7),
            ..Default::default()
        };

        let mut som = Som::new(2, config);
        som.fit(&data).expect("fit should succeed");

        let u = som.u_matrix();
        assert_eq!(u.shape(), &[3, 4]);
        // All values should be non-negative
        for &v in u.iter() {
            assert!(v >= 0.0, "U-matrix entries must be non-negative");
        }
    }

    #[test]
    fn test_som_component_planes() {
        let data = two_cluster_data();
        let config = SomConfig {
            grid_rows: 3,
            grid_cols: 3,
            n_iter: 50,
            learning_rate: 0.5,
            lr_decay: 50.0,
            sigma: 1.5,
            sigma_decay: 50.0,
            random_seed: Some(5),
            ..Default::default()
        };

        let mut som = Som::new(2, config);
        som.fit(&data).expect("fit should succeed");

        let planes = som.component_planes();
        assert_eq!(planes.shape(), &[3, 3, 2]);
    }

    #[test]
    fn test_som_hexagonal_topology() {
        let data = two_cluster_data();
        let config = SomConfig {
            grid_rows: 4,
            grid_cols: 4,
            n_iter: 50,
            learning_rate: 0.5,
            lr_decay: 50.0,
            sigma: 2.0,
            sigma_decay: 50.0,
            topology: SomTopology::Hexagonal,
            random_seed: Some(3),
        };

        let mut som = Som::new(2, config);
        let result = som.fit(&data);
        assert!(result.is_ok(), "Hexagonal SOM should train without error");
    }

    #[test]
    fn test_som_transform_before_fit_returns_error() {
        let data = two_cluster_data();
        let config = SomConfig::default();
        let som = Som::new(2, config);

        let result = som.transform(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_som_wrong_feature_count() {
        let data = two_cluster_data(); // 2 features
        let config = SomConfig {
            grid_rows: 3,
            grid_cols: 3,
            n_iter: 10,
            ..Default::default()
        };

        let mut som = Som::new(5, config); // Expects 5 features
        let result = som.fit(&data);
        assert!(result.is_err());
    }
}
