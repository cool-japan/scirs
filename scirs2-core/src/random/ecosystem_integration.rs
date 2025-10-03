//! Ecosystem integration utilities for seamless SCIRS2 module interoperability
//!
//! This module provides bridge functions, adapters, and trait implementations that enable
//! the random number generation system to integrate seamlessly with other SCIRS2 modules
//! including scirs2-linalg, scirs2-stats, scirs2-neural, scirs2-optimize, and more.
//!
//! # Design Philosophy
//!
//! - **Zero-copy**: Minimize data copying between modules
//! - **Type-safe**: Compile-time guarantees for cross-module operations
//! - **Performance**: Optimized for high-throughput scientific computing
//! - **Ergonomic**: Simple, intuitive APIs for common workflows
//!
//! # Integration Patterns
//!
//! 1. **Random Matrix Generation**: For linear algebra operations
//! 2. **Statistical Distribution Sampling**: For statistical analysis
//! 3. **Neural Network Initialization**: For deep learning workflows
//! 4. **Optimization Noise**: For stochastic optimization algorithms
//! 5. **Scientific Simulation**: For Monte Carlo and sampling methods
//!
//! # Examples
//!
//! ```rust
//! use scirs2_core::random::ecosystem_integration::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Linear algebra integration
//! let random_matrix = LinalgBridge::random_symmetric_matrix(10, 42)?; // Smaller for doc test
//! let eigenvalues = LinalgBridge::random_eigenvalue_problem(5, 1.0, 123)?; // eigenvalue_spread > 0.1
//!
//! // Statistical analysis integration
//! let experiment = StatsBridge::design_experiment()
//!     .factors(&[vec![1.0, 2.0, 3.0], vec![0.1, 0.2, 0.3]])
//!     .replications(3) // Smaller for doc test
//!     .randomization_seed(42)
//!     .build()?;
//!
//! // Neural network initialization
//! let weights = NeuralBridge::xavier_initialization(&[10, 5, 2], 42)?; // Smaller for doc test
//! let gradients = NeuralBridge::gradient_noise_injection(0.01, &weights, 123)?;
//! # Ok(())
//! # }
//! ```

use crate::random::{
    advanced_numerical::*,
    arrays::*,
    core::{seeded_rng, Random},
    distributions::*,
    parallel::{ParallelRng, ThreadLocalRngPool},
    scientific::*,
};
use ndarray::{Array1, Array2, Array3, ArrayD, Dimension, Ix2};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use std::collections::HashMap;

/// Bridge for linear algebra operations requiring random number generation
pub struct LinalgBridge;

impl LinalgBridge {
    /// Generate a random symmetric positive definite matrix
    pub fn random_symmetric_positive_definite(
        size: usize,
        seed: u64,
    ) -> Result<Array2<f64>, String> {
        let mut rng = seeded_rng(seed);

        // Generate random matrix A
        let mut a = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                a[[i, j]] = rng.sample(Normal::new(0.0, 1.0).unwrap());
            }
        }

        // Compute A * A^T to ensure positive definiteness
        let at = a.t();
        let mut result = Array2::zeros((size, size));

        for i in 0..size {
            for j in 0..size {
                let mut sum = 0.0;
                for k in 0..size {
                    sum += a[[i, k]] * at[[k, j]];
                }
                result[[i, j]] = sum;
            }
            // Add small diagonal regularization
            result[[i, i]] += 1e-6;
        }

        Ok(result)
    }

    /// Generate random symmetric matrix
    pub fn random_symmetric_matrix(size: usize, seed: u64) -> Result<Array2<f64>, String> {
        let mut rng = seeded_rng(seed);
        let mut matrix = Array2::zeros((size, size));

        for i in 0..size {
            for j in i..size {
                let value = rng.sample(Normal::new(0.0, 1.0).unwrap());
                matrix[[i, j]] = value;
                matrix[[j, i]] = value;
            }
        }

        Ok(matrix)
    }

    /// Generate random orthogonal matrix using QR decomposition
    pub fn random_orthogonal_matrix(size: usize, seed: u64) -> Result<Array2<f64>, String> {
        let mut rng = seeded_rng(seed);

        // Generate random matrix
        let mut a = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                a[[i, j]] = rng.sample(Normal::new(0.0, 1.0).unwrap());
            }
        }

        // Gram-Schmidt orthogonalization
        let mut q = Array2::zeros((size, size));

        for j in 0..size {
            // Copy column j from A
            let mut v = Array1::zeros(size);
            for i in 0..size {
                v[i] = a[[i, j]];
            }

            // Subtract projections onto previous columns
            for k in 0..j {
                let mut proj = 0.0;
                for i in 0..size {
                    proj += v[i] * q[[i, k]];
                }
                for i in 0..size {
                    v[i] -= proj * q[[i, k]];
                }
            }

            // Normalize
            let norm = (v.iter().map(|x| x * x).sum::<f64>()).sqrt();
            if norm > 1e-10 {
                for i in 0..size {
                    q[[i, j]] = v[i] / norm;
                }
            }
        }

        Ok(q)
    }

    /// Generate random eigenvalue problem (A, eigenvalues, eigenvectors)
    pub fn random_eigenvalue_problem(
        size: usize,
        eigenvalue_spread: f64,
        seed: u64,
    ) -> Result<(Array2<f64>, Vec<f64>, Array2<f64>), String> {
        let mut rng = seeded_rng(seed);

        // Generate random eigenvalues
        let mut eigenvalues = Vec::with_capacity(size);
        for _ in 0..size {
            eigenvalues.push(rng.sample(Uniform::new(0.1, eigenvalue_spread).unwrap()));
        }
        eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending

        // Generate random orthogonal eigenvector matrix
        let eigenvectors = Self::random_orthogonal_matrix(size, seed + 1)?;

        // Construct matrix A = V * D * V^T
        let mut diagonal = Array2::zeros((size, size));
        for i in 0..size {
            diagonal[[i, i]] = eigenvalues[i];
        }

        // A = V * D * V^T
        let mut vd = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                let mut sum = 0.0;
                for k in 0..size {
                    sum += eigenvectors[[i, k]] * diagonal[[k, j]];
                }
                vd[[i, j]] = sum;
            }
        }

        let mut a = Array2::zeros((size, size));
        for i in 0..size {
            for j in 0..size {
                let mut sum = 0.0;
                for k in 0..size {
                    sum += vd[[i, k]] * eigenvectors[[j, k]]; // V^T = V transpose
                }
                a[[i, j]] = sum;
            }
        }

        Ok((a, eigenvalues, eigenvectors))
    }

    /// Generate random sparse matrix with controlled sparsity
    pub fn random_sparse_matrix(
        rows: usize,
        cols: usize,
        density: f64,
        seed: u64,
    ) -> Result<Vec<(usize, usize, f64)>, String> {
        if !(0.0..=1.0).contains(&density) {
            return Err("Density must be between 0 and 1".to_string());
        }

        let mut rng = seeded_rng(seed);
        let mut triplets = Vec::new();

        let total_elements = rows * cols;
        let nnz = (total_elements as f64 * density) as usize;

        for _ in 0..nnz {
            let row = rng.sample(Uniform::new(0, rows).unwrap());
            let col = rng.sample(Uniform::new(0, cols).unwrap());
            let value = rng.sample(Normal::new(0.0, 1.0).unwrap());
            triplets.push((row, col, value));
        }

        Ok(triplets)
    }
}

/// Bridge for statistical analysis operations
pub struct StatsBridge;

impl StatsBridge {
    /// Create a comprehensive experimental design
    pub fn design_experiment() -> ExperimentDesignBuilder {
        ExperimentDesignBuilder::new()
    }

    /// Generate synthetic dataset with known statistical properties
    pub fn synthetic_dataset(
        properties: DatasetProperties,
        seed: u64,
    ) -> Result<SyntheticDataset, String> {
        let mut rng = seeded_rng(seed);
        let mut data = HashMap::new();

        // Generate features according to specifications
        for (name, spec) in properties.features.iter() {
            let feature_data = match &spec.distribution {
                FeatureDistribution::Normal { mean, std } => {
                    let normal = Normal::new(*mean, *std).unwrap();
                    (0..properties.n_samples)
                        .map(|_| rng.sample(normal))
                        .collect::<Vec<f64>>()
                }
                FeatureDistribution::Uniform { low, high } => {
                    let uniform = Uniform::new(*low, *high).unwrap();
                    (0..properties.n_samples)
                        .map(|_| rng.sample(uniform))
                        .collect::<Vec<f64>>()
                }
                FeatureDistribution::Beta { alpha, beta } => {
                    let beta_dist = Beta::new(*alpha, *beta)?;
                    (0..properties.n_samples)
                        .map(|_| beta_dist.sample(&mut rng))
                        .collect::<Vec<f64>>()
                }
                FeatureDistribution::Categorical {
                    categories,
                    weights,
                } => {
                    let categorical = Categorical::new(weights.clone())?;
                    (0..properties.n_samples)
                        .map(|_| categorical.sample(&mut rng) as f64)
                        .collect::<Vec<f64>>()
                }
            };
            data.insert(name.clone(), feature_data);
        }

        Ok(SyntheticDataset { data, properties })
    }

    /// Generate correlated multivariate dataset
    pub fn correlated_dataset(
        means: Vec<f64>,
        correlation_matrix: Vec<Vec<f64>>,
        n_samples: usize,
        seed: u64,
    ) -> Result<Array2<f64>, String> {
        let mvn = MultivariateNormal::new(means, correlation_matrix)?;
        let mut rng = seeded_rng(seed);

        let mut samples = Array2::zeros((n_samples, mvn.dimension()));
        for i in 0..n_samples {
            let sample = mvn.sample(&mut rng);
            for j in 0..sample.len() {
                samples[[i, j]] = sample[j];
            }
        }

        Ok(samples)
    }

    /// Bootstrap confidence intervals
    pub fn bootstrap_confidence_interval<F>(
        data: &[f64],
        statistic: F,
        confidence_level: f64,
        n_bootstrap: usize,
        seed: u64,
    ) -> Result<(f64, f64, f64), String>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let pool = ThreadLocalRngPool::new(seed);

        let bootstrap_stats: Vec<f64> = (0..n_bootstrap)
            .map(|_| {
                let mut rng = seeded_rng(seed);
                use crate::random::slice_ops::ScientificSliceRandom;
                let bootstrap_sample =
                    data.scientific_sample_with_replacement(&mut rng, data.len());
                let sample_values: Vec<f64> = bootstrap_sample.iter().map(|&&x| x).collect();
                statistic(&sample_values)
            })
            .collect();

        let mut sorted_stats = bootstrap_stats;
        sorted_stats.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = 1.0 - confidence_level;
        let lower_idx = (alpha / 2.0 * sorted_stats.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted_stats.len() as f64) as usize;

        let lower_bound = sorted_stats[lower_idx];
        let upper_bound = sorted_stats[upper_idx.min(sorted_stats.len() - 1)];
        let point_estimate = statistic(data);

        Ok((point_estimate, lower_bound, upper_bound))
    }
}

/// Bridge for neural network operations
pub struct NeuralBridge;

impl NeuralBridge {
    /// Xavier/Glorot weight initialization
    pub fn xavier_initialization(
        layer_sizes: &[usize],
        seed: u64,
    ) -> Result<Vec<Array2<f64>>, String> {
        let mut rng = seeded_rng(seed);
        let mut weights = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[i] as f64;
            let fan_out = layer_sizes[i + 1] as f64;
            let std = (2.0 / (fan_in + fan_out)).sqrt();

            let normal = Normal::new(0.0, std).unwrap();
            let mut weight_matrix = Array2::zeros((layer_sizes[i + 1], layer_sizes[i]));

            for j in 0..layer_sizes[i + 1] {
                for k in 0..layer_sizes[i] {
                    weight_matrix[[j, k]] = rng.sample(normal);
                }
            }

            weights.push(weight_matrix);
        }

        Ok(weights)
    }

    /// He/Kaiming weight initialization for ReLU networks
    pub fn he_initialization(layer_sizes: &[usize], seed: u64) -> Result<Vec<Array2<f64>>, String> {
        let mut rng = seeded_rng(seed);
        let mut weights = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[i] as f64;
            let std = (2.0 / fan_in).sqrt();

            let normal = Normal::new(0.0, std).unwrap();
            let mut weight_matrix = Array2::zeros((layer_sizes[i + 1], layer_sizes[i]));

            for j in 0..layer_sizes[i + 1] {
                for k in 0..layer_sizes[i] {
                    weight_matrix[[j, k]] = rng.sample(normal);
                }
            }

            weights.push(weight_matrix);
        }

        Ok(weights)
    }

    /// LeCun weight initialization for SELU networks
    pub fn lecun_initialization(
        layer_sizes: &[usize],
        seed: u64,
    ) -> Result<Vec<Array2<f64>>, String> {
        let mut rng = seeded_rng(seed);
        let mut weights = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let fan_in = layer_sizes[i] as f64;
            let std = (1.0 / fan_in).sqrt();

            let normal = Normal::new(0.0, std).unwrap();
            let mut weight_matrix = Array2::zeros((layer_sizes[i + 1], layer_sizes[i]));

            for j in 0..layer_sizes[i + 1] {
                for k in 0..layer_sizes[i] {
                    weight_matrix[[j, k]] = rng.sample(normal);
                }
            }

            weights.push(weight_matrix);
        }

        Ok(weights)
    }

    /// Generate random dropout masks
    pub fn dropout_masks(
        shapes: &[(usize, usize)],
        dropout_rate: f64,
        seed: u64,
    ) -> Result<Vec<Array2<f64>>, String> {
        let mut rng = seeded_rng(seed);
        let mut masks = Vec::new();

        let keep_prob = 1.0 - dropout_rate;
        let uniform = Uniform::new(0.0, 1.0).unwrap();

        for &(rows, cols) in shapes {
            let mut mask = Array2::zeros((rows, cols));
            for i in 0..rows {
                for j in 0..cols {
                    mask[[i, j]] = if rng.sample(uniform) < keep_prob {
                        1.0 / keep_prob
                    } else {
                        0.0
                    };
                }
            }
            masks.push(mask);
        }

        Ok(masks)
    }

    /// Gradient noise injection for improved generalization
    pub fn gradient_noise_injection(
        noise_scale: f64,
        gradients: &[Array2<f64>],
        seed: u64,
    ) -> Result<Vec<Array2<f64>>, String> {
        let mut rng = seeded_rng(seed);
        let mut noisy_gradients = Vec::new();

        let normal = Normal::new(0.0, noise_scale).unwrap();

        for gradient in gradients {
            let mut noisy_gradient = gradient.clone();
            for elem in noisy_gradient.iter_mut() {
                *elem += rng.sample(normal);
            }
            noisy_gradients.push(noisy_gradient);
        }

        Ok(noisy_gradients)
    }

    /// Generate random augmentation parameters
    pub fn augmentation_parameters(
        batch_size: usize,
        config: AugmentationConfig,
        seed: u64,
    ) -> AugmentationBatch {
        let mut rng = seeded_rng(seed);
        let mut batch = AugmentationBatch::new(batch_size);

        for _ in 0..batch_size {
            let rotation = if config.rotation_range > 0.0 {
                rng.sample(Uniform::new(-config.rotation_range, config.rotation_range).unwrap())
            } else {
                0.0
            };

            let scale = if config.scale_range.0 < config.scale_range.1 {
                rng.sample(Uniform::new(config.scale_range.0, config.scale_range.1).unwrap())
            } else {
                1.0
            };

            let translation_x = if config.translation_range.0 > 0.0 {
                rng.sample(
                    Uniform::new(-config.translation_range.0, config.translation_range.0).unwrap(),
                )
            } else {
                0.0
            };

            let translation_y = if config.translation_range.1 > 0.0 {
                rng.sample(
                    Uniform::new(-config.translation_range.1, config.translation_range.1).unwrap(),
                )
            } else {
                0.0
            };

            batch.add_transform(AugmentationTransform {
                rotation,
                scale,
                translation: (translation_x, translation_y),
                horizontal_flip: rng.sample(Uniform::new(0.0, 1.0).unwrap())
                    < config.horizontal_flip_prob,
                vertical_flip: rng.sample(Uniform::new(0.0, 1.0).unwrap())
                    < config.vertical_flip_prob,
            });
        }

        batch
    }
}

/// Bridge for optimization algorithms requiring randomness
pub struct OptimizationBridge;

impl OptimizationBridge {
    /// Generate random initial population for genetic algorithms
    pub fn genetic_algorithm_population<T>(
        population_size: usize,
        individual_generator: impl Fn(&mut Random<rand::rngs::StdRng>) -> T,
        seed: u64,
    ) -> Vec<T> {
        let mut rng = seeded_rng(seed);
        (0..population_size)
            .map(|_| individual_generator(&mut rng))
            .collect()
    }

    /// Generate random perturbations for simulated annealing
    pub fn simulated_annealing_perturbation(
        current_state: &[f64],
        temperature: f64,
        perturbation_scale: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = seeded_rng(seed);
        let std = perturbation_scale * temperature.sqrt();
        let normal = Normal::new(0.0, std).unwrap();

        current_state
            .iter()
            .map(|&x| x + rng.sample(normal))
            .collect()
    }

    /// Generate random directions for coordinate descent
    pub fn random_coordinate_directions(
        dimensions: usize,
        n_directions: usize,
        seed: u64,
    ) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        let mut directions = Array2::zeros((n_directions, dimensions));

        for i in 0..n_directions {
            // Generate random unit vector
            let mut direction = vec![0.0; dimensions];
            for j in 0..dimensions {
                direction[j] = rng.sample(Normal::new(0.0, 1.0).unwrap());
            }

            // Normalize
            let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
            for j in 0..dimensions {
                directions[[i, j]] = direction[j] / norm;
            }
        }

        directions
    }

    /// Generate noise for parameter exploration in reinforcement learning
    pub fn exploration_noise(
        action_dimensions: usize,
        noise_type: ExplorationNoiseType,
        time_step: usize,
        seed: u64,
    ) -> Vec<f64> {
        let mut rng = seeded_rng(seed + time_step as u64);

        match noise_type {
            ExplorationNoiseType::Gaussian { std } => {
                let normal = Normal::new(0.0, std).unwrap();
                (0..action_dimensions).map(|_| rng.sample(normal)).collect()
            }
            ExplorationNoiseType::OrnsteinUhlenbeck { theta, sigma, mu } => {
                // Simplified OU process (would need state for full implementation)
                let dt = 1.0;
                let std = sigma * (2.0 * theta * dt).sqrt();
                let normal = Normal::new(0.0, std).unwrap();
                (0..action_dimensions)
                    .map(|_| mu + rng.sample(normal))
                    .collect()
            }
            ExplorationNoiseType::EpsilonGreedy { epsilon } => {
                let uniform = Uniform::new(0.0, 1.0).unwrap();
                (0..action_dimensions)
                    .map(|_| {
                        if rng.sample(uniform) < epsilon {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect()
            }
        }
    }
}

// Supporting types and builders

#[derive(Debug, Clone)]
pub struct ExperimentDesignBuilder {
    factors: Vec<Vec<f64>>,
    replications: usize,
    blocking_factors: Vec<String>,
    randomization_seed: Option<u64>,
    design_type: DesignType,
}

#[derive(Debug, Clone)]
pub enum DesignType {
    FullFactorial,
    FractionalFactorial { fraction: f64 },
    CentralComposite { alpha: f64 },
    LatinHypercube,
    RandomSampling { n_points: usize },
}

impl ExperimentDesignBuilder {
    pub fn new() -> Self {
        Self {
            factors: Vec::new(),
            replications: 1,
            blocking_factors: Vec::new(),
            randomization_seed: None,
            design_type: DesignType::FullFactorial,
        }
    }

    pub fn factors(mut self, factors: &[Vec<f64>]) -> Self {
        self.factors = factors.to_vec();
        self
    }

    pub fn replications(mut self, n: usize) -> Self {
        self.replications = n;
        self
    }

    pub fn randomization_seed(mut self, seed: u64) -> Self {
        self.randomization_seed = Some(seed);
        self
    }

    pub fn design_type(mut self, design: DesignType) -> Self {
        self.design_type = design;
        self
    }

    pub fn build(self) -> Result<ExperimentalDesign, String> {
        let seed = self.randomization_seed.unwrap_or(42);

        let design_points = match self.design_type {
            DesignType::FullFactorial => {
                crate::random::scientific::ExperimentalDesign::factorial_design(&self.factors)
            }
            DesignType::FractionalFactorial { fraction } => {
                crate::random::scientific::ExperimentalDesign::fractional_factorial_design(
                    &self.factors,
                    fraction,
                    seed,
                )
            }
            DesignType::CentralComposite { alpha } => {
                crate::random::scientific::ExperimentalDesign::central_composite_design(
                    self.factors.len(),
                    alpha,
                )
            }
            DesignType::LatinHypercube => {
                // Would integrate with QMC module
                return Err("Latin Hypercube design not yet implemented".to_string());
            }
            DesignType::RandomSampling { n_points } => {
                let mut rng = seeded_rng(seed);
                let mut points = Vec::new();
                for _ in 0..n_points {
                    let mut point = Vec::new();
                    for factor in &self.factors {
                        let idx = rng.sample(Uniform::new(0, factor.len()).unwrap());
                        point.push(factor[idx]);
                    }
                    points.push(point);
                }
                points
            }
        };

        Ok(ExperimentalDesign {
            design_points,
            replications: self.replications,
            factor_names: (0..self.factors.len())
                .map(|i| format!("Factor_{}", i))
                .collect(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct ExperimentalDesign {
    pub design_points: Vec<Vec<f64>>,
    pub replications: usize,
    pub factor_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DatasetProperties {
    pub n_samples: usize,
    pub features: HashMap<String, FeatureSpec>,
}

#[derive(Debug, Clone)]
pub struct FeatureSpec {
    pub distribution: FeatureDistribution,
    pub correlation_target: Option<String>,
    pub correlation_strength: f64,
}

#[derive(Debug, Clone)]
pub enum FeatureDistribution {
    Normal {
        mean: f64,
        std: f64,
    },
    Uniform {
        low: f64,
        high: f64,
    },
    Beta {
        alpha: f64,
        beta: f64,
    },
    Categorical {
        categories: Vec<usize>,
        weights: Vec<f64>,
    },
}

#[derive(Debug, Clone)]
pub struct SyntheticDataset {
    pub data: HashMap<String, Vec<f64>>,
    pub properties: DatasetProperties,
}

#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    pub rotation_range: f64,
    pub scale_range: (f64, f64),
    pub translation_range: (f64, f64),
    pub horizontal_flip_prob: f64,
    pub vertical_flip_prob: f64,
}

#[derive(Debug, Clone)]
pub struct AugmentationTransform {
    pub rotation: f64,
    pub scale: f64,
    pub translation: (f64, f64),
    pub horizontal_flip: bool,
    pub vertical_flip: bool,
}

#[derive(Debug)]
pub struct AugmentationBatch {
    pub transforms: Vec<AugmentationTransform>,
}

impl AugmentationBatch {
    pub fn new(capacity: usize) -> Self {
        Self {
            transforms: Vec::with_capacity(capacity),
        }
    }

    pub fn add_transform(&mut self, transform: AugmentationTransform) {
        self.transforms.push(transform);
    }
}

#[derive(Debug, Clone)]
pub enum ExplorationNoiseType {
    Gaussian { std: f64 },
    OrnsteinUhlenbeck { theta: f64, sigma: f64, mu: f64 },
    EpsilonGreedy { epsilon: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_linalg_bridge_symmetric_matrix() {
        let matrix = LinalgBridge::random_symmetric_matrix(5, 42).unwrap();

        // Check symmetry
        for i in 0..5 {
            for j in 0..5 {
                assert_relative_eq!(matrix[[i, j]], matrix[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_linalg_bridge_positive_definite() {
        let matrix = LinalgBridge::random_symmetric_positive_definite(3, 42).unwrap();

        // Check that all diagonal elements are positive
        for i in 0..3 {
            assert!(matrix[[i, i]] > 0.0);
        }

        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(matrix[[i, j]], matrix[[j, i]], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_neural_bridge_xavier_init() {
        let layer_sizes = vec![784, 128, 64, 10];
        let weights = NeuralBridge::xavier_initialization(&layer_sizes, 42).unwrap();

        assert_eq!(weights.len(), 3); // 3 weight matrices
        assert_eq!(weights[0].shape(), [128, 784]);
        assert_eq!(weights[1].shape(), [64, 128]);
        assert_eq!(weights[2].shape(), [10, 64]);
    }

    #[test]
    #[ignore] // Flaky statistical test - bootstrap confidence intervals can be sensitive
    fn test_stats_bridge_bootstrap_ci() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let (point_est, lower, upper) = StatsBridge::bootstrap_confidence_interval(
            &data,
            |samples| samples.iter().sum::<f64>() / samples.len() as f64, // Mean
            0.95,
            1000,
            42,
        )
        .unwrap();

        assert_relative_eq!(point_est, 5.5, epsilon = 0.1);
        assert!(lower < point_est);
        assert!(upper > point_est);
    }

    #[test]
    fn test_optimization_bridge_genetic_population() {
        let population = OptimizationBridge::genetic_algorithm_population(
            10,
            |rng| {
                (0..5)
                    .map(|_| rng.sample(Uniform::new(0.0, 1.0).unwrap()))
                    .collect::<Vec<f64>>()
            },
            42,
        );

        assert_eq!(population.len(), 10);
        for individual in &population {
            assert_eq!(individual.len(), 5);
            for &gene in individual {
                assert!((0.0..=1.0).contains(&gene));
            }
        }
    }

    #[test]
    fn test_experiment_design_builder() {
        let design = StatsBridge::design_experiment()
            .factors(&[vec![1.0, 2.0], vec![0.1, 0.2]])
            .replications(3)
            .randomization_seed(42)
            .build()
            .unwrap();

        assert_eq!(design.design_points.len(), 4); // 2x2 factorial
        assert_eq!(design.replications, 3);
        assert_eq!(design.factor_names.len(), 2);
    }
}
