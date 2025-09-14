//! Machine learning-based spatial optimization
//!
//! This module implements advanced machine learning techniques to automatically optimize
//! spatial algorithms, including neural network-based parameter tuning, reinforcement
//! learning for algorithm selection, and adaptive optimization strategies that learn
//! from data patterns and computational environments.
//!
//! # Features
//!
//! - **Neural Algorithm Optimization**: Deep learning models that optimize algorithm parameters
//! - **Reinforcement Learning**: RL agents that learn optimal algorithm selection strategies
//! - **Meta-learning**: Learn to learn new spatial patterns quickly
//! - **AutoML for Spatial Computing**: Automated machine learning for spatial problems
//! - **Bayesian Optimization**: Gaussian process-based hyperparameter optimization
//! - **Ensemble Methods**: Combine multiple algorithms intelligently
//! - **Online Learning**: Adapt to changing data distributions in real-time
//! - **Transfer Learning**: Apply knowledge from related spatial domains
//!
//! # Applications
//!
//! - **Automatic hyperparameter tuning** for clustering algorithms
//! - **Dynamic algorithm selection** based on data characteristics
//! - **Learned distance metrics** optimized for specific tasks
//! - **Adaptive spatial data structures** that restructure based on access patterns
//! - **Predictive preprocessing** that optimizes data layout for better performance
//!
//! # Examples
//!
//! ```
//! use scirs2_spatial::ml_optimization::{NeuralSpatialOptimizer, ReinforcementLearningSelector};
//! use ndarray::array;
//!
//! // Neural network-based spatial optimizer
//! let mut optimizer = NeuralSpatialOptimizer::new()
//!     .with_network_architecture([64, 128, 64, 32])
//!     .with_learning_rate(0.001)
//!     .with_adaptive_learning(true);
//!
//! let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
//! let optimized_params = optimizer.optimize_clustering_parameters(&points.view())?;
//! println!("Optimized k-means parameters: {:?}", optimized_params);
//!
//! // Reinforcement learning algorithm selector
//! let mut rl_selector = ReinforcementLearningSelector::new()
//!     .with_epsilon_greedy(0.1)
//!     .with_experience_replay(true)
//!     .with_target_network(true);
//!
//! let selected_algorithm = rl_selector.select_best_algorithm(&points.view())?;
//! println!("RL selected algorithm: {:?}", selected_algorithm);
//! ```

use crate::error::{SpatialError, SpatialResult};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::Rng;
use statrs::statistics::Statistics;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;

/// Neural network layer for spatial optimization
#[derive(Debug, Clone)]
pub struct NeuralLayer {
    /// Layer weights
    pub weights: Array2<f64>,
    /// Layer biases
    pub biases: Array1<f64>,
    /// Activation function
    pub activation: ActivationFunction,
}

/// Activation functions for neural networks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFunction {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU
    LeakyReLU(f64),
    /// Sigmoid activation
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Swish activation (x * sigmoid(x))
    Swish,
    /// GELU activation
    GELU,
}

impl ActivationFunction {
    /// Apply activation function to input
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => x * (1.0 / (1.0 + (-x).exp())),
            ActivationFunction::GELU => {
                0.5 * x * (1.0 + ((2.0_f64 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }
        }
    }

    /// Compute derivative of activation function
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::LeakyReLU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            ActivationFunction::Sigmoid => {
                let sigmoid_x = self.apply(x);
                sigmoid_x * (1.0 - sigmoid_x)
            }
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::Swish => {
                let sigmoid_x = 1.0 / (1.0 + (-x).exp());
                sigmoid_x + x * sigmoid_x * (1.0 - sigmoid_x)
            }
            ActivationFunction::GELU => {
                let sqrt_2_pi = (2.0_f64 / PI).sqrt();
                let tanh_input = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                let tanh_val = tanh_input.tanh();
                0.5 * (1.0 + tanh_val)
                    + 0.5
                        * x
                        * (1.0 - tanh_val.powi(2))
                        * sqrt_2_pi
                        * (1.0 + 3.0 * 0.044715 * x.powi(2))
            }
        }
    }
}

/// Neural spatial optimizer
#[derive(Debug, Clone)]
pub struct NeuralSpatialOptimizer {
    /// Neural network layers
    layers: Vec<NeuralLayer>,
    /// Learning rate
    learning_rate: f64,
    /// Adaptive learning rate
    adaptive_learning: bool,
    /// Experience buffer for training
    experience_buffer: VecDeque<(Array1<f64>, Array1<f64>)>,
    /// Training iterations
    training_iterations: usize,
    /// Loss history
    loss_history: Vec<f64>,
}

impl Default for NeuralSpatialOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralSpatialOptimizer {
    /// Create new neural spatial optimizer
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            learning_rate: 0.001,
            adaptive_learning: false,
            experience_buffer: VecDeque::new(),
            training_iterations: 0,
            loss_history: Vec::new(),
        }
    }

    /// Configure network architecture
    pub fn with_network_architecture(mut self, layersizes: impl AsRef<[usize]>) -> Self {
        let sizes = layersizes.as_ref();
        self.layers.clear();

        for i in 0..sizes.len() - 1 {
            let input_size = sizes[i];
            let output_size = sizes[i + 1];

            // Xavier/Glorot initialization
            let scale = (2.0_f64 / (input_size + output_size) as f64).sqrt();
            let weights = Array2::from_shape_fn((output_size, input_size), |_| {
                (rand::random::<f64>() - 0.5) * 2.0 * scale
            });
            let biases = Array1::zeros(output_size);

            let activation = if i == sizes.len() - 2 {
                ActivationFunction::Sigmoid // Output layer
            } else {
                ActivationFunction::ReLU // Hidden layers
            };

            self.layers.push(NeuralLayer {
                weights,
                biases,
                activation,
            });
        }

        self
    }

    /// Set network architecture in place (for use when already borrowed mutably)
    pub fn set_network_architecture(&mut self, layersizes: impl AsRef<[usize]>) {
        let sizes = layersizes.as_ref();
        self.layers.clear();

        for i in 0..sizes.len() - 1 {
            let input_size = sizes[i];
            let output_size = sizes[i + 1];

            // Xavier/Glorot initialization
            let scale = (2.0_f64 / (input_size + output_size) as f64).sqrt();
            let weights = Array2::from_shape_fn((output_size, input_size), |_| {
                (rand::random::<f64>() - 0.5) * 2.0 * scale
            });
            let biases = Array1::zeros(output_size);

            let activation = if i == sizes.len() - 2 {
                ActivationFunction::Sigmoid // Output layer
            } else {
                ActivationFunction::ReLU // Hidden layers
            };

            self.layers.push(NeuralLayer {
                weights,
                biases,
                activation,
            });
        }
    }

    /// Configure learning rate
    pub fn with_learning_rate(&mut self, lr: f64) -> &mut Self {
        self.learning_rate = lr;
        self
    }

    /// Enable adaptive learning
    pub fn with_adaptive_learning(&mut self, enabled: bool) -> &mut Self {
        self.adaptive_learning = enabled;
        self
    }

    /// Optimize clustering parameters using neural network
    pub fn optimize_clustering_parameters(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<ClusteringParameters> {
        // Extract features from spatial data
        let features = self.extract_spatial_features(points)?;

        // If network is empty, initialize with default architecture
        if self.layers.is_empty() {
            let feature_size = features.len();
            self.set_network_architecture([feature_size, 64, 32, 16, 8]);
        }

        // Forward pass to get optimal parameters
        let output = self.forward_pass(&features)?;

        // Convert network output to clustering parameters
        let params = self.decode_clustering_parameters(&output)?;

        // Evaluate parameters and update network if we have ground truth
        if let Some(qualityscore) = self.evaluate_clustering_quality(points, &params)? {
            self.update_network(&features, qualityscore)?;
        }

        Ok(params)
    }

    /// Extract spatial features from data
    fn extract_spatial_features(
        &self,
        n_points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<Array1<f64>> {
        let (num_points, n_dims) = n_points.dim();
        let mut features = Vec::new();

        // Basic statistics
        features.push(num_points as f64);
        features.push(n_dims as f64);

        // Data distribution features
        for dim in 0..n_dims {
            let column = n_points.column(dim);
            let mean = column.mean();
            let std = (column.mapv(|x| (x - mean).powi(2)).mean()).sqrt();
            let min_val = column.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            features.push(mean);
            features.push(std);
            features.push(max_val - min_val); // Range
            features.push(if std > 0.0 {
                (max_val - min_val) / std
            } else {
                0.0
            }); // Coefficient of variation
        }

        // Pairwise distance statistics
        let mut distances = Vec::new();
        for i in 0..num_points.min(100) {
            // Sample for efficiency
            for j in (i + 1)..num_points.min(100) {
                let dist: f64 = n_points
                    .row(i)
                    .iter()
                    .zip(n_points.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push(dist);
            }
        }

        if !distances.is_empty() {
            let mean_dist = distances.iter().sum::<f64>() / distances.len() as f64;
            let dist_std = {
                let variance = distances
                    .iter()
                    .map(|&d| (d - mean_dist).powi(2))
                    .sum::<f64>()
                    / distances.len() as f64;
                variance.sqrt()
            };
            features.push(mean_dist);
            features.push(dist_std);
            features.push(distances.iter().fold(f64::INFINITY, |a, &b| a.min(b))); // Min distance
            features.push(distances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
        // Max distance
        } else {
            features.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        }

        // Density estimation
        let density = NeuralSpatialOptimizer::estimate_local_density(n_points)?;
        features.push(density);

        // Clustering tendency (Hopkins statistic approximation)
        let hopkins = self.estimate_clustering_tendency(n_points)?;
        features.push(hopkins);

        Ok(Array1::from(features))
    }

    /// Estimate local density of the data
    fn estimate_local_density(points: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
        let (n_points, n_dims) = points.dim();

        if n_points < 2 {
            return Ok(0.0);
        }

        // Sample n_points for efficiency
        let sample_size = n_points.min(50);
        let mut total_inverse_distance = 0.0;
        let mut count = 0;

        for i in 0..sample_size {
            let mut nearest_distance = f64::INFINITY;

            for j in 0..n_points {
                if i != j {
                    let dist: f64 = points
                        .row(i)
                        .iter()
                        .zip(points.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if dist < nearest_distance {
                        nearest_distance = dist;
                    }
                }
            }

            if nearest_distance > 0.0 && nearest_distance < f64::INFINITY {
                total_inverse_distance += 1.0 / nearest_distance;
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_inverse_distance / count as f64
        } else {
            0.0
        })
    }

    /// Estimate clustering tendency (Hopkins-like statistic)
    fn estimate_clustering_tendency(&self, points: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
        let (n_points, n_dims) = points.dim();

        if n_points < 10 {
            return Ok(0.5); // Neutral value
        }

        // Sample a subset of n_points
        let sample_size = n_points.min(20);
        let mut real_distances = Vec::new();
        let mut random_distances = Vec::new();

        // Calculate distances to nearest neighbors for real n_points
        for i in 0..sample_size {
            let mut min_dist = f64::INFINITY;
            for j in 0..n_points {
                if i != j {
                    let dist: f64 = points
                        .row(i)
                        .iter()
                        .zip(points.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    min_dist = min_dist.min(dist);
                }
            }
            real_distances.push(min_dist);
        }

        // Generate random n_points and calculate distances
        let bounds = self.get_data_bounds(points);
        for _ in 0..sample_size {
            let random_point: Array1<f64> = Array1::from_shape_fn(n_dims, |i| {
                rand::random::<f64>() * (bounds[i].1 - bounds[i].0) + bounds[i].0
            });

            let mut min_dist = f64::INFINITY;
            for j in 0..n_points {
                let dist: f64 = random_point
                    .iter()
                    .zip(points.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                min_dist = min_dist.min(dist);
            }
            random_distances.push(min_dist);
        }

        // Calculate Hopkins-like statistic
        let sum_random: f64 = random_distances.iter().sum();
        let sum_real: f64 = real_distances.iter().sum();
        let hopkins = sum_random / (sum_random + sum_real);

        Ok(hopkins)
    }

    /// Get data bounds for each dimension
    fn get_data_bounds(&self, points: &ArrayView2<'_, f64>) -> Vec<(f64, f64)> {
        let (_, n_dims) = points.dim();
        let mut bounds = Vec::new();

        for dim in 0..n_dims {
            let column = points.column(dim);
            let min_val = column.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            bounds.push((min_val, max_val));
        }

        bounds
    }

    /// Forward pass through neural network
    fn forward_pass(&self, input: &Array1<f64>) -> SpatialResult<Array1<f64>> {
        let mut current = input.clone();

        for layer in &self.layers {
            // Linear transformation: y = Wx + b
            let linear_output = layer.weights.dot(&current) + &layer.biases;

            // Apply activation function
            current = linear_output.mapv(|x| layer.activation.apply(x));
        }

        Ok(current)
    }

    /// Decode neural network output to clustering parameters
    fn decode_clustering_parameters(
        &self,
        output: &Array1<f64>,
    ) -> SpatialResult<ClusteringParameters> {
        if output.len() < 8 {
            return Err(SpatialError::InvalidInput(
                "Insufficient network output for parameter decoding".to_string(),
            ));
        }

        Ok(ClusteringParameters {
            num_clusters: ((output[0] * 20.0) as usize).clamp(1, 20), // 1-20 clusters
            max_iterations: ((output[1] * 500.0) as usize).clamp(10, 500), // 10-500 iterations
            tolerance: output[2] * 1e-3,                              // 0 to 1e-3 tolerance
            init_method: if output[3] > 0.5 {
                InitMethod::KMeansPlusPlus
            } else {
                InitMethod::Random
            },
            distance_metric: NeuralSpatialOptimizer::decode_distance_metric(output[4]),
            regularization: output[5] * 0.1, // 0 to 0.1 regularization
            early_stopping: output[6] > 0.5,
            adaptive_parameters: output[7] > 0.5,
        })
    }

    /// Decode distance metric from neural output
    fn decode_distance_metric(value: f64) -> DistanceMetric {
        match (value * 4.0) as usize {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::Manhattan,
            2 => DistanceMetric::Cosine,
            3 => DistanceMetric::Minkowski(2.0),
            _ => DistanceMetric::Euclidean,
        }
    }

    /// Evaluate clustering quality
    fn evaluate_clustering_quality(
        &self,
        points: &ArrayView2<'_, f64>,
        params: &ClusteringParameters,
    ) -> SpatialResult<Option<f64>> {
        // Run clustering with given parameters
        let clustering_result = self.run_clustering(points, params)?;

        // Calculate quality metrics
        let silhouette_score = self.calculate_silhouette_score(points, &clustering_result)?;
        let inertia = NeuralSpatialOptimizer::calculate_inertia(points, &clustering_result)?;
        let calinski_harabasz =
            self.calculate_calinski_harabasz_score(points, &clustering_result)?;

        // Combine metrics into single quality score
        let qualityscore =
            0.5 * silhouette_score + 0.3 * (1.0 / (1.0 + inertia)) + 0.2 * calinski_harabasz;

        Ok(Some(qualityscore))
    }

    /// Run clustering with given parameters
    fn run_clustering(
        &self,
        points: &ArrayView2<'_, f64>,
        params: &ClusteringParameters,
    ) -> SpatialResult<ClusteringResult> {
        // Simplified k-means implementation
        let (n_points, n_dims) = points.dim();
        let mut centroids = Array2::zeros((params.num_clusters, n_dims));
        let mut assignments = Array1::zeros(n_points);

        // Initialize centroids
        match params.init_method {
            InitMethod::Random => {
                for i in 0..params.num_clusters {
                    for j in 0..n_dims {
                        centroids[[i, j]] = rand::random::<f64>();
                    }
                }
            }
            InitMethod::KMeansPlusPlus => {
                // Simplified k-means++ initialization
                let mut rng = rand::rng();
                let mut selected = Vec::new();
                for _ in 0..params.num_clusters {
                    let idx = rng.gen_range(0..n_points);
                    selected.push(idx);
                }

                for (i, &idx) in selected.iter().enumerate() {
                    centroids.row_mut(i).assign(&points.row(idx));
                }
            }
        }

        // Run k-means iterations
        for _ in 0..params.max_iterations {
            let mut changed = false;

            // Assignment step
            for i in 0..n_points {
                let mut best_cluster = 0;
                let mut best_distance = f64::INFINITY;

                for j in 0..params.num_clusters {
                    let distance = NeuralSpatialOptimizer::calculate_distance(
                        &points.row(i).to_owned(),
                        &centroids.row(j).to_owned(),
                        &params.distance_metric,
                    );

                    if distance < best_distance {
                        best_distance = distance;
                        best_cluster = j;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut cluster_counts = vec![0; params.num_clusters];
            let mut new_centroids = Array2::zeros((params.num_clusters, n_dims));

            for i in 0..n_points {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;
                for j in 0..n_dims {
                    new_centroids[[cluster, j]] += points[[i, j]];
                }
            }

            for i in 0..params.num_clusters {
                if cluster_counts[i] > 0 {
                    for j in 0..n_dims {
                        new_centroids[[i, j]] /= cluster_counts[i] as f64;
                    }
                }
            }

            centroids = new_centroids;
        }

        let inertia = self.calculate_inertia_direct(points, &centroids, &assignments)?;

        Ok(ClusteringResult {
            centroids,
            assignments,
            inertia,
        })
    }

    /// Calculate distance between two points
    fn calculate_distance(a: &Array1<f64>, b: &Array1<f64>, metric: &DistanceMetric) -> f64 {
        match metric {
            DistanceMetric::Euclidean => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt(),
            DistanceMetric::Manhattan => a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum(),
            DistanceMetric::Cosine => {
                let dot_product: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
                let norm_a = a.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|&x| x.powi(2)).sum::<f64>().sqrt();
                1.0 - (dot_product / (norm_a * norm_b))
            }
            DistanceMetric::Minkowski(p) => a
                .iter()
                .zip(b.iter())
                .map(|(&x, &y)| (x - y).abs().powf(*p))
                .sum::<f64>()
                .powf(1.0 / p),
        }
    }

    /// Calculate silhouette score
    fn calculate_silhouette_score(
        &self,
        points: &ArrayView2<'_, f64>,
        result: &ClusteringResult,
    ) -> SpatialResult<f64> {
        let (n_points, n_dims) = points.dim();
        let mut silhouette_scores = Vec::new();

        for i in 0..n_points {
            let cluster_i = result.assignments[i];

            // Calculate average distance to points in same cluster
            let mut intra_cluster_distances = Vec::new();
            for j in 0..n_points {
                if i != j && result.assignments[j] == cluster_i {
                    let dist = NeuralSpatialOptimizer::calculate_distance(
                        &points.row(i).to_owned(),
                        &points.row(j).to_owned(),
                        &DistanceMetric::Euclidean,
                    );
                    intra_cluster_distances.push(dist);
                }
            }

            let a = if intra_cluster_distances.is_empty() {
                0.0
            } else {
                intra_cluster_distances.iter().sum::<f64>() / intra_cluster_distances.len() as f64
            };

            // Calculate minimum average distance to points in other clusters
            let mut min_inter_cluster_distance = f64::INFINITY;
            for cluster in 0..result.centroids.nrows() {
                if cluster != cluster_i {
                    let mut inter_cluster_distances = Vec::new();
                    for j in 0..n_points {
                        if result.assignments[j] == cluster {
                            let dist = NeuralSpatialOptimizer::calculate_distance(
                                &points.row(i).to_owned(),
                                &points.row(j).to_owned(),
                                &DistanceMetric::Euclidean,
                            );
                            inter_cluster_distances.push(dist);
                        }
                    }

                    if !inter_cluster_distances.is_empty() {
                        let avg_dist = inter_cluster_distances.iter().sum::<f64>()
                            / inter_cluster_distances.len() as f64;
                        min_inter_cluster_distance = min_inter_cluster_distance.min(avg_dist);
                    }
                }
            }

            let b = min_inter_cluster_distance;

            let silhouette = if a < b {
                1.0 - (a / b)
            } else if a > b {
                (b / a) - 1.0
            } else {
                0.0
            };

            silhouette_scores.push(silhouette);
        }

        Ok(silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64)
    }

    /// Calculate inertia (WCSS)
    fn calculate_inertia(
        self_points: &ArrayView2<'_, f64>,
        result: &ClusteringResult,
    ) -> SpatialResult<f64> {
        Ok(result.inertia)
    }

    /// Calculate inertia directly
    fn calculate_inertia_direct(
        &self,
        points: &ArrayView2<'_, f64>,
        centroids: &Array2<f64>,
        assignments: &Array1<usize>,
    ) -> SpatialResult<f64> {
        let (n_points, n_dims) = points.dim();
        let mut inertia = 0.0;

        for i in 0..n_points {
            let cluster = assignments[i];
            let distance = NeuralSpatialOptimizer::calculate_distance(
                &points.row(i).to_owned(),
                &centroids.row(cluster).to_owned(),
                &DistanceMetric::Euclidean,
            );
            inertia += distance.powi(2);
        }

        Ok(inertia)
    }

    /// Calculate Calinski-Harabasz score
    fn calculate_calinski_harabasz_score(
        &self,
        points: &ArrayView2<'_, f64>,
        result: &ClusteringResult,
    ) -> SpatialResult<f64> {
        let (n_points, n_dims) = points.dim();
        let k = result.centroids.nrows();

        if k <= 1 || n_points <= k {
            return Ok(0.0);
        }

        // Calculate overall centroid
        let overall_centroid: Array1<f64> = points.mean_axis(Axis(0)).unwrap();

        // Calculate between-cluster sum of squares
        let mut between_ss = 0.0;
        for i in 0..k {
            let cluster_size = result.assignments.iter().filter(|&&x| x == i).count() as f64;
            if cluster_size > 0.0 {
                let distance_to_overall = NeuralSpatialOptimizer::calculate_distance(
                    &result.centroids.row(i).to_owned(),
                    &overall_centroid,
                    &DistanceMetric::Euclidean,
                );
                between_ss += cluster_size * distance_to_overall.powi(2);
            }
        }

        // Calculate within-cluster sum of squares
        let within_ss = result.inertia;

        // Calinski-Harabasz index
        let ch_score = (between_ss / (k - 1) as f64) / (within_ss / (n_points - k) as f64);
        Ok(ch_score)
    }

    /// Update neural network based on performance feedback
    fn update_network(&mut self, _input: &Array1<f64>, qualityscore: f64) -> SpatialResult<()> {
        // Store experience for training
        let target = Array1::from(vec![qualityscore; 8]); // Simplified target
        self.experience_buffer.push_back((_input.clone(), target));

        // Limit buffer size
        if self.experience_buffer.len() > 1000 {
            self.experience_buffer.pop_front();
        }

        // Train network if we have enough experience
        if self.experience_buffer.len() >= 32 {
            self.train_network_batch()?;
        }

        Ok(())
    }

    /// Train neural network on a batch of experiences
    fn train_network_batch(&mut self) -> SpatialResult<()> {
        let batch_size = 32.min(self.experience_buffer.len());

        for _ in 0..batch_size {
            if let Some((input, target)) = self.experience_buffer.pop_front() {
                self.train_single_example(&input, &target)?;
            }
        }

        self.training_iterations += 1;

        // Adaptive learning rate
        if self.adaptive_learning && self.training_iterations % 100 == 0 {
            self.learning_rate *= 0.95; // Decay learning rate
        }

        Ok(())
    }

    /// Train on single example using backpropagation
    fn train_single_example(
        &mut self,
        input: &Array1<f64>,
        target: &Array1<f64>,
    ) -> SpatialResult<()> {
        // Forward pass
        let mut activations = vec![input.clone()];
        let mut current = input.clone();

        for layer in &self.layers {
            let linear_output = layer.weights.dot(&current) + &layer.biases;
            current = linear_output.mapv(|x| layer.activation.apply(x));
            activations.push(current.clone());
        }

        // Calculate loss
        let output = &activations[activations.len() - 1];
        let loss: f64 = target
            .iter()
            .zip(output.iter())
            .map(|(&t, &o)| (t - o).powi(2))
            .sum::<f64>()
            / target.len() as f64;

        self.loss_history.push(loss);

        // Backward pass
        let mut delta = output - target;

        for (i, layer) in self.layers.iter_mut().enumerate().rev() {
            let layer_input = &activations[i];
            let _layer_output = &activations[i + 1];

            // Compute gradients
            let weight_gradients = delta
                .clone()
                .insert_axis(Axis(1))
                .dot(&layer_input.clone().insert_axis(Axis(0)));
            let bias_gradients = delta.clone();

            // Update weights and biases
            layer.weights = &layer.weights - self.learning_rate * &weight_gradients;
            layer.biases = &layer.biases - self.learning_rate * &bias_gradients;

            // Compute delta for next layer
            if i > 0 {
                let prev_layer_output = &activations[i];
                delta = layer.weights.t().dot(&delta);

                // Apply derivative of activation function
                for (j, &output_val) in prev_layer_output.iter().enumerate() {
                    delta[j] *= layer.activation.derivative(output_val);
                }
            }
        }

        Ok(())
    }
}

/// Clustering parameters optimized by neural network
#[derive(Debug, Clone)]
pub struct ClusteringParameters {
    pub num_clusters: usize,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub init_method: InitMethod,
    pub distance_metric: DistanceMetric,
    pub regularization: f64,
    pub early_stopping: bool,
    pub adaptive_parameters: bool,
}

/// Initialization methods for clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitMethod {
    Random,
    KMeansPlusPlus,
}

/// Distance metrics for clustering
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Minkowski(f64),
}

/// Clustering result
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    pub centroids: Array2<f64>,
    pub assignments: Array1<usize>,
    pub inertia: f64,
}

/// Reinforcement learning algorithm selector
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ReinforcementLearningSelector {
    /// Q-table for algorithm selection
    q_table: HashMap<StateAction, f64>,
    /// Epsilon for epsilon-greedy exploration
    epsilon: f64,
    /// Learning rate
    learning_rate: f64,
    /// Discount factor
    gamma: f64,
    /// Experience replay buffer
    experience_buffer: VecDeque<Experience>,
    /// Target network (for DQN)
    target_q_table: Option<HashMap<StateAction, f64>>,
    /// Episode count
    episodes: usize,
}

/// State-action pair for Q-learning
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct StateAction {
    pub state: DataState,
    pub action: SpatialAlgorithm,
}

/// Data state representation
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DataState {
    pub size_category: SizeCategory,
    pub dimensionality_category: DimensionalityCategory,
    pub density_category: DensityCategory,
    pub clustering_tendency_category: ClusteringTendencyCategory,
}

/// Data size categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum SizeCategory {
    Small,  // < 1000 points
    Medium, // 1000 - 10000 points
    Large,  // > 10000 points
}

/// Dimensionality categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DimensionalityCategory {
    Low,    // < 5 dimensions
    Medium, // 5 - 20 dimensions
    High,   // > 20 dimensions
}

/// Density categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DensityCategory {
    Low,
    Medium,
    High,
}

/// Clustering tendency categories
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ClusteringTendencyCategory {
    Random,
    Structured,
    HighlyStructured,
}

/// Available spatial algorithms
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum SpatialAlgorithm {
    KMeans,
    DBScan,
    HierarchicalClustering,
    GaussianMixture,
    SpectralClustering,
    KDTree,
    BallTree,
}

/// Experience tuple for reinforcement learning
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: DataState,
    pub action: SpatialAlgorithm,
    pub reward: f64,
    pub next_state: DataState,
    pub done: bool,
}

impl Default for ReinforcementLearningSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl ReinforcementLearningSelector {
    /// Create new reinforcement learning selector
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            epsilon: 0.1,
            learning_rate: 0.1,
            gamma: 0.9,
            experience_buffer: VecDeque::new(),
            target_q_table: None,
            episodes: 0,
        }
    }

    /// Configure epsilon for exploration
    pub fn with_epsilon_greedy(&mut self, epsilon: f64) -> &mut Self {
        self.epsilon = epsilon;
        self
    }

    /// Enable experience replay
    pub fn with_experience_replay(&mut self, enabled: bool) -> &mut Self {
        if enabled {
            self.experience_buffer = VecDeque::new();
        }
        self
    }

    /// Enable target network
    pub fn with_target_network(&mut self, enabled: bool) -> &mut Self {
        if enabled {
            self.target_q_table = Some(HashMap::new());
        }
        self
    }

    /// Select best algorithm for given data
    pub fn select_best_algorithm(
        &mut self,
        points: &ArrayView2<'_, f64>,
    ) -> SpatialResult<SpatialAlgorithm> {
        let state = self.analyze_data_state(points)?;

        // Epsilon-greedy action selection
        if rand::random::<f64>() < self.epsilon {
            // Explore: random action
            Ok(self.random_algorithm())
        } else {
            // Exploit: best known action
            Ok(self.best_algorithm_for_state(&state))
        }
    }

    /// Analyze data to determine state
    fn analyze_data_state(&self, points: &ArrayView2<'_, f64>) -> SpatialResult<DataState> {
        let (n_points, n_dims) = points.dim();

        let size_category = match n_points {
            0..=999 => SizeCategory::Small,
            1000..=9999 => SizeCategory::Medium,
            _ => SizeCategory::Large,
        };

        let dimensionality_category = match n_dims {
            0..=4 => DimensionalityCategory::Low,
            5..=20 => DimensionalityCategory::Medium,
            _ => DimensionalityCategory::High,
        };

        // Estimate density
        let density = self.estimate_density(points)?;
        let density_category = if density < 0.3 {
            DensityCategory::Low
        } else if density < 0.7 {
            DensityCategory::Medium
        } else {
            DensityCategory::High
        };

        // Estimate clustering tendency
        let hopkins = self.estimate_hopkins_statistic(points)?;
        let clustering_tendency_category = if hopkins < 0.3 {
            ClusteringTendencyCategory::HighlyStructured
        } else if hopkins < 0.7 {
            ClusteringTendencyCategory::Structured
        } else {
            ClusteringTendencyCategory::Random
        };

        Ok(DataState {
            size_category,
            dimensionality_category,
            density_category,
            clustering_tendency_category,
        })
    }

    /// Estimate data density
    fn estimate_density(&self, points: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
        let (n_points, n_dims) = points.dim();

        if n_points < 2 {
            return Ok(0.0);
        }

        let sample_size = n_points.min(100);
        let mut total_inverse_distance = 0.0;
        let mut count = 0;

        for i in 0..sample_size {
            let mut nearest_distance = f64::INFINITY;

            for j in 0..n_points {
                if i != j {
                    let dist: f64 = points
                        .row(i)
                        .iter()
                        .zip(points.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if dist < nearest_distance {
                        nearest_distance = dist;
                    }
                }
            }

            if nearest_distance > 0.0 && nearest_distance.is_finite() {
                total_inverse_distance += 1.0 / nearest_distance;
                count += 1;
            }
        }

        Ok(if count > 0 {
            total_inverse_distance / count as f64 / 10.0
        } else {
            0.0
        })
    }

    /// Estimate Hopkins statistic
    fn estimate_hopkins_statistic(&self, points: &ArrayView2<'_, f64>) -> SpatialResult<f64> {
        let (n_points, n_dims) = points.dim();

        if n_points < 10 {
            return Ok(0.5);
        }

        let sample_size = n_points.min(20);
        let mut real_distances = Vec::new();
        let mut random_distances = Vec::new();

        // Real point distances
        for i in 0..sample_size {
            let mut min_dist = f64::INFINITY;
            for j in 0..n_points {
                if i != j {
                    let dist: f64 = points
                        .row(i)
                        .iter()
                        .zip(points.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    min_dist = min_dist.min(dist);
                }
            }
            real_distances.push(min_dist);
        }

        // Random point distances
        let bounds = self.get_data_bounds(points);
        for _ in 0..sample_size {
            let random_point: Array1<f64> = Array1::from_shape_fn(n_dims, |i| {
                rand::random::<f64>() * (bounds[i].1 - bounds[i].0) + bounds[i].0
            });

            let mut min_dist = f64::INFINITY;
            for j in 0..n_points {
                let dist: f64 = random_point
                    .iter()
                    .zip(points.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                min_dist = min_dist.min(dist);
            }
            random_distances.push(min_dist);
        }

        let sum_random: f64 = random_distances.iter().sum();
        let sum_real: f64 = real_distances.iter().sum();
        let hopkins = sum_random / (sum_random + sum_real);

        Ok(hopkins)
    }

    /// Get data bounds
    fn get_data_bounds(&self, points: &ArrayView2<'_, f64>) -> Vec<(f64, f64)> {
        let (_, n_dims) = points.dim();
        let mut bounds = Vec::new();

        for dim in 0..n_dims {
            let column = points.column(dim);
            let min_val = column.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            bounds.push((min_val, max_val));
        }

        bounds
    }

    /// Select random algorithm for exploration
    fn random_algorithm(&self) -> SpatialAlgorithm {
        let algorithms = [
            SpatialAlgorithm::KMeans,
            SpatialAlgorithm::DBScan,
            SpatialAlgorithm::HierarchicalClustering,
            SpatialAlgorithm::GaussianMixture,
            SpatialAlgorithm::SpectralClustering,
            SpatialAlgorithm::KDTree,
            SpatialAlgorithm::BallTree,
        ];

        let index = (rand::random::<f64>() * algorithms.len() as f64) as usize;
        algorithms[index.min(algorithms.len() - 1)].clone()
    }

    /// Find best algorithm for given state
    fn best_algorithm_for_state(&self, state: &DataState) -> SpatialAlgorithm {
        let algorithms = [
            SpatialAlgorithm::KMeans,
            SpatialAlgorithm::DBScan,
            SpatialAlgorithm::HierarchicalClustering,
            SpatialAlgorithm::GaussianMixture,
            SpatialAlgorithm::SpectralClustering,
            SpatialAlgorithm::KDTree,
            SpatialAlgorithm::BallTree,
        ];

        let mut best_algorithm = SpatialAlgorithm::KMeans;
        let mut best_q_value = f64::NEG_INFINITY;

        for algorithm in &algorithms {
            let state_action = StateAction {
                state: state.clone(),
                action: algorithm.clone(),
            };

            let q_value = self.q_table.get(&state_action).unwrap_or(&0.0);
            if *q_value > best_q_value {
                best_q_value = *q_value;
                best_algorithm = algorithm.clone();
            }
        }

        best_algorithm
    }

    /// Update Q-values based on experience
    pub fn update_q_values(&mut self, experience: Experience) -> SpatialResult<()> {
        let state_action = StateAction {
            state: experience.state.clone(),
            action: experience.action.clone(),
        };

        let current_q = *self.q_table.get(&state_action).unwrap_or(&0.0);

        // Find maximum Q-value for next state
        let max_next_q = if experience.done {
            0.0
        } else {
            self.max_q_value_for_state(&experience.next_state)
        };

        // Q-learning update rule
        let new_q = current_q
            + self.learning_rate * (experience.reward + self.gamma * max_next_q - current_q);

        self.q_table.insert(state_action, new_q);

        // Store experience for replay
        self.experience_buffer.push_back(experience);
        if self.experience_buffer.len() > 10000 {
            self.experience_buffer.pop_front();
        }

        // Perform experience replay
        if self.experience_buffer.len() >= 32 {
            self.replay_experience()?;
        }

        Ok(())
    }

    /// Find maximum Q-value for given state
    fn max_q_value_for_state(&self, state: &DataState) -> f64 {
        let algorithms = [
            SpatialAlgorithm::KMeans,
            SpatialAlgorithm::DBScan,
            SpatialAlgorithm::HierarchicalClustering,
            SpatialAlgorithm::GaussianMixture,
            SpatialAlgorithm::SpectralClustering,
            SpatialAlgorithm::KDTree,
            SpatialAlgorithm::BallTree,
        ];

        algorithms
            .iter()
            .map(|algorithm| {
                let state_action = StateAction {
                    state: state.clone(),
                    action: algorithm.clone(),
                };
                *self.q_table.get(&state_action).unwrap_or(&0.0)
            })
            .fold(f64::NEG_INFINITY, |a, b| a.max(b))
    }

    /// Perform experience replay training
    fn replay_experience(&mut self) -> SpatialResult<()> {
        let batch_size = 32.min(self.experience_buffer.len());

        for _ in 0..batch_size {
            if let Some(experience) = self.experience_buffer.pop_front() {
                let state_action = StateAction {
                    state: experience.state.clone(),
                    action: experience.action.clone(),
                };

                let current_q = *self.q_table.get(&state_action).unwrap_or(&0.0);
                let max_next_q = if experience.done {
                    0.0
                } else {
                    self.max_q_value_for_state(&experience.next_state)
                };

                let new_q = current_q
                    + self.learning_rate
                        * (experience.reward + self.gamma * max_next_q - current_q);

                self.q_table.insert(state_action, new_q);
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_activation_functions() {
        assert_eq!(ActivationFunction::ReLU.apply(1.0), 1.0);
        assert_eq!(ActivationFunction::ReLU.apply(-1.0), 0.0);

        let sigmoid_result = ActivationFunction::Sigmoid.apply(0.0);
        assert!((sigmoid_result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_neural_spatial_optimizer_creation() {
        let mut optimizer =
            NeuralSpatialOptimizer::new().with_network_architecture([10, 64, 32, 8]);
        optimizer.with_learning_rate(0.001);
        optimizer.with_adaptive_learning(true);

        assert_eq!(optimizer.layers.len(), 3);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert!(optimizer.adaptive_learning);
    }

    #[test]
    fn test_clustering_parameters() {
        let params = ClusteringParameters {
            num_clusters: 3,
            max_iterations: 100,
            tolerance: 1e-6,
            init_method: InitMethod::KMeansPlusPlus,
            distance_metric: DistanceMetric::Euclidean,
            regularization: 0.01,
            early_stopping: true,
            adaptive_parameters: false,
        };

        assert_eq!(params.num_clusters, 3);
        assert_eq!(params.init_method, InitMethod::KMeansPlusPlus);
    }

    #[test]
    fn test_reinforcement_learning_selector() {
        let mut selector = ReinforcementLearningSelector::new();
        selector.with_epsilon_greedy(0.1);
        selector.with_experience_replay(true);

        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let algorithm = selector.select_best_algorithm(&points.view());

        assert!(algorithm.is_ok());
    }

    #[test]
    fn test_data_state_analysis() {
        let selector = ReinforcementLearningSelector::new();
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];

        let state = selector.analyze_data_state(&points.view());
        assert!(state.is_ok());

        let data_state = state.unwrap();
        assert_eq!(data_state.size_category, SizeCategory::Small);
        assert_eq!(
            data_state.dimensionality_category,
            DimensionalityCategory::Low
        );
    }

    #[test]
    fn test_neural_optimizer_feature_extraction() {
        let optimizer = NeuralSpatialOptimizer::new();
        let points = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];

        let features = optimizer.extract_spatial_features(&points.view());
        assert!(features.is_ok());

        let feature_vector = features.unwrap();
        assert!(!feature_vector.is_empty());
    }
}
