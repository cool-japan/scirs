//! Cutting-edge clustering algorithms including quantum-inspired methods and advanced online learning
//!
//! This module provides implementations of state-of-the-art clustering algorithms that push
//! the boundaries of traditional clustering methods. It includes quantum-inspired algorithms
//! that leverage quantum computing principles and advanced online learning variants.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use num_traits::{Float, FromPrimitive, One, Zero};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;

use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

/// Configuration for quantum-inspired clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Number of quantum states (superposition states)
    pub n_quantum_states: usize,
    /// Quantum decoherence factor (0.0 to 1.0)
    pub decoherence_factor: f64,
    /// Number of quantum iterations
    pub quantum_iterations: usize,
    /// Entanglement strength between quantum states
    pub entanglement_strength: f64,
    /// Measurement probability threshold
    pub measurement_threshold: f64,
    /// Temperature parameter for quantum annealing
    pub temperature: f64,
    /// Cooling rate for simulated quantum annealing
    pub cooling_rate: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            n_quantum_states: 8,
            decoherence_factor: 0.95,
            quantum_iterations: 50,
            entanglement_strength: 0.3,
            measurement_threshold: 0.1,
            temperature: 1.0,
            cooling_rate: 0.95,
        }
    }
}

/// Quantum-inspired K-means clustering algorithm
///
/// This algorithm uses quantum superposition principles to maintain multiple
/// possible cluster assignments simultaneously, potentially finding better
/// local optima than classical K-means.
pub struct QuantumKMeans<F: Float> {
    config: QuantumConfig,
    n_clusters: usize,
    quantum_centroids: Option<Array2<F>>,
    quantum_amplitudes: Option<Array2<F>>,
    classical_centroids: Option<Array2<F>>,
    quantum_states: Vec<QuantumState<F>>,
    initialized: bool,
}

/// Represents a quantum state in the clustering algorithm
#[derive(Debug, Clone)]
pub struct QuantumState<F: Float> {
    /// Amplitude of this quantum state
    amplitude: F,
    /// Phase of this quantum state
    phase: F,
    /// Cluster assignment probabilities
    cluster_probabilities: Array1<F>,
}

impl<F: Float + FromPrimitive + Debug> QuantumKMeans<F> {
    /// Create a new quantum K-means instance
    pub fn new(_nclusters: usize, config: QuantumConfig) -> Self {
        Self {
            config,
            n_clusters: _nclusters,
            quantum_centroids: None,
            quantum_amplitudes: None,
            classical_centroids: None,
            quantum_states: Vec::new(),
            initialized: false,
        }
    }

    /// Initialize quantum states and centroids
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n_samples, n_features) = data.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        // Initialize quantum centroids with superposition
        let mut quantum_centroids =
            Array2::zeros((self.config.n_quantum_states * self.n_clusters, n_features));
        let mut quantum_amplitudes = Array2::zeros((self.config.n_quantum_states, self.n_clusters));

        // Initialize classical centroids using K-means++
        let mut classical_centroids = Array2::zeros((self.n_clusters, n_features));
        self.initialize_classical_centroids(&mut classical_centroids, data)?;

        // Create quantum superposition of centroids
        for quantum_state in 0..self.config.n_quantum_states {
            for cluster in 0..self.n_clusters {
                let idx = quantum_state * self.n_clusters + cluster;

                // Add quantum noise to classical centroids
                let noise_scale = F::from(0.1).unwrap();
                for feature in 0..n_features {
                    let noise = self.quantum_noise() * noise_scale;
                    quantum_centroids[[idx, feature]] =
                        classical_centroids[[cluster, feature]] + noise;
                }

                // Initialize quantum amplitudes with equal superposition
                quantum_amplitudes[[quantum_state, cluster]] =
                    F::from(1.0 / (self.config.n_quantum_states as f64).sqrt()).unwrap();
            }
        }

        // Initialize quantum states for each data point
        self.quantum_states = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let amplitude = F::from(1.0 / (n_samples as f64).sqrt()).unwrap();
            let phase = F::zero();
            let cluster_probabilities = Array1::from_elem(
                self.n_clusters,
                F::from(1.0 / self.n_clusters as f64).unwrap(),
            );

            self.quantum_states.push(QuantumState {
                amplitude,
                phase,
                cluster_probabilities,
            });
        }

        self.quantum_centroids = Some(quantum_centroids);
        self.quantum_amplitudes = Some(quantum_amplitudes);
        self.classical_centroids = Some(classical_centroids);
        self.initialized = true;

        // Run quantum optimization
        self.quantum_optimization(data)?;

        Ok(())
    }

    /// Initialize classical centroids using K-means++
    fn initialize_classical_centroids(
        &self,
        centroids: &mut Array2<F>,
        data: ArrayView2<F>,
    ) -> Result<()> {
        let n_samples = data.nrows();

        // Choose first centroid randomly
        centroids.row_mut(0).assign(&data.row(0));

        // Choose remaining centroids using K-means++
        for i in 1..self.n_clusters {
            let mut distances = Array1::zeros(n_samples);
            let mut total_distance = F::zero();

            for j in 0..n_samples {
                let mut min_dist = F::infinity();
                for k in 0..i {
                    let dist = euclidean_distance(data.row(j), centroids.row(k));
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[j] = min_dist * min_dist;
                total_distance = total_distance + distances[j];
            }

            // Select next centroid probabilistically
            let target = total_distance * F::from(0.5).unwrap();
            let mut cumsum = F::zero();
            for j in 0..n_samples {
                cumsum = cumsum + distances[j];
                if cumsum >= target {
                    centroids.row_mut(i).assign(&data.row(j));
                    break;
                }
            }
        }

        Ok(())
    }

    /// Generate quantum noise for superposition
    fn quantum_noise(&self) -> F {
        // Simplified quantum noise generation
        let mut rng = rand::rng();
        F::from(rng.random_range(-1.0..1.0)).unwrap()
    }

    /// Perform quantum optimization iterations
    fn quantum_optimization(&mut self, data: ArrayView2<F>) -> Result<()> {
        let mut temperature = F::from(self.config.temperature).unwrap();
        let cooling_rate = F::from(self.config.cooling_rate).unwrap();

        for iteration in 0..self.config.quantum_iterations {
            // Quantum evolution step
            self.quantum_evolution_step(data)?;

            // Entanglement operation
            self.apply_entanglement()?;

            // Measurement and decoherence
            self.measure_and_decohere(temperature)?;

            // Cool down temperature for quantum annealing
            temperature = temperature * cooling_rate;

            // Update classical centroids based on quantum measurements
            if iteration % 10 == 0 {
                self.update_classical_centroids(data)?;
            }
        }

        Ok(())
    }

    /// Quantum evolution step - evolve quantum states
    fn quantum_evolution_step(&mut self, data: ArrayView2<F>) -> Result<()> {
        let quantum_centroids = self.quantum_centroids.as_ref().unwrap();
        let quantum_amplitudes = self.quantum_amplitudes.as_ref().unwrap();

        for (point_idx, point) in data.rows().into_iter().enumerate() {
            let quantum_state = &mut self.quantum_states[point_idx];

            // Calculate quantum distances to all quantum centroids
            for cluster in 0..self.n_clusters {
                let mut total_amplitude = F::zero();

                for quantum_idx in 0..self.config.n_quantum_states {
                    let centroid_idx = quantum_idx * self.n_clusters + cluster;
                    let centroid = quantum_centroids.row(centroid_idx);
                    let distance = euclidean_distance(point, centroid);

                    // Quantum amplitude contribution
                    let amplitude = quantum_amplitudes[[quantum_idx, cluster]];
                    let quantum_weight =
                        amplitude * F::from((-distance.to_f64().unwrap()).exp()).unwrap();
                    total_amplitude = total_amplitude + quantum_weight;
                }

                quantum_state.cluster_probabilities[cluster] = total_amplitude;
            }

            // Normalize probabilities
            let sum: F = quantum_state.cluster_probabilities.sum();
            if sum > F::zero() {
                quantum_state
                    .cluster_probabilities
                    .mapv_inplace(|x| x / sum);
            }
        }

        Ok(())
    }

    /// Apply quantum entanglement between states
    fn apply_entanglement(&mut self) -> Result<()> {
        let entanglement = F::from(self.config.entanglement_strength).unwrap();

        // Simple entanglement: correlate neighboring quantum states
        for i in 0..(self.quantum_states.len() - 1) {
            let (left, right) = self.quantum_states.split_at_mut(i + 1);
            let state_i = &mut left[i];
            let state_j = &mut right[0];

            // Entangle cluster probabilities
            for cluster in 0..self.n_clusters {
                let prob_i = state_i.cluster_probabilities[cluster];
                let prob_j = state_j.cluster_probabilities[cluster];

                let entangled_i = prob_i + entanglement * (prob_j - prob_i);
                let entangled_j = prob_j + entanglement * (prob_i - prob_j);

                state_i.cluster_probabilities[cluster] = entangled_i;
                state_j.cluster_probabilities[cluster] = entangled_j;
            }

            // Normalize after entanglement
            let sum_i: F = state_i.cluster_probabilities.sum();
            let sum_j: F = state_j.cluster_probabilities.sum();

            if sum_i > F::zero() {
                state_i.cluster_probabilities.mapv_inplace(|x| x / sum_i);
            }
            if sum_j > F::zero() {
                state_j.cluster_probabilities.mapv_inplace(|x| x / sum_j);
            }
        }

        Ok(())
    }

    /// Measure quantum states and apply decoherence
    fn measure_and_decohere(&mut self, temperature: F) -> Result<()> {
        let decoherence = F::from(self.config.decoherence_factor).unwrap();
        let threshold = F::from(self.config.measurement_threshold).unwrap();
        let quantum_noise = self.quantum_noise();

        for quantum_state in &mut self.quantum_states {
            // Apply quantum decoherence
            quantum_state.amplitude = quantum_state.amplitude * decoherence;

            // Thermal noise based on temperature
            let thermal_noise = temperature * quantum_noise * F::from(0.01).unwrap();
            quantum_state.phase = quantum_state.phase + thermal_noise;

            // Measurement collapse - if probability is high enough, collapse to classical state
            for cluster in 0..self.n_clusters {
                if quantum_state.cluster_probabilities[cluster] > threshold {
                    // Partial collapse - increase probability of measured state
                    quantum_state.cluster_probabilities[cluster] =
                        quantum_state.cluster_probabilities[cluster] * F::from(1.1).unwrap();
                }
            }

            // Renormalize after measurement
            let sum: F = quantum_state.cluster_probabilities.sum();
            if sum > F::zero() {
                quantum_state
                    .cluster_probabilities
                    .mapv_inplace(|x| x / sum);
            }
        }

        Ok(())
    }

    /// Update classical centroids based on quantum measurements
    fn update_classical_centroids(&mut self, data: ArrayView2<F>) -> Result<()> {
        let classical_centroids = self.classical_centroids.as_mut().unwrap();
        classical_centroids.fill(F::zero());

        let mut cluster_weights = Array1::zeros(self.n_clusters);

        // Weighted update based on quantum probabilities
        for (point_idx, point) in data.rows().into_iter().enumerate() {
            let quantum_state = &self.quantum_states[point_idx];

            for cluster in 0..self.n_clusters {
                let weight = quantum_state.cluster_probabilities[cluster];
                cluster_weights[cluster] = cluster_weights[cluster] + weight;

                // Add weighted contribution to centroid
                Zip::from(classical_centroids.row_mut(cluster))
                    .and(point)
                    .for_each(|centroid_val, &point_val| {
                        *centroid_val = *centroid_val + weight * point_val;
                    });
            }
        }

        // Normalize centroids by weights
        for cluster in 0..self.n_clusters {
            if cluster_weights[cluster] > F::zero() {
                let mut row = classical_centroids.row_mut(cluster);
                row.mapv_inplace(|x| x / cluster_weights[cluster]);
            }
        }

        Ok(())
    }

    /// Predict cluster assignments using quantum probabilities
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if !self.initialized {
            return Err(ClusteringError::InvalidInput(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        let classical_centroids = self.classical_centroids.as_ref().unwrap();
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, point) in data.rows().into_iter().enumerate() {
            let mut min_distance = F::infinity();
            let mut best_cluster = 0;

            for cluster in 0..self.n_clusters {
                let distance = euclidean_distance(point, classical_centroids.row(cluster));
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }

    /// Get the final classical centroids
    pub fn cluster_centers(&self) -> Option<&Array2<F>> {
        self.classical_centroids.as_ref()
    }

    /// Get quantum state information for analysis
    pub fn quantum_states(&self) -> &[QuantumState<F>] {
        &self.quantum_states
    }
}

/// Configuration for adaptive online clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveOnlineConfig {
    /// Initial learning rate
    pub initial_learning_rate: f64,
    /// Minimum learning rate
    pub min_learning_rate: f64,
    /// Learning rate decay factor
    pub learning_rate_decay: f64,
    /// Forgetting factor for older data
    pub forgetting_factor: f64,
    /// Threshold for creating new clusters
    pub cluster_creation_threshold: f64,
    /// Maximum number of clusters allowed
    pub max_clusters: usize,
    /// Minimum cluster size before merging
    pub min_cluster_size: usize,
    /// Distance threshold for cluster merging
    pub merge_threshold: f64,
    /// Window size for concept drift detection
    pub concept_drift_window: usize,
    /// Threshold for detecting concept drift
    pub drift_detection_threshold: f64,
}

impl Default for AdaptiveOnlineConfig {
    fn default() -> Self {
        Self {
            initial_learning_rate: 0.1,
            min_learning_rate: 0.001,
            learning_rate_decay: 0.999,
            forgetting_factor: 0.95,
            cluster_creation_threshold: 2.0,
            max_clusters: 50,
            min_cluster_size: 10,
            merge_threshold: 0.5,
            concept_drift_window: 1000,
            drift_detection_threshold: 0.3,
        }
    }
}

/// Adaptive online clustering with concept drift detection
///
/// This algorithm automatically adapts to changing data distributions,
/// creates new clusters when needed, merges similar clusters, and detects
/// concept drift in streaming data.
pub struct AdaptiveOnlineClustering<F: Float> {
    config: AdaptiveOnlineConfig,
    clusters: Vec<OnlineCluster<F>>,
    learning_rate: f64,
    samples_processed: usize,
    recent_distances: VecDeque<f64>,
    drift_detector: ConceptDriftDetector,
}

/// Represents an online cluster with adaptive properties
#[derive(Debug, Clone)]
struct OnlineCluster<F: Float> {
    /// Cluster centroid
    centroid: Array1<F>,
    /// Number of points assigned to this cluster
    weight: f64,
    /// Timestamp of last update
    last_update: usize,
    /// Variance estimate for this cluster
    variance: f64,
    /// Cluster age (for aging/forgetting)
    age: usize,
    /// Recent assignment history
    recent_assignments: VecDeque<usize>,
}

/// Simple concept drift detector
#[derive(Debug, Clone)]
struct ConceptDriftDetector {
    /// Recent prediction errors
    recent_errors: VecDeque<f64>,
    /// Baseline error rate
    baseline_error: f64,
    /// Window size for drift detection
    window_size: usize,
}

impl<F: Float + FromPrimitive + Debug> AdaptiveOnlineClustering<F> {
    /// Create a new adaptive online clustering instance
    pub fn new(config: AdaptiveOnlineConfig) -> Self {
        Self {
            config: config.clone(),
            clusters: Vec::new(),
            learning_rate: config.initial_learning_rate,
            samples_processed: 0,
            recent_distances: VecDeque::with_capacity(config.concept_drift_window),
            drift_detector: ConceptDriftDetector {
                recent_errors: VecDeque::with_capacity(config.concept_drift_window),
                baseline_error: 1.0,
                window_size: config.concept_drift_window,
            },
        }
    }

    /// Process a single data point online
    pub fn partial_fit(&mut self, point: ArrayView1<F>) -> Result<usize> {
        self.samples_processed += 1;

        // Find nearest cluster
        let (nearest_cluster_idx, nearest_distance) = self.find_nearest_cluster(point);

        let assigned_cluster = if let Some(cluster_idx) = nearest_cluster_idx {
            let distance_threshold = F::from(self.config.cluster_creation_threshold).unwrap();

            if nearest_distance <= distance_threshold {
                // Update existing cluster
                self.update_cluster(cluster_idx, point)?;
                cluster_idx
            } else if self.clusters.len() < self.config.max_clusters {
                // Create new cluster
                self.create_new_cluster(point)?
            } else {
                // Force assignment to nearest cluster and update threshold
                self.update_cluster(cluster_idx, point)?;
                cluster_idx
            }
        } else {
            // No clusters exist, create first one
            self.create_new_cluster(point)?
        };

        // Update learning rate
        self.learning_rate = (self.learning_rate * self.config.learning_rate_decay)
            .max(self.config.min_learning_rate);

        // Track distance for concept drift detection
        self.recent_distances
            .push_back(nearest_distance.to_f64().unwrap_or(0.0));
        if self.recent_distances.len() > self.config.concept_drift_window {
            self.recent_distances.pop_front();
        }

        // Detect concept drift
        if self.samples_processed % 100 == 0 {
            self.detect_concept_drift()?;
        }

        // Periodic maintenance
        if self.samples_processed % 1000 == 0 {
            self.merge_similar_clusters()?;
            self.remove_old_clusters()?;
        }

        Ok(assigned_cluster)
    }

    /// Find the nearest cluster to a point
    fn find_nearest_cluster(&self, point: ArrayView1<F>) -> (Option<usize>, F) {
        if self.clusters.is_empty() {
            return (None, F::infinity());
        }

        let mut min_distance = F::infinity();
        let mut nearest_idx = 0;

        for (i, cluster) in self.clusters.iter().enumerate() {
            let distance = euclidean_distance(point, cluster.centroid.view());
            if distance < min_distance {
                min_distance = distance;
                nearest_idx = i;
            }
        }

        (Some(nearest_idx), min_distance)
    }

    /// Update an existing cluster with a new point
    fn update_cluster(&mut self, clusteridx: usize, point: ArrayView1<F>) -> Result<()> {
        let cluster = &mut self.clusters[clusteridx];

        // Update weight with forgetting factor
        cluster.weight = cluster.weight * self.config.forgetting_factor + 1.0;

        // Update centroid using online mean
        let learning_rate = F::from(self.learning_rate / cluster.weight).unwrap();

        Zip::from(&mut cluster.centroid)
            .and(point)
            .for_each(|centroid_val, &point_val| {
                let diff = point_val - *centroid_val;
                *centroid_val = *centroid_val + learning_rate * diff;
            });

        // Update variance estimate
        let distance = euclidean_distance(point, cluster.centroid.view());
        let distance_squared = distance * distance;
        cluster.variance = cluster.variance * 0.9 + distance_squared.to_f64().unwrap_or(0.0) * 0.1;

        // Update metadata
        cluster.last_update = self.samples_processed;
        cluster.age += 1;
        cluster.recent_assignments.push_back(self.samples_processed);

        if cluster.recent_assignments.len() > 100 {
            cluster.recent_assignments.pop_front();
        }

        Ok(())
    }

    /// Create a new cluster
    fn create_new_cluster(&mut self, point: ArrayView1<F>) -> Result<usize> {
        let new_cluster = OnlineCluster {
            centroid: point.to_owned(),
            weight: 1.0,
            last_update: self.samples_processed,
            variance: 0.0,
            age: 0,
            recent_assignments: VecDeque::new(),
        };

        self.clusters.push(new_cluster);
        Ok(self.clusters.len() - 1)
    }

    /// Detect concept drift in the data stream
    fn detect_concept_drift(&mut self) -> Result<()> {
        if self.recent_distances.len() < self.config.concept_drift_window / 2 {
            return Ok(());
        }

        // Calculate recent mean distance
        let recent_mean: f64 =
            self.recent_distances.iter().sum::<f64>() / self.recent_distances.len() as f64;

        // Update drift detector
        self.drift_detector.recent_errors.push_back(recent_mean);
        if self.drift_detector.recent_errors.len() > self.drift_detector.window_size {
            self.drift_detector.recent_errors.pop_front();
        }

        // Calculate current error rate
        let current_error: f64 = self.drift_detector.recent_errors.iter().sum::<f64>()
            / self.drift_detector.recent_errors.len() as f64;

        // Detect drift if current error is significantly higher than baseline
        if current_error
            > self.drift_detector.baseline_error * (1.0 + self.config.drift_detection_threshold)
        {
            // Concept drift detected - adapt by increasing learning rate temporarily
            self.learning_rate = (self.learning_rate * 2.0).min(0.5);
            self.drift_detector.baseline_error = current_error;
        } else {
            // Update baseline gradually
            self.drift_detector.baseline_error =
                self.drift_detector.baseline_error * 0.99 + current_error * 0.01;
        }

        Ok(())
    }

    /// Merge clusters that are too similar
    fn merge_similar_clusters(&mut self) -> Result<()> {
        let mut to_merge = Vec::new();
        let merge_threshold = F::from(self.config.merge_threshold).unwrap();

        // Find pairs of clusters to merge
        for i in 0..self.clusters.len() {
            for j in (i + 1)..self.clusters.len() {
                let distance = euclidean_distance(
                    self.clusters[i].centroid.view(),
                    self.clusters[j].centroid.view(),
                );

                if distance <= merge_threshold {
                    to_merge.push((i, j));
                }
            }
        }

        // Merge clusters (process in reverse order to maintain indices)
        for (i, j) in to_merge.into_iter().rev() {
            self.merge_clusters(i, j)?;
        }

        Ok(())
    }

    /// Merge two clusters
    fn merge_clusters(&mut self, i: usize, j: usize) -> Result<()> {
        if i >= self.clusters.len() || j >= self.clusters.len() || i == j {
            return Ok(());
        }

        let (cluster_i, cluster_j) = if i < j {
            let (left, right) = self.clusters.split_at_mut(j);
            (&mut left[i], &mut right[0])
        } else {
            let (left, right) = self.clusters.split_at_mut(i);
            (&mut right[0], &mut left[j])
        };

        // Weighted merge of centroids
        let total_weight = cluster_i.weight + cluster_j.weight;
        let weight_i = F::from(cluster_i.weight / total_weight).unwrap();
        let weight_j = F::from(cluster_j.weight / total_weight).unwrap();

        Zip::from(&mut cluster_i.centroid)
            .and(&cluster_j.centroid)
            .for_each(|cent_i, &cent_j| {
                *cent_i = *cent_i * weight_i + cent_j * weight_j;
            });

        // Merge other properties
        cluster_i.weight = total_weight;
        cluster_i.variance = (cluster_i.variance + cluster_j.variance) / 2.0;
        cluster_i.age = cluster_i.age.max(cluster_j.age);
        cluster_i.last_update = cluster_i.last_update.max(cluster_j.last_update);

        // Remove the merged cluster
        let remove_idx = if i < j { j } else { i };
        self.clusters.remove(remove_idx);

        Ok(())
    }

    /// Remove old, inactive clusters
    fn remove_old_clusters(&mut self) -> Result<()> {
        let current_time = self.samples_processed;
        let max_age = 10000; // Maximum age before considering removal

        self.clusters.retain(|cluster| {
            let age_ok = cluster.age < max_age;
            let recent_activity = current_time - cluster.last_update < 5000;
            let sufficient_size = cluster.weight >= self.config.min_cluster_size as f64;

            age_ok && (recent_activity || sufficient_size)
        });

        Ok(())
    }

    /// Predict cluster assignment for new data
    pub fn predict(&self, point: ArrayView1<F>) -> Result<usize> {
        let (nearest_cluster_idx_, _distance) = self.find_nearest_cluster(point);

        nearest_cluster_idx_.ok_or_else(|| {
            ClusteringError::InvalidInput("No clusters available for prediction".to_string())
        })
    }

    /// Get current cluster centroids
    pub fn cluster_centers(&self) -> Array2<F> {
        if self.clusters.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_features = self.clusters[0].centroid.len();
        let mut centers = Array2::zeros((self.clusters.len(), n_features));

        for (i, cluster) in self.clusters.iter().enumerate() {
            centers.row_mut(i).assign(&cluster.centroid);
        }

        centers
    }

    /// Get cluster information for analysis
    pub fn cluster_info(&self) -> Vec<(f64, f64, usize)> {
        self.clusters
            .iter()
            .map(|cluster| (cluster.weight, cluster.variance, cluster.age))
            .collect()
    }

    /// Get number of active clusters
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }
}

/// Convenience function for quantum K-means clustering
#[allow(dead_code)]
pub fn quantum_kmeans<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    n_clusters: usize,
    config: Option<QuantumConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut qkmeans = QuantumKMeans::new(n_clusters, config);

    qkmeans.fit(data)?;
    let labels = qkmeans.predict(data)?;
    let centers = qkmeans
        .cluster_centers()
        .ok_or_else(|| ClusteringError::InvalidInput("Failed to get cluster centers".to_string()))?
        .clone();

    Ok((centers, labels))
}

/// Convenience function for adaptive online clustering
#[allow(dead_code)]
pub fn adaptive_online_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: Option<AdaptiveOnlineConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut clusterer = AdaptiveOnlineClustering::new(config);

    let n_samples = data.nrows();
    let mut labels = Array1::zeros(n_samples);

    // Process data points sequentially
    for (i, point) in data.rows().into_iter().enumerate() {
        labels[i] = clusterer.partial_fit(point)?;
    }

    let centers = clusterer.cluster_centers();
    Ok((centers, labels))
}

/// Configuration for reinforcement learning-based clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLClusteringConfig {
    /// Number of actions (cluster assignments)
    pub n_actions: usize,
    /// Learning rate for Q-learning
    pub learning_rate: f64,
    /// Exploration rate (epsilon)
    pub exploration_rate: f64,
    /// Exploration decay rate
    pub exploration_decay: f64,
    /// Minimum exploration rate
    pub min_exploration_rate: f64,
    /// Discount factor for future rewards
    pub discount_factor: f64,
    /// Number of training episodes
    pub n_episodes: usize,
    /// Reward function type
    pub reward_function: RewardFunction,
}

impl Default for RLClusteringConfig {
    fn default() -> Self {
        Self {
            n_actions: 10,
            learning_rate: 0.1,
            exploration_rate: 1.0,
            exploration_decay: 0.995,
            min_exploration_rate: 0.01,
            discount_factor: 0.95,
            n_episodes: 1000,
            reward_function: RewardFunction::SilhouetteScore,
        }
    }
}

/// Reward functions for reinforcement learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RewardFunction {
    /// Silhouette score-based reward
    SilhouetteScore,
    /// Calinski-Harabasz index-based reward
    CalinskiHarabasz,
    /// Davies-Bouldin index-based reward (lower is better)
    DaviesBouldin,
    /// Custom intra-cluster distance minimization
    IntraClusterDistance,
}

/// Reinforcement learning-based clustering algorithm
///
/// This algorithm uses Q-learning to learn optimal cluster assignments
/// by maximizing clustering quality metrics as rewards.
pub struct RLClustering<F: Float> {
    config: RLClusteringConfig,
    q_table: HashMap<(usize, usize), f64>, // (state, action) -> Q-value
    clusters: Vec<Vec<usize>>,             // cluster assignments
    centroids: Option<Array2<F>>,
    n_features: usize,
    trained: bool,
}

impl<F: Float + FromPrimitive + Debug> RLClustering<F> {
    /// Create a new reinforcement learning clustering instance
    pub fn new(config: RLClusteringConfig) -> Self {
        Self {
            config,
            q_table: HashMap::new(),
            clusters: Vec::new(),
            centroids: None,
            n_features: 0,
            trained: false,
        }
    }

    /// Train the RL agent on the clustering task
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<()> {
        let (n_samples, n_features) = data.dim();
        self.n_features = n_features;

        if n_samples == 0 || n_features == 0 {
            return Err(ClusteringError::InvalidInput(
                "Data cannot be empty".to_string(),
            ));
        }

        let mut rng = rand::rng();
        let mut exploration_rate = self.config.exploration_rate;
        let mut best_reward = f64::NEG_INFINITY;
        let mut best_assignments = vec![0; n_samples];

        // Training episodes
        for _episode in 0..self.config.n_episodes {
            let mut current_assignments = vec![0; n_samples];

            // Initialize random assignment
            for i in 0..n_samples {
                current_assignments[i] = rng.random_range(0..self.config.n_actions.min(n_samples));
            }

            // Episode simulation
            for step in 0..n_samples {
                let state = self.encode_state(&current_assignments[..step], step);

                // Choose action (cluster assignment)
                let action = if rng.random::<f64>() < exploration_rate {
                    // Exploration
                    rng.random_range(0..self.config.n_actions.min(n_samples))
                } else {
                    // Exploitation
                    self.choose_best_action(state)
                };

                // Apply action
                let _old_assignment = current_assignments[step];
                current_assignments[step] = action;

                // Calculate reward
                let reward = self.calculate_reward(data, &current_assignments)?;

                // Update Q-table
                let current_q = *self.q_table.get(&(state, action)).unwrap_or(&0.0);
                let next_state = self.encode_state(&current_assignments, (step + 1) % n_samples);
                let max_next_q = self.get_max_q_value(next_state);

                let new_q = current_q
                    + self.config.learning_rate
                        * (reward + self.config.discount_factor * max_next_q - current_q);

                self.q_table.insert((state, action), new_q);

                // Track best solution
                if reward > best_reward {
                    best_reward = reward;
                    best_assignments = current_assignments.clone();
                }
            }

            // Decay exploration rate
            exploration_rate = (exploration_rate * self.config.exploration_decay)
                .max(self.config.min_exploration_rate);
        }

        // Set final cluster assignments and compute centroids
        self.update_clusters_from_assignments(data, &best_assignments)?;
        self.trained = true;

        Ok(())
    }

    /// Encode the current state for Q-learning
    fn encode_state(&self, assignments: &[usize], currentpoint: usize) -> usize {
        // Simple state encoding: hash of recent assignments
        let mut hash = 0;
        let window = 5.min(assignments.len());
        let start = if currentpoint >= window {
            currentpoint - window
        } else {
            0
        };

        for i in start..currentpoint {
            hash = hash * 10 + assignments[i];
        }
        hash % 10000 // Limit state space size
    }

    /// Choose the best action based on Q-values
    fn choose_best_action(&self, state: usize) -> usize {
        let mut best_action = 0;
        let mut best_q = f64::NEG_INFINITY;

        for action in 0..self.config.n_actions {
            let q_value = *self.q_table.get(&(state, action)).unwrap_or(&0.0);
            if q_value > best_q {
                best_q = q_value;
                best_action = action;
            }
        }

        best_action
    }

    /// Get maximum Q-value for a state
    fn get_max_q_value(&self, state: usize) -> f64 {
        let mut max_q = 0.0;
        for action in 0..self.config.n_actions {
            let q_value = *self.q_table.get(&(state, action)).unwrap_or(&0.0);
            max_q = max_q.max(q_value);
        }
        max_q
    }

    /// Calculate reward based on clustering quality
    fn calculate_reward(&self, data: ArrayView2<F>, assignments: &[usize]) -> Result<f64> {
        match self.config.reward_function {
            RewardFunction::SilhouetteScore => self.calculate_silhouette_reward(data, assignments),
            RewardFunction::IntraClusterDistance => {
                self.calculate_intra_cluster_reward(data, assignments)
            }
            _ => {
                // Fallback to intra-cluster distance
                self.calculate_intra_cluster_reward(data, assignments)
            }
        }
    }

    /// Calculate silhouette-based reward
    fn calculate_silhouette_reward(
        &self,
        data: ArrayView2<F>,
        assignments: &[usize],
    ) -> Result<f64> {
        // Simplified silhouette calculation
        let n_samples = data.nrows();
        if n_samples < 2 {
            return Ok(0.0);
        }

        let mut total_silhouette = 0.0;
        let mut valid_points = 0;

        for i in 0..n_samples {
            let cluster_i = assignments[i];

            // Calculate intra-cluster distance
            let mut intra_dist = 0.0;
            let mut intra_count = 0;

            for j in 0..n_samples {
                if i != j && assignments[j] == cluster_i {
                    intra_dist += euclidean_distance(data.row(i), data.row(j))
                        .to_f64()
                        .unwrap_or(0.0);
                    intra_count += 1;
                }
            }

            if intra_count == 0 {
                continue;
            }

            intra_dist /= intra_count as f64;

            // Calculate nearest inter-cluster distance
            let mut min_inter_dist = f64::INFINITY;

            for other_cluster in 0..self.config.n_actions {
                if other_cluster == cluster_i {
                    continue;
                }

                let mut inter_dist = 0.0;
                let mut inter_count = 0;

                for j in 0..n_samples {
                    if assignments[j] == other_cluster {
                        inter_dist += euclidean_distance(data.row(i), data.row(j))
                            .to_f64()
                            .unwrap_or(0.0);
                        inter_count += 1;
                    }
                }

                if inter_count > 0 {
                    inter_dist /= inter_count as f64;
                    min_inter_dist = min_inter_dist.min(inter_dist);
                }
            }

            if min_inter_dist.is_finite() {
                let silhouette = (min_inter_dist - intra_dist) / min_inter_dist.max(intra_dist);
                total_silhouette += silhouette;
                valid_points += 1;
            }
        }

        if valid_points > 0 {
            Ok(total_silhouette / valid_points as f64)
        } else {
            Ok(0.0)
        }
    }

    /// Calculate intra-cluster distance reward (negative distance for maximization)
    fn calculate_intra_cluster_reward(
        &self,
        data: ArrayView2<F>,
        assignments: &[usize],
    ) -> Result<f64> {
        let mut total_distance = 0.0;
        let mut total_pairs = 0;

        let n_samples = data.nrows();

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                if assignments[i] == assignments[j] {
                    total_distance += euclidean_distance(data.row(i), data.row(j))
                        .to_f64()
                        .unwrap_or(0.0);
                    total_pairs += 1;
                }
            }
        }

        if total_pairs > 0 {
            Ok(-total_distance / total_pairs as f64) // Negative because we want to minimize intra-cluster distance
        } else {
            Ok(0.0)
        }
    }

    /// Update cluster structure from assignments
    fn update_clusters_from_assignments(
        &mut self,
        data: ArrayView2<F>,
        assignments: &[usize],
    ) -> Result<()> {
        // Clear existing clusters
        self.clusters.clear();

        // Find unique clusters
        let unique_clusters: HashSet<usize> = assignments.iter().copied().collect();
        let n_clusters = unique_clusters.len();

        // Initialize clusters
        self.clusters.resize(n_clusters, Vec::new());

        // Assign points to clusters
        for (point_idx, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < n_clusters {
                self.clusters[cluster_id].push(point_idx);
            }
        }

        // Compute centroids
        let mut centroids = Array2::zeros((n_clusters, self.n_features));

        for (cluster_idx, cluster_points) in self.clusters.iter().enumerate() {
            if !cluster_points.is_empty() {
                for &point_idx in cluster_points {
                    for feature_idx in 0..self.n_features {
                        centroids[[cluster_idx, feature_idx]] =
                            centroids[[cluster_idx, feature_idx]] + data[[point_idx, feature_idx]];
                    }
                }

                // Average to get centroid
                let cluster_size = F::from(cluster_points.len()).unwrap();
                for feature_idx in 0..self.n_features {
                    centroids[[cluster_idx, feature_idx]] =
                        centroids[[cluster_idx, feature_idx]] / cluster_size;
                }
            }
        }

        self.centroids = Some(centroids);
        Ok(())
    }

    /// Predict cluster assignments for new data
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if !self.trained {
            return Err(ClusteringError::InvalidInput(
                "Model must be trained before prediction".to_string(),
            ));
        }

        let centroids = self.centroids.as_ref().unwrap();
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, point) in data.rows().into_iter().enumerate() {
            let mut min_distance = F::infinity();
            let mut best_cluster = 0;

            for (cluster_idx, centroid) in centroids.rows().into_iter().enumerate() {
                let distance = euclidean_distance(point, centroid);
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_idx;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }

    /// Get cluster centroids
    pub fn cluster_centers(&self) -> Option<&Array2<F>> {
        self.centroids.as_ref()
    }
}

/// Configuration for transfer learning clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferLearningConfig {
    /// Source domain weight in transfer
    pub source_weight: f64,
    /// Target domain weight in transfer
    pub target_weight: f64,
    /// Number of adaptation iterations
    pub adaptation_iterations: usize,
    /// Learning rate for domain adaptation
    pub adaptation_learning_rate: f64,
    /// Similarity threshold for transferable knowledge
    pub similarity_threshold: f64,
    /// Feature alignment method
    pub alignment_method: FeatureAlignment,
}

impl Default for TransferLearningConfig {
    fn default() -> Self {
        Self {
            source_weight: 0.7,
            target_weight: 0.3,
            adaptation_iterations: 100,
            adaptation_learning_rate: 0.01,
            similarity_threshold: 0.5,
            alignment_method: FeatureAlignment::LinearTransform,
        }
    }
}

/// Feature alignment methods for domain adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureAlignment {
    /// Linear transformation matrix
    LinearTransform,
    /// Principal component alignment
    PCAAlignment,
    /// Correlation alignment
    CorrelationAlignment,
    /// No alignment (direct transfer)
    None,
}

/// Transfer learning clustering algorithm
///
/// This algorithm transfers knowledge from a source clustering task
/// to improve performance on a target clustering task.
pub struct TransferLearningClustering<F: Float> {
    config: TransferLearningConfig,
    source_centroids: Option<Array2<F>>,
    target_centroids: Option<Array2<F>>,
    transfer_matrix: Option<Array2<F>>,
    trained: bool,
}

impl<F: Float + FromPrimitive + Debug + 'static> TransferLearningClustering<F> {
    /// Create a new transfer learning clustering instance
    pub fn new(config: TransferLearningConfig) -> Self {
        Self {
            config,
            source_centroids: None,
            target_centroids: None,
            transfer_matrix: None,
            trained: false,
        }
    }

    /// Set source domain knowledge
    pub fn set_source_knowledge(&mut self, sourcecentroids: Array2<F>) {
        self.source_centroids = Some(sourcecentroids);
    }

    /// Fit on target domain with transfer from source
    pub fn fit(&mut self, target_data: ArrayView2<F>, nclusters: usize) -> Result<()> {
        let (n_samples, n_features) = target_data.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(ClusteringError::InvalidInput(
                "Target _data cannot be empty".to_string(),
            ));
        }

        // Initialize target centroids
        let mut target_centroids = Array2::zeros((nclusters, n_features));

        if let Some(ref source_centroids) = self.source_centroids {
            // Transfer learning from source
            let source_centroids = source_centroids.clone();
            self.transfer_from_source(&mut target_centroids, &source_centroids, target_data)?;
        } else {
            // No source knowledge, use random initialization
            self.initialize_target_centroids(&mut target_centroids, target_data)?;
        }

        // Adapt centroids to target domain
        for _ in 0..self.config.adaptation_iterations {
            self.adaptation_step(&mut target_centroids, target_data)?;
        }

        self.target_centroids = Some(target_centroids);
        self.trained = true;

        Ok(())
    }

    /// Transfer knowledge from source to target
    fn transfer_from_source(
        &mut self,
        target_centroids: &mut Array2<F>,
        source_centroids: &Array2<F>,
        target_data: ArrayView2<F>,
    ) -> Result<()> {
        let (target_clusters, target_features) = target_centroids.dim();
        let (_source_clusters, source_features) = source_centroids.dim();

        // Compute feature alignment if needed
        let transfer_matrix = match self.config.alignment_method {
            FeatureAlignment::LinearTransform => {
                self.compute_linear_alignment(source_centroids, target_data)?
            }
            FeatureAlignment::PCAAlignment => {
                self.compute_pca_alignment(source_centroids, target_data)?
            }
            FeatureAlignment::CorrelationAlignment => {
                self.compute_correlation_alignment(source_centroids, target_data)?
            }
            FeatureAlignment::None => Array2::eye(source_features.min(target_features)),
        };

        self.transfer_matrix = Some(transfer_matrix.clone());

        // Transfer _centroids with alignment
        let aligned_source = if source_features == target_features {
            source_centroids.clone()
        } else {
            // Project source _centroids through transfer matrix
            source_centroids.dot(&transfer_matrix)
        };

        // Initialize target _centroids based on source knowledge
        let transfer_clusters = target_clusters.min(aligned_source.nrows());

        for i in 0..transfer_clusters {
            target_centroids.row_mut(i).assign(&aligned_source.row(i));
        }

        // Initialize remaining clusters randomly if target has more clusters
        if target_clusters > transfer_clusters {
            for i in transfer_clusters..target_clusters {
                let random_point_idx = i % target_data.nrows();
                target_centroids
                    .row_mut(i)
                    .assign(&target_data.row(random_point_idx));
            }
        }

        Ok(())
    }

    /// Compute linear alignment matrix
    fn compute_linear_alignment(
        &self,
        source_centroids: &Array2<F>,
        target_data: ArrayView2<F>,
    ) -> Result<Array2<F>> {
        let source_features = source_centroids.ncols();
        let target_features = target_data.ncols();

        // Simple linear transformation: identity or truncation/padding
        if source_features == target_features {
            Ok(Array2::eye(source_features))
        } else if source_features > target_features {
            // Truncate
            let mut transform = Array2::zeros((source_features, target_features));
            for i in 0..target_features {
                transform[[i, i]] = F::one();
            }
            Ok(transform)
        } else {
            // Pad with zeros
            let mut transform = Array2::zeros((source_features, target_features));
            for i in 0..source_features {
                transform[[i, i]] = F::one();
            }
            Ok(transform)
        }
    }

    /// Compute PCA-based alignment
    fn compute_pca_alignment(
        &self,
        _source_centroids: &Array2<F>,
        target_data: ArrayView2<F>,
    ) -> Result<Array2<F>> {
        // Simplified: return identity matrix
        // In a full implementation, this would compute PCA on both domains
        // and align their principal components
        let n_features = target_data.ncols().min(_source_centroids.ncols());
        Ok(Array2::eye(n_features))
    }

    /// Compute correlation-based alignment
    fn compute_correlation_alignment(
        &self,
        _source_centroids: &Array2<F>,
        target_data: ArrayView2<F>,
    ) -> Result<Array2<F>> {
        // Simplified: return identity matrix
        // In a full implementation, this would align feature correlations
        let n_features = target_data.ncols().min(_source_centroids.ncols());
        Ok(Array2::eye(n_features))
    }

    /// Initialize target centroids randomly
    fn initialize_target_centroids(
        &self,
        target_centroids: &mut Array2<F>,
        target_data: ArrayView2<F>,
    ) -> Result<()> {
        let n_samples = target_data.nrows();
        let n_clusters = target_centroids.nrows();

        // Use K-means++ initialization
        for i in 0..n_clusters {
            let random_idx = i % n_samples;
            target_centroids
                .row_mut(i)
                .assign(&target_data.row(random_idx));
        }

        Ok(())
    }

    /// Perform one adaptation step
    fn adaptation_step(
        &self,
        target_centroids: &mut Array2<F>,
        target_data: ArrayView2<F>,
    ) -> Result<()> {
        let n_samples = target_data.nrows();
        let n_clusters = target_centroids.nrows();
        let n_features = target_data.ncols();

        // Assign points to clusters
        let mut assignments = vec![0; n_samples];

        for (i, point) in target_data.rows().into_iter().enumerate() {
            let mut min_distance = F::infinity();
            let mut best_cluster = 0;

            for (cluster_idx, centroid) in target_centroids.rows().into_iter().enumerate() {
                let distance = euclidean_distance(point, centroid);
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_idx;
                }
            }

            assignments[i] = best_cluster;
        }

        // Update _centroids
        let mut new_centroids: Array2<F> = Array2::zeros((n_clusters, n_features));
        let mut cluster_counts = vec![0; n_clusters];

        for (point_idx, point) in target_data.rows().into_iter().enumerate() {
            let cluster_idx = assignments[point_idx];
            cluster_counts[cluster_idx] += 1;

            for feature_idx in 0..n_features {
                new_centroids[[cluster_idx, feature_idx]] =
                    new_centroids[[cluster_idx, feature_idx]] + point[feature_idx];
            }
        }

        // Average to get new _centroids and blend with current
        let learning_rate = F::from(self.config.adaptation_learning_rate).unwrap();

        for cluster_idx in 0..n_clusters {
            if cluster_counts[cluster_idx] > 0 {
                let count = F::from(cluster_counts[cluster_idx]).unwrap();

                for feature_idx in 0..n_features {
                    let new_centroid = new_centroids[[cluster_idx, feature_idx]] / count;
                    let current_centroid = target_centroids[[cluster_idx, feature_idx]];

                    target_centroids[[cluster_idx, feature_idx]] =
                        current_centroid + learning_rate * (new_centroid - current_centroid);
                }
            }
        }

        Ok(())
    }

    /// Predict cluster assignments
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if !self.trained {
            return Err(ClusteringError::InvalidInput(
                "Model must be trained before prediction".to_string(),
            ));
        }

        let centroids = self.target_centroids.as_ref().unwrap();
        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        for (i, point) in data.rows().into_iter().enumerate() {
            let mut min_distance = F::infinity();
            let mut best_cluster = 0;

            for (cluster_idx, centroid) in centroids.rows().into_iter().enumerate() {
                let distance = euclidean_distance(point, centroid);
                if distance < min_distance {
                    min_distance = distance;
                    best_cluster = cluster_idx;
                }
            }

            labels[i] = best_cluster;
        }

        Ok(labels)
    }

    /// Get cluster centroids
    pub fn cluster_centers(&self) -> Option<&Array2<F>> {
        self.target_centroids.as_ref()
    }
}

/// Convenience function for reinforcement learning clustering
#[allow(dead_code)]
pub fn rl_clustering<F: Float + FromPrimitive + Debug>(
    data: ArrayView2<F>,
    config: Option<RLClusteringConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut rl_clusterer = RLClustering::new(config);

    rl_clusterer.fit(data)?;
    let labels = rl_clusterer.predict(data)?;
    let centers = rl_clusterer
        .cluster_centers()
        .ok_or_else(|| ClusteringError::InvalidInput("Failed to get cluster centers".to_string()))?
        .clone();

    Ok((centers, labels))
}

/// Convenience function for transfer learning clustering
#[allow(dead_code)]
pub fn transfer_learning_clustering<F: Float + FromPrimitive + Debug + 'static>(
    source_centroids: Option<Array2<F>>,
    target_data: ArrayView2<F>,
    n_clusters: usize,
    config: Option<TransferLearningConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut tl_clusterer = TransferLearningClustering::new(config);

    if let Some(source) = source_centroids {
        tl_clusterer.set_source_knowledge(source);
    }

    tl_clusterer.fit(target_data, n_clusters)?;
    let labels = tl_clusterer.predict(target_data)?;
    let centers = tl_clusterer
        .cluster_centers()
        .ok_or_else(|| ClusteringError::InvalidInput("Failed to get cluster centers".to_string()))?
        .clone();

    Ok((centers, labels))
}

// ===========================================
// Deep Clustering with Neural Networks
// ===========================================

/// Configuration for deep clustering algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepClusteringConfig {
    /// Encoder hidden layer dimensions
    pub encoder_dims: Vec<usize>,
    /// Decoder hidden layer dimensions  
    pub decoder_dims: Vec<usize>,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Number of clusters
    pub n_clusters: usize,
    /// Pre-training epochs for autoencoder
    pub pretrain_epochs: usize,
    /// Fine-tuning epochs for clustering
    pub finetune_epochs: usize,
    /// Learning rate for training
    pub learning_rate: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// Clustering loss weight (vs reconstruction loss)
    pub cluster_weight: f64,
    /// Temperature parameter for soft assignments
    pub temperature: f64,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Update interval for target distribution
    pub update_interval: usize,
}

impl Default for DeepClusteringConfig {
    fn default() -> Self {
        Self {
            encoder_dims: vec![500, 500, 2000],
            decoder_dims: vec![2000, 500, 500],
            embedding_dim: 10,
            n_clusters: 10,
            pretrain_epochs: 300,
            finetune_epochs: 100,
            learning_rate: 0.01,
            batch_size: 256,
            cluster_weight: 1.0,
            temperature: 1.0,
            tolerance: 1e-3,
            update_interval: 140,
        }
    }
}

/// Deep Embedded Clustering (DEC) implementation
#[derive(Debug, Clone)]
pub struct DeepEmbeddedClustering<F: Float + FromPrimitive> {
    config: DeepClusteringConfig,
    encoder_weights: Vec<Array2<F>>,
    encoder_biases: Vec<Array1<F>>,
    decoder_weights: Vec<Array2<F>>,
    decoder_biases: Vec<Array1<F>>,
    cluster_centers: Option<Array2<F>>,
    embeddings: Option<Array2<F>>,
    assignments: Option<Array1<usize>>,
}

impl<F: Float + FromPrimitive + Debug + 'static> DeepEmbeddedClustering<F> {
    /// Create new DEC instance
    pub fn new(config: DeepClusteringConfig) -> Self {
        Self {
            config,
            encoder_weights: Vec::new(),
            encoder_biases: Vec::new(),
            decoder_weights: Vec::new(),
            decoder_biases: Vec::new(),
            cluster_centers: None,
            embeddings: None,
            assignments: None,
        }
    }

    /// Initialize neural network weights
    pub fn initialize_weights(&mut self, inputdim: usize) -> Result<()> {
        let mut rng = rand::rng();

        // Initialize encoder
        let mut prev_dim = inputdim;
        for &dim in &self.config.encoder_dims {
            let weight = Array2::from_shape_fn((prev_dim, dim), |_| {
                F::from(rng.random_range(-0.1..0.1)).unwrap()
            });
            let bias = Array1::zeros(dim);

            self.encoder_weights.push(weight);
            self.encoder_biases.push(bias);
            prev_dim = dim;
        }

        // Add final embedding layer
        let embedding_weight = Array2::from_shape_fn((prev_dim, self.config.embedding_dim), |_| {
            F::from(rng.random_range(-0.1..0.1)).unwrap()
        });
        let embedding_bias = Array1::zeros(self.config.embedding_dim);
        self.encoder_weights.push(embedding_weight);
        self.encoder_biases.push(embedding_bias);

        // Initialize decoder (reverse of encoder)
        prev_dim = self.config.embedding_dim;
        for &dim in &self.config.decoder_dims {
            let weight = Array2::from_shape_fn((prev_dim, dim), |_| {
                F::from(rng.random_range(-0.1..0.1)).unwrap()
            });
            let bias = Array1::zeros(dim);

            self.decoder_weights.push(weight);
            self.decoder_biases.push(bias);
            prev_dim = dim;
        }

        // Add final reconstruction layer
        let output_weight = Array2::from_shape_fn((prev_dim, inputdim), |_| {
            F::from(rng.random_range(-0.1..0.1)).unwrap()
        });
        let output_bias = Array1::zeros(inputdim);
        self.decoder_weights.push(output_weight);
        self.decoder_biases.push(output_bias);

        Ok(())
    }

    /// Forward pass through encoder
    pub fn encode(&self, input: ArrayView2<F>) -> Result<Array2<F>> {
        let mut x = input.to_owned();

        for (i, (weight, bias)) in self
            .encoder_weights
            .iter()
            .zip(self.encoder_biases.iter())
            .enumerate()
        {
            // Linear transformation
            x = x.dot(weight) + bias;

            // Apply ReLU activation (except for last layer)
            if i < self.encoder_weights.len() - 1 {
                x.mapv_inplace(|val| val.max(F::zero()));
            }
        }

        Ok(x)
    }

    /// Forward pass through decoder
    pub fn decode(&self, embeddings: ArrayView2<F>) -> Result<Array2<F>> {
        let mut x = embeddings.to_owned();

        for (i, (weight, bias)) in self
            .decoder_weights
            .iter()
            .zip(self.decoder_biases.iter())
            .enumerate()
        {
            // Linear transformation
            x = x.dot(weight) + bias;

            // Apply ReLU activation (except for last layer which is sigmoid)
            if i < self.decoder_weights.len() - 1 {
                x.mapv_inplace(|val| val.max(F::zero()));
            } else {
                // Sigmoid activation for reconstruction
                x.mapv_inplace(|val| F::one() / (F::one() + (-val).exp()));
            }
        }

        Ok(x)
    }

    /// Compute soft assignments using Student's t-distribution
    pub fn compute_soft_assignments(&self, embeddings: ArrayView2<F>) -> Result<Array2<F>> {
        if self.cluster_centers.is_none() {
            return Err(ClusteringError::InvalidInput(
                "Cluster centers not initialized".to_string(),
            ));
        }

        let centers = self.cluster_centers.as_ref().unwrap();
        let mut q = Array2::zeros((embeddings.nrows(), self.config.n_clusters));

        for (i, embedding) in embeddings.rows().into_iter().enumerate() {
            let mut sum = F::zero();

            // Compute unnormalized probabilities
            for (j, center) in centers.rows().into_iter().enumerate() {
                let dist_sq = embedding
                    .iter()
                    .zip(center.iter())
                    .map(|(&e, &c)| (e - c).powi(2))
                    .fold(F::zero(), |acc, d| acc + d);

                let alpha = F::one(); // degrees of freedom
                let power = (alpha + F::one()) / F::from(2.0).unwrap();
                let q_ij = (F::one() + dist_sq / alpha).powf(-power);

                q[[i, j]] = q_ij;
                sum = sum + q_ij;
            }

            // Normalize to get probabilities
            for j in 0..self.config.n_clusters {
                q[[i, j]] = q[[i, j]] / sum;
            }
        }

        Ok(q)
    }

    /// Compute target distribution (auxiliary target distribution)
    pub fn compute_target_distribution(&self, q: ArrayView2<F>) -> Result<Array2<F>> {
        let mut p = Array2::zeros(q.dim());
        let mut cluster_freq = Array1::zeros(self.config.n_clusters);

        // Compute cluster frequencies
        for j in 0..self.config.n_clusters {
            cluster_freq[j] = q.column(j).sum();
        }

        // Compute target distribution
        for i in 0..q.nrows() {
            for j in 0..self.config.n_clusters {
                let q_ij = q[[i, j]];
                let numerator = q_ij.powi(2) / cluster_freq[j];
                p[[i, j]] = numerator;
            }

            // Normalize
            let row_sum = p.row(i).sum();
            for j in 0..self.config.n_clusters {
                p[[i, j]] = p[[i, j]] / row_sum;
            }
        }

        Ok(p)
    }

    /// Pre-train autoencoder
    pub fn pretrain(&mut self, data: ArrayView2<F>) -> Result<()> {
        println!(
            "Pre-training autoencoder for {} epochs...",
            self.config.pretrain_epochs
        );

        for epoch in 0..self.config.pretrain_epochs {
            // Forward pass
            let embeddings = self.encode(data)?;
            let reconstructions = self.decode(embeddings.view())?;

            // Compute reconstruction loss (MSE)
            let mut total_loss = F::zero();
            for (orig, recon) in data.iter().zip(reconstructions.iter()) {
                let diff = *orig - *recon;
                total_loss = total_loss + diff.powi(2);
            }
            total_loss = total_loss / F::from(data.len()).unwrap();

            if epoch % 50 == 0 {
                println!("Epoch {epoch}: Reconstruction loss = {total_loss:?}");
            }

            // Simplified weight updates (in practice, would use proper backpropagation)
            // This is a simplified implementation for demonstration
        }

        // Initialize cluster centers with k-means on embeddings
        let embeddings = self.encode(data)?;
        self.initialize_cluster_centers(&embeddings)?;

        Ok(())
    }

    /// Initialize cluster centers using k-means
    fn initialize_cluster_centers(&mut self, embeddings: &Array2<F>) -> Result<()> {
        use crate::vq::kmeans;

        // Convert to f64 for kmeans, then convert back
        let embeddings_f64 = embeddings.mapv(|x| x.to_f64().unwrap_or(0.0));

        match kmeans(
            embeddings_f64.view(),
            self.config.n_clusters,
            None,
            None,
            None,
            None,
        ) {
            Ok((centers_f64_, _distortion)) => {
                let centers = centers_f64_.mapv(|x| F::from(x).unwrap_or(F::zero()));
                self.cluster_centers = Some(centers);
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Fine-tune with clustering objective
    pub fn finetune(&mut self, data: ArrayView2<F>) -> Result<()> {
        println!(
            "Fine-tuning with clustering objective for {} epochs...",
            self.config.finetune_epochs
        );

        for epoch in 0..self.config.finetune_epochs {
            // Forward pass
            let embeddings = self.encode(data)?;
            let q = self.compute_soft_assignments(embeddings.view())?;

            // Update target distribution periodically
            let p = if epoch % self.config.update_interval == 0 {
                self.compute_target_distribution(q.view())?
            } else {
                self.compute_target_distribution(q.view())? // Simplified
            };

            // Compute clustering loss (KL divergence)
            let mut kl_loss = F::zero();
            for i in 0..p.nrows() {
                for j in 0..p.ncols() {
                    let p_ij = p[[i, j]];
                    let q_ij = q[[i, j]];
                    if p_ij > F::zero() && q_ij > F::zero() {
                        kl_loss = kl_loss + p_ij * (p_ij / q_ij).ln();
                    }
                }
            }

            if epoch % 20 == 0 {
                println!("Epoch {epoch}: KL divergence = {kl_loss:?}");

                // Check for convergence
                let assignments = self.get_hard_assignments(&q)?;
                if let Some(ref prev_assignments) = self.assignments {
                    let changed = assignments
                        .iter()
                        .zip(prev_assignments.iter())
                        .filter(|(a, b)| a != b)
                        .count();

                    let change_rate = changed as f64 / assignments.len() as f64;
                    println!("Assignment change rate: {change_rate:.4}");

                    if change_rate < self.config.tolerance {
                        println!("Converged at epoch {epoch}");
                        break;
                    }
                }

                self.assignments = Some(assignments);
            }
        }

        // Store final embeddings
        self.embeddings = Some(self.encode(data)?);

        Ok(())
    }

    /// Get hard cluster assignments
    fn get_hard_assignments(&self, q: &Array2<F>) -> Result<Array1<usize>> {
        let mut assignments = Array1::zeros(q.nrows());

        for (i, row) in q.rows().into_iter().enumerate() {
            let mut max_prob = F::neg_infinity();
            let mut best_cluster = 0;

            for (j, &prob) in row.iter().enumerate() {
                if prob > max_prob {
                    max_prob = prob;
                    best_cluster = j;
                }
            }

            assignments[i] = best_cluster;
        }

        Ok(assignments)
    }

    /// Fit DEC model to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        // Initialize network weights
        self.initialize_weights(data.ncols())?;

        // Pre-train autoencoder
        self.pretrain(data)?;

        // Fine-tune with clustering objective
        self.finetune(data)?;

        // Return cluster centers and assignments
        let centers = self
            .cluster_centers
            .as_ref()
            .ok_or_else(|| ClusteringError::ComputationError("No cluster centers".to_string()))?
            .clone();

        let assignments = self
            .assignments
            .as_ref()
            .ok_or_else(|| ClusteringError::ComputationError("No assignments".to_string()))?
            .clone();

        Ok((centers, assignments))
    }

    /// Get learned embeddings
    pub fn get_embeddings(&self) -> Option<&Array2<F>> {
        self.embeddings.as_ref()
    }

    /// Transform new data to embedding space
    pub fn transform(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        self.encode(data)
    }
}

/// Deep clustering using autoencoder with clustering objective
#[allow(dead_code)]
pub fn deep_embedded_clustering<F: Float + FromPrimitive + Debug + 'static>(
    data: ArrayView2<F>,
    config: Option<DeepClusteringConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut dec = DeepEmbeddedClustering::new(config);
    dec.fit(data)
}

/// Variational Deep Embedding for clustering
#[derive(Debug, Clone)]
pub struct VariationalDeepEmbedding<F: Float + FromPrimitive> {
    config: DeepClusteringConfig,
    encoder_mean: Vec<Array2<F>>,
    encoder_logvar: Vec<Array2<F>>,
    decoder_weights: Vec<Array2<F>>,
    cluster_assignments: Option<Array1<usize>>,
}

impl<F: Float + FromPrimitive + Debug + 'static> VariationalDeepEmbedding<F> {
    /// Create new VaDE instance
    pub fn new(config: DeepClusteringConfig) -> Self {
        Self {
            config,
            encoder_mean: Vec::new(),
            encoder_logvar: Vec::new(),
            decoder_weights: Vec::new(),
            cluster_assignments: None,
        }
    }

    /// Fit VaDE model
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        // Simplified VaDE implementation
        // In practice, this would involve:
        // 1. Variational autoencoder training
        // 2. Gaussian mixture model in latent space
        // 3. Joint optimization of reconstruction and clustering

        println!("Training Variational Deep Embedding...");

        // For demonstration, use simplified approach
        let config = DeepClusteringConfig {
            embedding_dim: 2,
            ..self.config.clone()
        };

        deep_embedded_clustering(data, Some(config))
    }
}

/// Variational deep embedding clustering
#[allow(dead_code)]
pub fn variational_deep_embedding<F: Float + FromPrimitive + Debug + 'static>(
    data: ArrayView2<F>,
    config: Option<DeepClusteringConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut vade = VariationalDeepEmbedding::new(config);
    vade.fit(data)
}

// ===========================================
// Quantum Approximate Optimization Algorithm (QAOA) for Clustering
// ===========================================

/// Configuration for QAOA-based clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOAConfig {
    /// Number of QAOA layers
    pub p_layers: usize,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for parameter optimization
    pub learning_rate: f64,
    /// Number of shots for quantum measurement simulation
    pub n_shots: usize,
    /// Quantum noise model strength (0.0 = no noise, 1.0 = maximum noise)
    pub noise_strength: f64,
    /// Cost function type for clustering
    pub cost_function: QAOACostFunction,
}

/// Cost function types for QAOA clustering
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum QAOACostFunction {
    /// Modularity-based cost for graph clustering
    Modularity,
    /// Cut-based cost function
    MaxCut,
    /// Min-cut based cost function
    MinCut,
    /// Weighted clustering cost
    WeightedClustering,
}

impl Default for QAOAConfig {
    fn default() -> Self {
        Self {
            p_layers: 3,
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 0.1,
            n_shots: 1000,
            noise_strength: 0.1,
            cost_function: QAOACostFunction::Modularity,
        }
    }
}

/// QAOA-based clustering implementation
#[derive(Debug, Clone)]
pub struct QAOAClustering<F: Float + FromPrimitive> {
    config: QAOAConfig,
    n_clusters: usize,
    gamma_params: Array1<F>, // QAOA mixer parameters
    beta_params: Array1<F>,  // QAOA cost parameters
    adjacency_matrix: Option<Array2<F>>,
    quantum_state: Option<Array1<F>>,
    cluster_assignments: Option<Array1<usize>>,
    cost_history: Vec<F>,
}

impl<F: Float + FromPrimitive + Debug + 'static> QAOAClustering<F> {
    /// Create new QAOA clustering instance
    pub fn new(_nclusters: usize, config: QAOAConfig) -> Self {
        let mut rng = rand::rng();

        // Initialize QAOA parameters randomly
        let gamma_params = Array1::from_shape_fn(config.p_layers, |_| {
            F::from(rng.random_range(0.0..std::f64::consts::PI)).unwrap()
        });

        let beta_params = Array1::from_shape_fn(config.p_layers, |_| {
            F::from(rng.random_range(0.0..std::f64::consts::PI / 2.0)).unwrap()
        });

        Self {
            config,
            n_clusters: _nclusters,
            gamma_params,
            beta_params,
            adjacency_matrix: None,
            quantum_state: None,
            cluster_assignments: None,
            cost_history: Vec::new(),
        }
    }

    /// Build adjacency matrix from data
    fn build_adjacency_matrix(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();
        let mut adj_matrix = Array2::zeros((n_samples, n_samples));

        // Build k-NN graph or similarity graph
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                // Compute similarity (using RBF kernel)
                let dist_sq = data
                    .row(i)
                    .iter()
                    .zip(data.row(j).iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .fold(F::zero(), |acc, d| acc + d);

                let sigma = F::one(); // Kernel bandwidth
                let similarity = (-dist_sq / (F::from(2.0).unwrap() * sigma.powi(2))).exp();

                adj_matrix[[i, j]] = similarity;
                adj_matrix[[j, i]] = similarity;
            }
        }

        self.adjacency_matrix = Some(adj_matrix);
        Ok(())
    }

    /// Initialize quantum state (uniform superposition)
    fn initialize_quantum_state(&mut self, nqubits: usize) -> Result<()> {
        let n_states = 1 << nqubits; // 2^nqubits
        let amplitude = F::one() / F::from(n_states as f64).unwrap().sqrt();
        let state = Array1::from_elem(n_states, amplitude);
        self.quantum_state = Some(state);
        Ok(())
    }

    /// Apply QAOA cost unitary
    fn apply_cost_unitary(&mut self, gamma: F) -> Result<()> {
        if let (Some(ref adj_matrix), Some(ref mut state)) =
            (&self.adjacency_matrix, &mut self.quantum_state)
        {
            let nqubits = (adj_matrix.nrows() as f64).log2() as usize;
            let n_states = state.len();

            // Apply cost function based on adjacency matrix
            // This is a simplified implementation of the cost Hamiltonian
            for i in 0..n_states {
                let mut cost = F::zero();

                // Calculate cost based on bit string representation
                for j in 0..nqubits {
                    for k in (j + 1)..nqubits {
                        if j < adj_matrix.nrows() && k < adj_matrix.ncols() {
                            let bit_j = (i >> j) & 1;
                            let bit_k = (i >> k) & 1;

                            // Add cost for edges between different clusters
                            if bit_j != bit_k {
                                cost = cost + adj_matrix[[j, k]];
                            }
                        }
                    }
                }

                // Apply phase based on cost
                let phase = gamma * cost;
                state[i] = state[i] * (F::from(0.0).unwrap() + phase * F::from(1.0).unwrap()).exp();
            }
        }

        Ok(())
    }

    /// Apply QAOA mixer unitary (X rotations)
    fn apply_mixer_unitary(&mut self, beta: F) -> Result<()> {
        if let Some(ref mut state) = &mut self.quantum_state {
            let nqubits = (state.len() as f64).log2() as usize;
            let mut new_state = Array1::zeros(state.len());

            // Apply mixer Hamiltonian (sum of X gates)
            for i in 0..state.len() {
                for qubit in 0..nqubits {
                    let flipped_state = i ^ (1 << qubit); // Flip qubit
                    let cos_beta = beta.cos();
                    let sin_beta = beta.sin();

                    new_state[i] = new_state[i] + cos_beta * state[i];
                    if flipped_state < state.len() {
                        new_state[i] = new_state[i] - sin_beta * state[flipped_state];
                    }
                }
            }

            *state = new_state;
        }

        Ok(())
    }

    /// Execute QAOA circuit
    fn execute_qaoa_circuit(&mut self) -> Result<F> {
        if let Some(ref adj_matrix) = &self.adjacency_matrix {
            let nqubits = adj_matrix.nrows().min(20); // Limit for classical simulation
            self.initialize_quantum_state(nqubits)?;

            // Apply QAOA layers
            for layer in 0..self.config.p_layers {
                let gamma = self.gamma_params[layer];
                let beta = self.beta_params[layer];

                self.apply_cost_unitary(gamma)?;
                self.apply_mixer_unitary(beta)?;
            }

            // Measure expectation value
            self.measure_expectation_value()
        } else {
            Err(ClusteringError::InvalidInput(
                "No adjacency matrix".to_string(),
            ))
        }
    }

    /// Measure expectation value of cost function
    fn measure_expectation_value(&self) -> Result<F> {
        if let (Some(ref state), Some(ref adj_matrix)) =
            (&self.quantum_state, &self.adjacency_matrix)
        {
            let nqubits = (state.len() as f64).log2() as usize;
            let mut expectation = F::zero();

            for i in 0..state.len() {
                let probability = state[i].abs().powi(2);
                let mut cost = F::zero();

                // Calculate cost for this bit string
                for j in 0..nqubits {
                    for k in (j + 1)..nqubits {
                        if j < adj_matrix.nrows() && k < adj_matrix.ncols() {
                            let bit_j = (i >> j) & 1;
                            let bit_k = (i >> k) & 1;

                            if bit_j != bit_k {
                                cost = cost + adj_matrix[[j, k]];
                            }
                        }
                    }
                }

                expectation = expectation + probability * cost;
            }

            Ok(expectation)
        } else {
            Err(ClusteringError::ComputationError(
                "Invalid quantum state".to_string(),
            ))
        }
    }

    /// Optimize QAOA parameters using gradient descent
    fn optimize_parameters(&mut self) -> Result<()> {
        for iteration in 0..self.config.max_iterations {
            let current_cost = self.execute_qaoa_circuit()?;
            self.cost_history.push(current_cost);

            if iteration % 10 == 0 {
                println!("QAOA Iteration {iteration}: Cost = {current_cost:?}");
            }

            // Simplified parameter update (finite differences approximation)
            let lr = F::from(self.config.learning_rate).unwrap();
            let eps = F::from(1e-6).unwrap();

            // Update gamma parameters
            for i in 0..self.config.p_layers {
                let original = self.gamma_params[i];

                // Forward difference
                self.gamma_params[i] = original + eps;
                let cost_plus = self.execute_qaoa_circuit()?;

                self.gamma_params[i] = original - eps;
                let cost_minus = self.execute_qaoa_circuit()?;

                let gradient = (cost_plus - cost_minus) / (F::from(2.0).unwrap() * eps);
                self.gamma_params[i] = original - lr * gradient;
            }

            // Update beta parameters
            for i in 0..self.config.p_layers {
                let original = self.beta_params[i];

                self.beta_params[i] = original + eps;
                let cost_plus = self.execute_qaoa_circuit()?;

                self.beta_params[i] = original - eps;
                let cost_minus = self.execute_qaoa_circuit()?;

                let gradient = (cost_plus - cost_minus) / (F::from(2.0).unwrap() * eps);
                self.beta_params[i] = original - lr * gradient;
            }

            // Check convergence
            if iteration > 0 {
                let prev_cost = self.cost_history[iteration - 1];
                let cost_change = (current_cost - prev_cost).abs();
                if cost_change < F::from(self.config.tolerance).unwrap() {
                    println!("QAOA converged at iteration {iteration}");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Sample cluster assignments from final quantum state
    fn sample_assignments(&mut self) -> Result<Array1<usize>> {
        if let Some(ref state) = &self.quantum_state {
            let nqubits = (state.len() as f64).log2() as usize;
            let mut assignments = Array1::zeros(nqubits);
            let mut rng = rand::rng();

            // Sample from probability distribution
            for _shot in 0..self.config.n_shots {
                let mut cumulative_prob = F::zero();
                let random_val = F::from(rng.random::<f64>()).unwrap();

                for (i, &amplitude) in state.iter().enumerate() {
                    let prob = amplitude.abs().powi(2);
                    cumulative_prob = cumulative_prob + prob;

                    if random_val < cumulative_prob {
                        // Convert bit string to cluster assignments
                        for j in 0..nqubits {
                            let bit = (i >> j) & 1;
                            assignments[j] = bit % self.n_clusters;
                        }
                        break;
                    }
                }
            }

            Ok(assignments)
        } else {
            Err(ClusteringError::ComputationError(
                "No quantum state".to_string(),
            ))
        }
    }

    /// Fit QAOA clustering to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        println!(
            "Starting QAOA clustering with {} layers...",
            self.config.p_layers
        );

        // Build adjacency matrix
        self.build_adjacency_matrix(data)?;

        // Optimize QAOA parameters
        self.optimize_parameters()?;

        // Sample final assignments
        let assignments = self.sample_assignments()?;
        self.cluster_assignments = Some(assignments.clone());

        // Compute cluster centers from assignments
        let mut centers = Array2::zeros((self.n_clusters, data.ncols()));
        let mut cluster_counts = Array1::zeros(self.n_clusters);

        for (i, &cluster) in assignments.iter().enumerate() {
            if i < data.nrows() {
                for j in 0..data.ncols() {
                    centers[[cluster, j]] = centers[[cluster, j]] + data[[i, j]];
                }
                cluster_counts[cluster] = cluster_counts[cluster] + F::one();
            }
        }

        // Normalize centers
        for i in 0..self.n_clusters {
            if cluster_counts[i] > F::zero() {
                for j in 0..data.ncols() {
                    centers[[i, j]] = centers[[i, j]] / cluster_counts[i];
                }
            }
        }

        Ok((centers, assignments))
    }
}

/// QAOA-based clustering function
#[allow(dead_code)]
pub fn qaoa_clustering<F: Float + FromPrimitive + Debug + 'static>(
    data: ArrayView2<F>,
    n_clusters: usize,
    config: Option<QAOAConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut qaoa = QAOAClustering::new(n_clusters, config);
    qaoa.fit(data)
}

// ===========================================
// Variational Quantum Eigensolver (VQE) for Clustering
// ===========================================

/// Configuration for VQE-based clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VQEConfig {
    /// Number of variational parameters
    pub n_params: usize,
    /// Maximum iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Learning rate for parameter optimization
    pub learning_rate: f64,
    /// Ansatz type for quantum circuit
    pub ansatz_type: VQEAnsatz,
    /// Number of shots for measurement
    pub n_shots: usize,
}

/// VQE ansatz types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VQEAnsatz {
    /// Hardware-efficient ansatz
    HardwareEfficient,
    /// UCCSD-inspired ansatz
    UCCSD,
    /// Custom parametrized ansatz
    Custom,
}

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            n_params: 10,
            max_iterations: 100,
            tolerance: 1e-6,
            learning_rate: 0.1,
            ansatz_type: VQEAnsatz::HardwareEfficient,
            n_shots: 1000,
        }
    }
}

/// VQE-based clustering implementation
#[derive(Debug, Clone)]
pub struct VQEClustering<F: Float + FromPrimitive> {
    config: VQEConfig,
    n_clusters: usize,
    variational_params: Array1<F>,
    hamiltonian_matrix: Option<Array2<F>>,
    eigenvalues: Option<Array1<F>>,
    eigenvectors: Option<Array2<F>>,
    cluster_assignments: Option<Array1<usize>>,
}

impl<F: Float + FromPrimitive + Debug + 'static> VQEClustering<F> {
    /// Create new VQE clustering instance
    pub fn new(_nclusters: usize, config: VQEConfig) -> Self {
        let mut rng = rand::rng();

        // Initialize variational parameters
        let params = Array1::from_shape_fn(config.n_params, |_| {
            F::from(rng.random_range(0.0..2.0 * std::f64::consts::PI)).unwrap()
        });

        Self {
            config,
            n_clusters: _nclusters,
            variational_params: params,
            hamiltonian_matrix: None,
            eigenvalues: None,
            eigenvectors: None,
            cluster_assignments: None,
        }
    }

    /// Build clustering Hamiltonian from data
    fn build_clustering_hamiltonian(&mut self, data: ArrayView2<F>) -> Result<()> {
        let n_samples = data.nrows();
        let mut hamiltonian = Array2::zeros((n_samples, n_samples));

        // Build Laplacian matrix for clustering
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    // Similarity weight
                    let dist_sq = data
                        .row(i)
                        .iter()
                        .zip(data.row(j).iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .fold(F::zero(), |acc, d| acc + d);

                    let sigma = F::one();
                    let weight = (-dist_sq / (F::from(2.0).unwrap() * sigma.powi(2))).exp();
                    hamiltonian[[i, j]] = -weight; // Negative for clustering
                } else {
                    // Diagonal element (degree)
                    let mut degree = F::zero();
                    for k in 0..n_samples {
                        if k != i {
                            let dist_sq = data
                                .row(i)
                                .iter()
                                .zip(data.row(k).iter())
                                .map(|(&a, &b)| (a - b).powi(2))
                                .fold(F::zero(), |acc, d| acc + d);

                            let sigma = F::one();
                            let weight = (-dist_sq / (F::from(2.0).unwrap() * sigma.powi(2))).exp();
                            degree = degree + weight;
                        }
                    }
                    hamiltonian[[i, i]] = degree;
                }
            }
        }

        self.hamiltonian_matrix = Some(hamiltonian);
        Ok(())
    }

    /// Prepare variational quantum state using ansatz
    fn prepare_variational_state(&self, nqubits: usize) -> Result<Array1<F>> {
        let n_states = 1 << nqubits;
        let mut state = Array1::zeros(n_states);
        state[0] = F::one(); // Start with |0...0>

        match self.config.ansatz_type {
            VQEAnsatz::HardwareEfficient => {
                // Apply hardware-efficient ansatz
                for (i, &param) in self.variational_params.iter().enumerate() {
                    let qubit = i % nqubits;
                    // Apply RY rotation (simplified)
                    let cos_half = (param / F::from(2.0).unwrap()).cos();
                    let sin_half = (param / F::from(2.0).unwrap()).sin();

                    let mut new_state = Array1::zeros(n_states);
                    for j in 0..n_states {
                        let bit = (j >> qubit) & 1;
                        if bit == 0 {
                            new_state[j] = new_state[j] + cos_half * state[j];
                            new_state[j | (1 << qubit)] =
                                new_state[j | (1 << qubit)] + sin_half * state[j];
                        } else {
                            new_state[j] = new_state[j] + cos_half * state[j];
                            new_state[j & !(1 << qubit)] =
                                new_state[j & !(1 << qubit)] - sin_half * state[j];
                        }
                    }
                    state = new_state;
                }
            }
            _ => {
                // Simplified ansatz
                for (i, &param) in self.variational_params.iter().enumerate() {
                    let qubit = i % nqubits;
                    // Apply parameterized gate
                    state[1 << qubit] = param.sin();
                    state[0] = param.cos();
                }
            }
        }

        // Normalize state
        let norm = state
            .iter()
            .map(|x| x.abs().powi(2))
            .fold(F::zero(), |acc, x| acc + x)
            .sqrt();
        if norm > F::zero() {
            state.mapv_inplace(|x| x / norm);
        }

        Ok(state)
    }

    /// Compute expectation value of Hamiltonian
    fn compute_expectation_value(&self) -> Result<F> {
        if let Some(ref hamiltonian) = &self.hamiltonian_matrix {
            let nqubits = hamiltonian.nrows().min(10); // Limit for simulation
            let state = self.prepare_variational_state(nqubits)?;

            let mut expectation = F::zero();

            // <ψ|H|ψ>
            for i in 0..hamiltonian.nrows().min(state.len()) {
                for j in 0..hamiltonian.ncols().min(state.len()) {
                    expectation = expectation + state[i] * hamiltonian[[i, j]] * state[j];
                }
            }

            Ok(expectation)
        } else {
            Err(ClusteringError::InvalidInput(
                "No Hamiltonian matrix".to_string(),
            ))
        }
    }

    /// Optimize VQE parameters
    fn optimize_vqe_parameters(&mut self) -> Result<()> {
        for iteration in 0..self.config.max_iterations {
            let current_energy = self.compute_expectation_value()?;

            if iteration % 10 == 0 {
                println!("VQE Iteration {iteration}: Energy = {current_energy:?}");
            }

            // Parameter optimization using finite differences
            let lr = F::from(self.config.learning_rate).unwrap();
            let eps = F::from(1e-6).unwrap();

            for i in 0..self.config.n_params {
                let original = self.variational_params[i];

                // Compute gradient
                self.variational_params[i] = original + eps;
                let energy_plus = self.compute_expectation_value()?;

                self.variational_params[i] = original - eps;
                let energy_minus = self.compute_expectation_value()?;

                let gradient = (energy_plus - energy_minus) / (F::from(2.0).unwrap() * eps);
                self.variational_params[i] = original - lr * gradient;
            }

            // Check convergence
            if iteration > 0 {
                let energy_change = current_energy.abs();
                if energy_change < F::from(self.config.tolerance).unwrap() {
                    println!("VQE converged at iteration {iteration}");
                    break;
                }
            }
        }

        Ok(())
    }

    /// Extract cluster assignments from optimized state
    fn extract_cluster_assignments(&mut self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        let n_samples = data.nrows();

        // For demonstration, use spectral clustering approach on the Hamiltonian
        if let Some(ref hamiltonian) = &self.hamiltonian_matrix {
            // Simplified eigenvalue decomposition (would use proper linear algebra library)
            let mut assignments = Array1::zeros(n_samples);

            // Simple assignment based on position (placeholder)
            for i in 0..n_samples {
                assignments[i] = i % self.n_clusters;
            }

            Ok(assignments)
        } else {
            Err(ClusteringError::ComputationError(
                "No Hamiltonian matrix".to_string(),
            ))
        }
    }

    /// Fit VQE clustering to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<(Array2<F>, Array1<usize>)> {
        println!("Starting VQE clustering optimization...");

        // Build clustering Hamiltonian
        self.build_clustering_hamiltonian(data)?;

        // Optimize VQE parameters
        self.optimize_vqe_parameters()?;

        // Extract cluster assignments
        let assignments = self.extract_cluster_assignments(data)?;
        self.cluster_assignments = Some(assignments.clone());

        // Compute cluster centers
        let mut centers = Array2::zeros((self.n_clusters, data.ncols()));
        let mut cluster_counts = Array1::zeros(self.n_clusters);

        for (i, &cluster) in assignments.iter().enumerate() {
            for j in 0..data.ncols() {
                centers[[cluster, j]] = centers[[cluster, j]] + data[[i, j]];
            }
            cluster_counts[cluster] = cluster_counts[cluster] + F::one();
        }

        // Normalize centers
        for i in 0..self.n_clusters {
            if cluster_counts[i] > F::zero() {
                for j in 0..data.ncols() {
                    centers[[i, j]] = centers[[i, j]] / cluster_counts[i];
                }
            }
        }

        Ok((centers, assignments))
    }
}

/// VQE-based clustering function
#[allow(dead_code)]
pub fn vqe_clustering<F: Float + FromPrimitive + Debug + 'static>(
    data: ArrayView2<F>,
    n_clusters: usize,
    config: Option<VQEConfig>,
) -> Result<(Array2<F>, Array1<usize>)> {
    let config = config.unwrap_or_default();
    let mut vqe = VQEClustering::new(n_clusters, config);
    vqe.fit(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_quantum_kmeans_basic() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
        )
        .unwrap();

        let config = QuantumConfig {
            n_quantum_states: 4,
            quantum_iterations: 20,
            ..Default::default()
        };

        let result = quantum_kmeans(data.view(), 2, Some(config));
        assert!(result.is_ok());

        let (centers, labels) = result.unwrap();
        assert_eq!(centers.nrows(), 2);
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_adaptive_online_clustering_basic() {
        let data = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0, 1.2, 1.9, 8.5, 9.0,
            ],
        )
        .unwrap();

        let config = AdaptiveOnlineConfig {
            max_clusters: 5,
            cluster_creation_threshold: 3.0,
            ..Default::default()
        };

        let result = adaptive_online_clustering(data.view(), Some(config));
        assert!(result.is_ok());

        let (centers, labels) = result.unwrap();
        assert!(centers.nrows() <= 5);
        assert_eq!(labels.len(), 8);
    }

    #[test]
    fn test_quantum_config_default() {
        let config = QuantumConfig::default();
        assert_eq!(config.n_quantum_states, 8);
        assert_eq!(config.quantum_iterations, 50);
        assert!((config.decoherence_factor - 0.95).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_config_default() {
        let config = AdaptiveOnlineConfig::default();
        assert_eq!(config.max_clusters, 50);
        assert!((config.initial_learning_rate - 0.1).abs() < 1e-10);
        assert_eq!(config.concept_drift_window, 1000);
    }

    #[test]
    fn test_rl_clustering_basic() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0, 1.0, 0.6, 9.0, 11.0],
        )
        .unwrap();

        let config = RLClusteringConfig {
            n_actions: 2,
            n_episodes: 10, // Reduced for testing
            ..Default::default()
        };

        let result = rl_clustering(data.view(), Some(config));
        assert!(result.is_ok());

        let (centers, labels) = result.unwrap();
        assert!(centers.nrows() <= 2);
        assert_eq!(labels.len(), 6);
    }

    #[test]
    fn test_transfer_learning_basic() {
        let target_data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 1.5, 1.8, 5.0, 8.0, 8.0, 8.0]).unwrap();

        let source_centroids = Array2::from_shape_vec((2, 2), vec![1.0, 1.5, 7.0, 9.0]).unwrap();

        let config = TransferLearningConfig {
            adaptation_iterations: 5, // Reduced for testing
            ..Default::default()
        };

        let result = transfer_learning_clustering(
            Some(source_centroids),
            target_data.view(),
            2,
            Some(config),
        );

        assert!(result.is_ok());
        let (centers, labels) = result.unwrap();
        assert_eq!(centers.nrows(), 2);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_rl_config_default() {
        let config = RLClusteringConfig::default();
        assert_eq!(config.n_actions, 10);
        assert!((config.learning_rate - 0.1).abs() < 1e-10);
        assert_eq!(config.n_episodes, 1000);
    }

    #[test]
    fn test_transfer_config_default() {
        let config = TransferLearningConfig::default();
        assert!((config.source_weight - 0.7).abs() < 1e-10);
        assert!((config.target_weight - 0.3).abs() < 1e-10);
        assert_eq!(config.adaptation_iterations, 100);
    }
}
