//! Time series clustering and classification algorithms
//!
//! This module provides various methods for clustering and classifying time series:
//! - Time series clustering algorithms (k-means, hierarchical, DBSCAN)
//! - Distance measures for time series (DTW, Euclidean, correlation-based)
//! - Time series classification methods (1-NN DTW, shapelet-based, feature-based)
//! - Shape-based clustering and classification

use crate::correlation::{CorrelationAnalyzer, DTWConfig};
use crate::error::TimeSeriesError;
use ndarray::{s, Array1, Array2};
use std::collections::{HashMap, HashSet};

/// Result type for clustering and classification
pub type ClusteringResult<T> = Result<T, TimeSeriesError>;

/// Clustering result
#[derive(Debug, Clone)]
pub struct TimeSeriesClusteringResult {
    /// Cluster assignments for each time series
    pub cluster_labels: Array1<usize>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Cluster centers (centroids)
    pub centroids: Vec<Array1<f64>>,
    /// Inertia (sum of squared distances to centroids)
    pub inertia: f64,
    /// Silhouette coefficient for clustering quality
    pub silhouette_score: f64,
    /// Distance matrix between time series
    pub distance_matrix: Array2<f64>,
    /// Algorithm used for clustering
    pub algorithm: ClusteringAlgorithm,
}

/// Classification result
#[derive(Debug, Clone)]
pub struct TimeSeriesClassificationResult {
    /// Predicted class labels
    pub predicted_labels: Array1<usize>,
    /// Prediction probabilities (if available)
    pub probabilities: Option<Array2<f64>>,
    /// Confidence scores
    pub confidence_scores: Array1<f64>,
    /// Distance/similarity to nearest neighbors
    pub neighbor_distances: Option<Array2<f64>>,
    /// Classification algorithm used
    pub algorithm: ClassificationAlgorithm,
    /// Number of classes
    pub n_classes: usize,
}

/// Shapelet discovery result
#[derive(Debug, Clone)]
pub struct ShapeletResult {
    /// Discovered shapelets
    pub shapelets: Vec<Shapelet>,
    /// Information gain for each shapelet
    pub information_gains: Array1<f64>,
    /// Shapelet discovery algorithm used
    pub algorithm: ShapeletAlgorithm,
    /// Time series used for discovery
    pub n_series: usize,
}

/// Individual shapelet
#[derive(Debug, Clone)]
pub struct Shapelet {
    /// Shapelet subsequence
    pub data: Array1<f64>,
    /// Source series index
    pub series_index: usize,
    /// Starting position in source series
    pub start_position: usize,
    /// Length of shapelet
    pub length: usize,
    /// Information gain
    pub information_gain: f64,
    /// Quality score
    pub quality: f64,
}

/// Clustering algorithms
#[derive(Debug, Clone, Copy)]
pub enum ClusteringAlgorithm {
    /// K-means clustering
    KMeans,
    /// Hierarchical clustering
    Hierarchical,
    /// DBSCAN density-based clustering
    DBSCAN,
    /// Spectral clustering
    Spectral,
    /// Gaussian mixture models
    GMM,
}

/// Classification algorithms
#[derive(Debug, Clone, Copy)]
pub enum ClassificationAlgorithm {
    /// k-Nearest Neighbors with DTW
    KnnDTW,
    /// Shapelet-based classification
    Shapelet,
    /// Feature-based classification
    FeatureBased,
    /// Ensemble methods
    Ensemble,
}

/// Distance measures for time series
#[derive(Debug, Clone, Copy)]
pub enum TimeSeriesDistance {
    /// Euclidean distance
    Euclidean,
    /// Dynamic Time Warping
    DTW,
    /// Correlation-based distance
    Correlation,
    /// Cosine distance
    Cosine,
    /// Manhattan distance
    Manhattan,
    /// Minkowski distance
    Minkowski(f64),
}

/// Shapelet discovery algorithms
#[derive(Debug, Clone, Copy)]
pub enum ShapeletAlgorithm {
    /// Brute force search
    BruteForce,
    /// Fast shapelet discovery
    Fast,
    /// Learning shapelets
    Learning,
}

/// Configuration for k-means clustering
#[derive(Debug, Clone)]
pub struct KMeansConfig {
    /// Number of clusters
    pub n_clusters: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of random initializations
    pub n_init: usize,
    /// Distance measure
    pub distance: TimeSeriesDistance,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            max_iterations: 300,
            tolerance: 1e-4,
            n_init: 10,
            distance: TimeSeriesDistance::Euclidean,
            random_seed: None,
        }
    }
}

/// Configuration for hierarchical clustering
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Number of clusters (if None, use all levels)
    pub n_clusters: Option<usize>,
    /// Linkage method
    pub linkage: LinkageMethod,
    /// Distance measure
    pub distance: TimeSeriesDistance,
    /// Distance threshold for automatic cluster determination
    pub distance_threshold: Option<f64>,
}

/// Linkage methods for hierarchical clustering
#[derive(Debug, Clone, Copy)]
pub enum LinkageMethod {
    /// Single linkage (minimum distance)
    Single,
    /// Complete linkage (maximum distance)
    Complete,
    /// Average linkage
    Average,
    /// Ward linkage (minimizes within-cluster variance)
    Ward,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            n_clusters: Some(3),
            linkage: LinkageMethod::Average,
            distance: TimeSeriesDistance::Euclidean,
            distance_threshold: None,
        }
    }
}

/// Configuration for DBSCAN clustering
#[derive(Debug, Clone)]
pub struct DBSCANConfig {
    /// Epsilon (neighborhood radius)
    pub eps: f64,
    /// Minimum number of points for core point
    pub min_samples: usize,
    /// Distance measure
    pub distance: TimeSeriesDistance,
}

impl Default for DBSCANConfig {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            distance: TimeSeriesDistance::Euclidean,
        }
    }
}

/// Configuration for k-NN classification
#[derive(Debug, Clone)]
pub struct KNNConfig {
    /// Number of neighbors
    pub k: usize,
    /// Distance measure
    pub distance: TimeSeriesDistance,
    /// Weighting scheme
    pub weights: WeightingScheme,
    /// DTW configuration (if using DTW distance)
    pub dtw_config: Option<DTWConfig>,
}

/// Weighting schemes for k-NN
#[derive(Debug, Clone, Copy)]
pub enum WeightingScheme {
    /// Uniform weights
    Uniform,
    /// Distance-based weights
    Distance,
    /// Exponential weights
    Exponential,
}

impl Default for KNNConfig {
    fn default() -> Self {
        Self {
            k: 5,
            distance: TimeSeriesDistance::DTW,
            weights: WeightingScheme::Distance,
            dtw_config: Some(DTWConfig::default()),
        }
    }
}

/// Configuration for shapelet discovery
#[derive(Debug, Clone)]
pub struct ShapeletConfig {
    /// Minimum shapelet length
    pub min_length: usize,
    /// Maximum shapelet length
    pub max_length: usize,
    /// Number of shapelets to discover
    pub n_shapelets: usize,
    /// Discovery algorithm
    pub algorithm: ShapeletAlgorithm,
    /// Minimum information gain threshold
    pub min_info_gain: f64,
    /// Random seed
    pub random_seed: Option<u64>,
}

impl Default for ShapeletConfig {
    fn default() -> Self {
        Self {
            min_length: 10,
            max_length: 50,
            n_shapelets: 10,
            algorithm: ShapeletAlgorithm::Fast,
            min_info_gain: 0.01,
            random_seed: None,
        }
    }
}

/// Main struct for time series clustering and classification
pub struct TimeSeriesClusterer {
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Correlation analyzer for DTW computations
    correlation_analyzer: CorrelationAnalyzer,
}

impl TimeSeriesClusterer {
    /// Create a new time series clusterer
    pub fn new() -> Self {
        Self {
            random_seed: None,
            correlation_analyzer: CorrelationAnalyzer::new(),
        }
    }

    /// Create a new clusterer with random seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            random_seed: Some(seed),
            correlation_analyzer: CorrelationAnalyzer::with_seed(seed),
        }
    }

    /// Perform k-means clustering on time series data
    ///
    /// # Arguments
    ///
    /// * `data` - Matrix where each row is a time series
    /// * `config` - Configuration for k-means clustering
    ///
    /// # Returns
    ///
    /// Result containing clustering results and statistics
    pub fn kmeans_clustering(
        &self,
        data: &Array2<f64>,
        config: &KMeansConfig,
    ) -> ClusteringResult<TimeSeriesClusteringResult> {
        let (n_series, _series_length) = data.dim();

        if n_series < config.n_clusters {
            return Err(TimeSeriesError::InvalidInput(
                "Number of time series must be at least equal to number of clusters".to_string(),
            ));
        }

        // Compute distance matrix
        let distance_matrix = self.compute_distance_matrix(data, config.distance)?;

        let mut best_result = None;
        let mut best_inertia = f64::INFINITY;

        // Try multiple random initializations
        for _init in 0..config.n_init {
            // Initialize centroids randomly
            let mut centroids = self.initialize_centroids_randomly(data, config.n_clusters)?;
            let mut cluster_labels = Array1::zeros(n_series);
            let mut prev_inertia = f64::INFINITY;

            for _iter in 0..config.max_iterations {
                // Assign points to nearest centroids
                for i in 0..n_series {
                    let mut min_distance = f64::INFINITY;
                    let mut best_cluster = 0;

                    #[allow(clippy::needless_range_loop)]
                    for k in 0..config.n_clusters {
                        let distance = self.compute_series_distance(
                            &data.row(i).to_owned(),
                            &centroids[k],
                            config.distance,
                        )?;

                        if distance < min_distance {
                            min_distance = distance;
                            best_cluster = k;
                        }
                    }

                    cluster_labels[i] = best_cluster;
                }

                // Update centroids
                #[allow(clippy::needless_range_loop)]
                for k in 0..config.n_clusters {
                    let cluster_points: Vec<usize> = cluster_labels
                        .iter()
                        .enumerate()
                        .filter(|(_, &label)| label == k)
                        .map(|(i, _)| i)
                        .collect();

                    if !cluster_points.is_empty() {
                        centroids[k] = self.compute_centroid(data, &cluster_points)?;
                    }
                }

                // Compute inertia
                let inertia =
                    self.compute_inertia(data, &centroids, &cluster_labels, config.distance)?;

                // Check convergence
                if (prev_inertia - inertia).abs() < config.tolerance {
                    break;
                }
                prev_inertia = inertia;
            }

            let final_inertia =
                self.compute_inertia(data, &centroids, &cluster_labels, config.distance)?;

            if final_inertia < best_inertia {
                best_inertia = final_inertia;

                let silhouette_score =
                    self.compute_silhouette_score(&distance_matrix, &cluster_labels)?;

                best_result = Some(TimeSeriesClusteringResult {
                    cluster_labels,
                    n_clusters: config.n_clusters,
                    centroids,
                    inertia: final_inertia,
                    silhouette_score,
                    distance_matrix: distance_matrix.clone(),
                    algorithm: ClusteringAlgorithm::KMeans,
                });
            }
        }

        best_result.ok_or_else(|| {
            TimeSeriesError::ComputationError("K-means clustering failed".to_string())
        })
    }

    /// Perform hierarchical clustering on time series data
    ///
    /// # Arguments
    ///
    /// * `data` - Matrix where each row is a time series
    /// * `config` - Configuration for hierarchical clustering
    ///
    /// # Returns
    ///
    /// Result containing clustering results and dendrogram information
    pub fn hierarchical_clustering(
        &self,
        data: &Array2<f64>,
        config: &HierarchicalConfig,
    ) -> ClusteringResult<TimeSeriesClusteringResult> {
        let _n_series = data.nrows();

        // Compute distance matrix
        let distance_matrix = self.compute_distance_matrix(data, config.distance)?;

        // Perform hierarchical clustering
        let cluster_labels = self.hierarchical_clustering_impl(&distance_matrix, config)?;

        let n_clusters = cluster_labels.iter().max().unwrap_or(&0) + 1;

        // Compute centroids for each cluster
        let mut centroids = Vec::new();
        for k in 0..n_clusters {
            let cluster_points: Vec<usize> = cluster_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == k)
                .map(|(i, _)| i)
                .collect();

            if !cluster_points.is_empty() {
                centroids.push(self.compute_centroid(data, &cluster_points)?);
            } else {
                centroids.push(Array1::zeros(data.ncols()));
            }
        }

        let inertia = self.compute_inertia(data, &centroids, &cluster_labels, config.distance)?;
        let silhouette_score = self.compute_silhouette_score(&distance_matrix, &cluster_labels)?;

        Ok(TimeSeriesClusteringResult {
            cluster_labels,
            n_clusters,
            centroids,
            inertia,
            silhouette_score,
            distance_matrix,
            algorithm: ClusteringAlgorithm::Hierarchical,
        })
    }

    /// Perform DBSCAN clustering on time series data
    ///
    /// # Arguments
    ///
    /// * `data` - Matrix where each row is a time series
    /// * `config` - Configuration for DBSCAN clustering
    ///
    /// # Returns
    ///
    /// Result containing clustering results (noise points labeled as -1)
    pub fn dbscan_clustering(
        &self,
        data: &Array2<f64>,
        config: &DBSCANConfig,
    ) -> ClusteringResult<TimeSeriesClusteringResult> {
        let n_series = data.nrows();

        // Compute distance matrix
        let distance_matrix = self.compute_distance_matrix(data, config.distance)?;

        // DBSCAN algorithm
        let mut cluster_labels = Array1::from_elem(n_series, usize::MAX); // MAX = unvisited
        let mut cluster_id = 0;

        for i in 0..n_series {
            if cluster_labels[i] != usize::MAX {
                continue; // Already processed
            }

            let neighbors = self.get_neighbors(&distance_matrix, i, config.eps);

            if neighbors.len() < config.min_samples {
                cluster_labels[i] = usize::MAX - 1; // Mark as noise (will be converted to special value)
            } else {
                self.expand_cluster(
                    &distance_matrix,
                    &mut cluster_labels,
                    i,
                    &neighbors,
                    cluster_id,
                    config,
                )?;
                cluster_id += 1;
            }
        }

        // Convert noise points and renumber clusters
        let mut final_labels = Array1::zeros(n_series);
        let mut cluster_map = HashMap::new();
        let mut next_cluster = 0;

        for i in 0..n_series {
            if cluster_labels[i] == usize::MAX - 1 {
                final_labels[i] = usize::MAX; // Keep as special noise marker
            } else if cluster_labels[i] != usize::MAX {
                let mapped_cluster = *cluster_map.entry(cluster_labels[i]).or_insert_with(|| {
                    let cluster = next_cluster;
                    next_cluster += 1;
                    cluster
                });
                final_labels[i] = mapped_cluster;
            }
        }

        let n_clusters = next_cluster;

        // Compute centroids (excluding noise points)
        let mut centroids = Vec::new();
        for k in 0..n_clusters {
            let cluster_points: Vec<usize> = final_labels
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == k)
                .map(|(i, _)| i)
                .collect();

            if !cluster_points.is_empty() {
                centroids.push(self.compute_centroid(data, &cluster_points)?);
            } else {
                centroids.push(Array1::zeros(data.ncols()));
            }
        }

        // For inertia and silhouette calculation, treat noise as separate points
        let labels_for_metrics =
            final_labels.mapv(|x| if x == usize::MAX { usize::MAX } else { x });

        let inertia = if n_clusters > 0 {
            self.compute_inertia_with_noise(data, &centroids, &labels_for_metrics, config.distance)?
        } else {
            0.0
        };

        let silhouette_score = if n_clusters > 1 {
            self.compute_silhouette_score_with_noise(&distance_matrix, &labels_for_metrics)?
        } else {
            0.0
        };

        Ok(TimeSeriesClusteringResult {
            cluster_labels: labels_for_metrics,
            n_clusters,
            centroids,
            inertia,
            silhouette_score,
            distance_matrix,
            algorithm: ClusteringAlgorithm::DBSCAN,
        })
    }

    /// Perform k-NN classification on time series data
    ///
    /// # Arguments
    ///
    /// * `train_data` - Training time series matrix
    /// * `train_labels` - Training labels
    /// * `test_data` - Test time series matrix
    /// * `config` - Configuration for k-NN classification
    ///
    /// # Returns
    ///
    /// Result containing classification predictions and confidence scores
    pub fn knn_classification(
        &self,
        train_data: &Array2<f64>,
        train_labels: &Array1<usize>,
        test_data: &Array2<f64>,
        config: &KNNConfig,
    ) -> ClusteringResult<TimeSeriesClassificationResult> {
        let n_train = train_data.nrows();
        let n_test = test_data.nrows();
        let n_classes = train_labels.iter().max().unwrap_or(&0) + 1;

        if train_labels.len() != n_train {
            return Err(TimeSeriesError::InvalidInput(
                "Training data and labels must have the same number of samples".to_string(),
            ));
        }

        let mut predicted_labels = Array1::zeros(n_test);
        let mut confidence_scores = Array1::zeros(n_test);
        let mut probabilities = Array2::zeros((n_test, n_classes));
        let mut neighbor_distances = Array2::zeros((n_test, config.k.min(n_train)));

        for i in 0..n_test {
            let test_series = test_data.row(i).to_owned();

            // Compute distances to all training samples
            let mut distances = Vec::new();
            for j in 0..n_train {
                let train_series = train_data.row(j).to_owned();
                let distance =
                    self.compute_series_distance(&test_series, &train_series, config.distance)?;
                distances.push((distance, train_labels[j]));
            }

            // Sort by distance and take k nearest neighbors
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let k_neighbors = distances.iter().take(config.k).cloned().collect::<Vec<_>>();

            // Store neighbor distances
            for (idx, (dist, _)) in k_neighbors.iter().enumerate() {
                if idx < neighbor_distances.ncols() {
                    neighbor_distances[[i, idx]] = *dist;
                }
            }

            // Compute weighted votes
            let mut class_votes = vec![0.0; n_classes];
            let mut total_weight = 0.0;

            for (distance, class_label) in &k_neighbors {
                let weight = match config.weights {
                    WeightingScheme::Uniform => 1.0,
                    WeightingScheme::Distance => {
                        if *distance < f64::EPSILON {
                            1e6
                        } else {
                            1.0 / distance
                        }
                    }
                    WeightingScheme::Exponential => (-distance).exp(),
                };

                class_votes[*class_label] += weight;
                total_weight += weight;
            }

            // Normalize to get probabilities
            if total_weight > 0.0 {
                for class in 0..n_classes {
                    probabilities[[i, class]] = class_votes[class] / total_weight;
                }
            }

            // Predict class with highest vote
            let predicted_class = class_votes
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predicted_labels[i] = predicted_class;
            confidence_scores[i] = class_votes[predicted_class] / total_weight;
        }

        Ok(TimeSeriesClassificationResult {
            predicted_labels,
            probabilities: Some(probabilities),
            confidence_scores,
            neighbor_distances: Some(neighbor_distances),
            algorithm: ClassificationAlgorithm::KnnDTW,
            n_classes,
        })
    }

    /// Discover shapelets in time series data
    ///
    /// # Arguments
    ///
    /// * `data` - Time series data matrix
    /// * `labels` - Class labels for supervised shapelet discovery
    /// * `config` - Configuration for shapelet discovery
    ///
    /// # Returns
    ///
    /// Result containing discovered shapelets and their quality measures
    pub fn discover_shapelets(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        config: &ShapeletConfig,
    ) -> ClusteringResult<ShapeletResult> {
        let n_series = data.nrows();
        let series_length = data.ncols();

        if labels.len() != n_series {
            return Err(TimeSeriesError::InvalidInput(
                "Data and labels must have the same number of samples".to_string(),
            ));
        }

        let mut candidate_shapelets = Vec::new();

        // Generate candidate shapelets
        for series_idx in 0..n_series {
            for length in config.min_length..=config.max_length.min(series_length) {
                for start_pos in 0..=(series_length - length) {
                    let shapelet_data = data
                        .slice(s![series_idx, start_pos..start_pos + length])
                        .to_owned();

                    let shapelet = Shapelet {
                        data: shapelet_data,
                        series_index: series_idx,
                        start_position: start_pos,
                        length,
                        information_gain: 0.0,
                        quality: 0.0,
                    };

                    candidate_shapelets.push(shapelet);
                }
            }
        }

        // Evaluate candidate shapelets
        let mut shapelet_scores = Vec::new();

        for (idx, shapelet) in candidate_shapelets.iter().enumerate() {
            let info_gain = self.compute_shapelet_information_gain(data, labels, shapelet)?;

            if info_gain >= config.min_info_gain {
                shapelet_scores.push((idx, info_gain));
            }
        }

        // Sort by information gain and select top shapelets
        shapelet_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let n_selected = config.n_shapelets.min(shapelet_scores.len());
        let mut selected_shapelets = Vec::new();
        let mut information_gains = Array1::zeros(n_selected);

        for i in 0..n_selected {
            let (shapelet_idx, info_gain) = shapelet_scores[i];
            let mut shapelet = candidate_shapelets[shapelet_idx].clone();
            shapelet.information_gain = info_gain;
            shapelet.quality = info_gain; // Simple quality measure

            selected_shapelets.push(shapelet);
            information_gains[i] = info_gain;
        }

        Ok(ShapeletResult {
            shapelets: selected_shapelets,
            information_gains,
            algorithm: config.algorithm,
            n_series,
        })
    }

    // Helper methods

    fn compute_distance_matrix(
        &self,
        data: &Array2<f64>,
        distance: TimeSeriesDistance,
    ) -> ClusteringResult<Array2<f64>> {
        let n_series = data.nrows();
        let mut distance_matrix = Array2::zeros((n_series, n_series));

        for i in 0..n_series {
            for j in i..n_series {
                let dist = if i == j {
                    0.0
                } else {
                    self.compute_series_distance(
                        &data.row(i).to_owned(),
                        &data.row(j).to_owned(),
                        distance,
                    )?
                };

                distance_matrix[[i, j]] = dist;
                distance_matrix[[j, i]] = dist;
            }
        }

        Ok(distance_matrix)
    }

    fn compute_series_distance(
        &self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        distance: TimeSeriesDistance,
    ) -> ClusteringResult<f64> {
        match distance {
            TimeSeriesDistance::Euclidean => Ok((x - y).mapv(|d| d * d).sum().sqrt()),
            TimeSeriesDistance::DTW => {
                let dtw_config = DTWConfig::default();
                let result = self
                    .correlation_analyzer
                    .dynamic_time_warping(x, y, &dtw_config)
                    .map_err(|_| {
                        TimeSeriesError::ComputationError("DTW computation failed".to_string())
                    })?;
                Ok(result.distance)
            }
            TimeSeriesDistance::Correlation => {
                let correlation = self.pearson_correlation(x, y)?;
                Ok(1.0 - correlation.abs()) // Convert to distance
            }
            TimeSeriesDistance::Cosine => {
                let dot_product = x.dot(y);
                let norm_x = (x.dot(x)).sqrt();
                let norm_y = (y.dot(y)).sqrt();

                if norm_x < f64::EPSILON || norm_y < f64::EPSILON {
                    Ok(1.0)
                } else {
                    let cosine_sim = dot_product / (norm_x * norm_y);
                    Ok(1.0 - cosine_sim.abs())
                }
            }
            TimeSeriesDistance::Manhattan => Ok((x - y).mapv(|d| d.abs()).sum()),
            TimeSeriesDistance::Minkowski(p) => {
                Ok((x - y).mapv(|d| d.abs().powf(p)).sum().powf(1.0 / p))
            }
        }
    }

    fn pearson_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> ClusteringResult<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(0.0);
        }

        let n = x.len() as f64;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let diff_x = xi - mean_x;
            let diff_y = yi - mean_y;
            numerator += diff_x * diff_y;
            sum_sq_x += diff_x * diff_x;
            sum_sq_y += diff_y * diff_y;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator < f64::EPSILON {
            return Ok(0.0);
        }

        Ok(numerator / denominator)
    }

    fn initialize_centroids_randomly(
        &self,
        data: &Array2<f64>,
        k: usize,
    ) -> ClusteringResult<Vec<Array1<f64>>> {
        let n_series = data.nrows();
        let mut centroids = Vec::new();

        // Simple random initialization - pick k random time series as initial centroids
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        if let Some(seed) = self.random_seed {
            seed.hash(&mut hasher);
        }

        let mut selected_indices = HashSet::new();

        for _i in 0..k {
            let mut idx = (hasher.finish() as usize) % n_series;
            while selected_indices.contains(&idx) {
                hasher.write_usize(idx);
                idx = (hasher.finish() as usize) % n_series;
            }
            selected_indices.insert(idx);
            centroids.push(data.row(idx).to_owned());
        }

        Ok(centroids)
    }

    fn compute_centroid(
        &self,
        data: &Array2<f64>,
        indices: &[usize],
    ) -> ClusteringResult<Array1<f64>> {
        if indices.is_empty() {
            return Ok(Array1::zeros(data.ncols()));
        }

        let mut centroid = Array1::zeros(data.ncols());

        for &idx in indices {
            centroid = centroid + data.row(idx);
        }

        centroid /= indices.len() as f64;
        Ok(centroid)
    }

    fn compute_inertia(
        &self,
        data: &Array2<f64>,
        centroids: &[Array1<f64>],
        labels: &Array1<usize>,
        distance: TimeSeriesDistance,
    ) -> ClusteringResult<f64> {
        let mut inertia = 0.0;

        for i in 0..data.nrows() {
            let cluster_id = labels[i];
            if cluster_id < centroids.len() {
                let dist = self.compute_series_distance(
                    &data.row(i).to_owned(),
                    &centroids[cluster_id],
                    distance,
                )?;
                inertia += dist * dist;
            }
        }

        Ok(inertia)
    }

    fn compute_inertia_with_noise(
        &self,
        data: &Array2<f64>,
        centroids: &[Array1<f64>],
        labels: &Array1<usize>,
        distance: TimeSeriesDistance,
    ) -> ClusteringResult<f64> {
        let mut inertia = 0.0;

        for i in 0..data.nrows() {
            let cluster_id = labels[i];
            if cluster_id != usize::MAX && cluster_id < centroids.len() {
                let dist = self.compute_series_distance(
                    &data.row(i).to_owned(),
                    &centroids[cluster_id],
                    distance,
                )?;
                inertia += dist * dist;
            }
        }

        Ok(inertia)
    }

    fn compute_silhouette_score(
        &self,
        distance_matrix: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> ClusteringResult<f64> {
        let n_samples = distance_matrix.nrows();
        let mut silhouette_scores = Vec::new();

        for i in 0..n_samples {
            let cluster_i = labels[i];

            // Compute average distance to points in same cluster (a_i)
            let same_cluster_distances: Vec<f64> = (0..n_samples)
                .filter(|&j| j != i && labels[j] == cluster_i)
                .map(|j| distance_matrix[[i, j]])
                .collect();

            let a_i = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };

            // Compute minimum average distance to points in other clusters (b_i)
            let unique_clusters: HashSet<usize> = labels.iter().cloned().collect();
            let mut min_avg_distance = f64::INFINITY;

            for &other_cluster in &unique_clusters {
                if other_cluster != cluster_i {
                    let other_cluster_distances: Vec<f64> = (0..n_samples)
                        .filter(|&j| labels[j] == other_cluster)
                        .map(|j| distance_matrix[[i, j]])
                        .collect();

                    if !other_cluster_distances.is_empty() {
                        let avg_distance = other_cluster_distances.iter().sum::<f64>()
                            / other_cluster_distances.len() as f64;
                        min_avg_distance = min_avg_distance.min(avg_distance);
                    }
                }
            }

            let b_i = min_avg_distance;

            // Compute silhouette coefficient
            let silhouette = if a_i.max(b_i) > 0.0 {
                (b_i - a_i) / a_i.max(b_i)
            } else {
                0.0
            };

            silhouette_scores.push(silhouette);
        }

        Ok(silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64)
    }

    fn compute_silhouette_score_with_noise(
        &self,
        distance_matrix: &Array2<f64>,
        labels: &Array1<usize>,
    ) -> ClusteringResult<f64> {
        // Filter out noise points (labeled as usize::MAX) for silhouette calculation
        let valid_indices: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &label)| label != usize::MAX)
            .map(|(i, _)| i)
            .collect();

        if valid_indices.len() < 2 {
            return Ok(0.0);
        }

        let filtered_labels = Array1::from_iter(valid_indices.iter().map(|&i| labels[i]));
        let mut filtered_distance_matrix =
            Array2::zeros((valid_indices.len(), valid_indices.len()));

        for (i, &idx_i) in valid_indices.iter().enumerate() {
            for (j, &idx_j) in valid_indices.iter().enumerate() {
                filtered_distance_matrix[[i, j]] = distance_matrix[[idx_i, idx_j]];
            }
        }

        self.compute_silhouette_score(&filtered_distance_matrix, &filtered_labels)
    }

    fn hierarchical_clustering_impl(
        &self,
        distance_matrix: &Array2<f64>,
        config: &HierarchicalConfig,
    ) -> ClusteringResult<Array1<usize>> {
        let n_samples = distance_matrix.nrows();
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();
        let cluster_distances = distance_matrix.clone();

        let target_clusters = config.n_clusters.unwrap_or(1);

        while clusters.len() > target_clusters {
            // Find closest pair of clusters
            let mut min_distance = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 1;

            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let distance = self.compute_cluster_distance(
                        &clusters[i],
                        &clusters[j],
                        &cluster_distances,
                        config.linkage,
                    );

                    if distance < min_distance {
                        min_distance = distance;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Check distance threshold
            if let Some(threshold) = config.distance_threshold {
                if min_distance > threshold {
                    break;
                }
            }

            // Merge clusters
            let mut new_cluster = clusters[merge_i].clone();
            new_cluster.extend(clusters[merge_j].clone());

            // Remove old clusters (remove higher index first)
            if merge_i < merge_j {
                clusters.remove(merge_j);
                clusters.remove(merge_i);
            } else {
                clusters.remove(merge_i);
                clusters.remove(merge_j);
            }

            clusters.push(new_cluster);
        }

        // Create labels array
        let mut labels = Array1::zeros(n_samples);
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &sample_id in cluster {
                labels[sample_id] = cluster_id;
            }
        }

        Ok(labels)
    }

    #[allow(clippy::only_used_in_recursion)]
    fn compute_cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        distance_matrix: &Array2<f64>,
        linkage: LinkageMethod,
    ) -> f64 {
        match linkage {
            LinkageMethod::Single => {
                // Minimum distance between any two points
                let mut min_dist = f64::INFINITY;
                for &i in cluster1 {
                    for &j in cluster2 {
                        min_dist = min_dist.min(distance_matrix[[i, j]]);
                    }
                }
                min_dist
            }
            LinkageMethod::Complete => {
                // Maximum distance between any two points
                let mut max_dist: f64 = 0.0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        max_dist = max_dist.max(distance_matrix[[i, j]]);
                    }
                }
                max_dist
            }
            LinkageMethod::Average => {
                // Average distance between all pairs
                let mut sum_dist = 0.0;
                let mut count = 0;
                for &i in cluster1 {
                    for &j in cluster2 {
                        sum_dist += distance_matrix[[i, j]];
                        count += 1;
                    }
                }
                if count > 0 {
                    sum_dist / count as f64
                } else {
                    0.0
                }
            }
            LinkageMethod::Ward => {
                // Ward linkage - simplified implementation
                // This would require proper centroid calculation
                self.compute_cluster_distance(
                    cluster1,
                    cluster2,
                    distance_matrix,
                    LinkageMethod::Average,
                )
            }
        }
    }

    fn get_neighbors(&self, distance_matrix: &Array2<f64>, point: usize, eps: f64) -> Vec<usize> {
        (0..distance_matrix.nrows())
            .filter(|&i| i != point && distance_matrix[[point, i]] <= eps)
            .collect()
    }

    fn expand_cluster(
        &self,
        distance_matrix: &Array2<f64>,
        labels: &mut Array1<usize>,
        point: usize,
        neighbors: &[usize],
        cluster_id: usize,
        config: &DBSCANConfig,
    ) -> ClusteringResult<()> {
        labels[point] = cluster_id;
        let mut seed_set = neighbors.to_vec();
        let mut i = 0;

        while i < seed_set.len() {
            let q = seed_set[i];

            if labels[q] == usize::MAX - 1 {
                // Was noise, now part of cluster
                labels[q] = cluster_id;
            }

            if labels[q] == usize::MAX {
                // Unvisited
                labels[q] = cluster_id;
                let q_neighbors = self.get_neighbors(distance_matrix, q, config.eps);

                if q_neighbors.len() >= config.min_samples {
                    // Add new neighbors to seed set
                    for &neighbor in &q_neighbors {
                        if !seed_set.contains(&neighbor) {
                            seed_set.push(neighbor);
                        }
                    }
                }
            }

            i += 1;
        }

        Ok(())
    }

    fn compute_shapelet_information_gain(
        &self,
        data: &Array2<f64>,
        labels: &Array1<usize>,
        shapelet: &Shapelet,
    ) -> ClusteringResult<f64> {
        let n_samples = data.nrows();
        let mut distances = Vec::new();

        // Compute distances from each time series to the shapelet
        for i in 0..n_samples {
            let series = data.row(i).to_owned();
            let min_distance = self.compute_min_distance_to_shapelet(&series, &shapelet.data)?;
            distances.push((min_distance, labels[i]));
        }

        // Sort by distance
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Compute initial entropy
        let initial_entropy = self.compute_entropy(labels)?;
        let mut best_info_gain: f64 = 0.0;

        // Try different split points
        for split_idx in 1..distances.len() {
            let _threshold = distances[split_idx].0;

            // Split data based on threshold
            let left_labels: Vec<usize> = distances[..split_idx]
                .iter()
                .map(|(_, label)| *label)
                .collect();
            let right_labels: Vec<usize> = distances[split_idx..]
                .iter()
                .map(|(_, label)| *label)
                .collect();

            if left_labels.is_empty() || right_labels.is_empty() {
                continue;
            }

            // Compute weighted entropy after split
            let left_entropy = self.compute_entropy(&Array1::from_vec(left_labels.clone()))?;
            let right_entropy = self.compute_entropy(&Array1::from_vec(right_labels.clone()))?;

            let left_weight = left_labels.len() as f64 / n_samples as f64;
            let right_weight = right_labels.len() as f64 / n_samples as f64;

            let weighted_entropy = left_weight * left_entropy + right_weight * right_entropy;
            let info_gain = initial_entropy - weighted_entropy;

            best_info_gain = best_info_gain.max(info_gain);
        }

        Ok(best_info_gain)
    }

    fn compute_min_distance_to_shapelet(
        &self,
        series: &Array1<f64>,
        shapelet: &Array1<f64>,
    ) -> ClusteringResult<f64> {
        let series_len = series.len();
        let shapelet_len = shapelet.len();

        if shapelet_len > series_len {
            return Ok(f64::INFINITY);
        }

        let mut min_distance = f64::INFINITY;

        for start_pos in 0..=(series_len - shapelet_len) {
            let subsequence = series
                .slice(s![start_pos..start_pos + shapelet_len])
                .to_owned();
            let distance = self.compute_series_distance(
                &subsequence,
                shapelet,
                TimeSeriesDistance::Euclidean,
            )?;
            min_distance = min_distance.min(distance);
        }

        Ok(min_distance)
    }

    fn compute_entropy(&self, labels: &Array1<usize>) -> ClusteringResult<f64> {
        if labels.is_empty() {
            return Ok(0.0);
        }

        let mut class_counts = HashMap::new();
        for &label in labels {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let total = labels.len() as f64;
        let mut entropy = 0.0;

        for count in class_counts.values() {
            let p = *count as f64 / total;
            if p > 0.0 {
                entropy -= p * p.log2();
            }
        }

        Ok(entropy)
    }
}

impl Default for TimeSeriesClusterer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_kmeans_clustering() {
        // Create simple test data
        let mut data = Array2::zeros((6, 10));

        // Cluster 1: sine waves
        for i in 0..3 {
            for j in 0..10 {
                data[[i, j]] = (j as f64 * 0.1).sin();
            }
        }

        // Cluster 2: cosine waves
        for i in 3..6 {
            for j in 0..10 {
                data[[i, j]] = (j as f64 * 0.1).cos();
            }
        }

        let clusterer = TimeSeriesClusterer::new();
        let config = KMeansConfig {
            n_clusters: 2,
            ..Default::default()
        };

        let result = clusterer.kmeans_clustering(&data, &config).unwrap();

        assert_eq!(result.n_clusters, 2);
        assert_eq!(result.cluster_labels.len(), 6);
        assert_eq!(result.centroids.len(), 2);
        assert!(result.inertia >= 0.0);
        assert!(result.silhouette_score >= -1.0 && result.silhouette_score <= 1.0);
    }

    #[test]
    fn test_knn_classification() {
        // Create simple test data
        let mut train_data = Array2::zeros((4, 10));
        let mut test_data = Array2::zeros((2, 10));

        // Class 0: sine waves
        for i in 0..2 {
            for j in 0..10 {
                train_data[[i, j]] = (j as f64 * 0.1).sin();
            }
        }

        // Class 1: cosine waves
        for i in 2..4 {
            for j in 0..10 {
                train_data[[i, j]] = (j as f64 * 0.1).cos();
            }
        }

        // Test data: one sine, one cosine
        for j in 0..10 {
            test_data[[0, j]] = (j as f64 * 0.1).sin();
            test_data[[1, j]] = (j as f64 * 0.1).cos();
        }

        let train_labels = Array1::from_vec(vec![0, 0, 1, 1]);

        let clusterer = TimeSeriesClusterer::new();
        let config = KNNConfig {
            k: 3,
            distance: TimeSeriesDistance::Euclidean,
            ..Default::default()
        };

        let result = clusterer
            .knn_classification(&train_data, &train_labels, &test_data, &config)
            .unwrap();

        assert_eq!(result.predicted_labels.len(), 2);
        assert_eq!(result.n_classes, 2);
        assert!(result.probabilities.is_some());
        assert_eq!(result.confidence_scores.len(), 2);

        // Should classify correctly
        assert_eq!(result.predicted_labels[0], 0); // Sine should be class 0
        assert_eq!(result.predicted_labels[1], 1); // Cosine should be class 1
    }

    #[test]
    fn test_discover_shapelets() {
        // Create simple test data with discriminative patterns
        let mut data = Array2::zeros((4, 20));
        let labels = Array1::from_vec(vec![0, 0, 1, 1]);

        // Class 0: has a peak in the middle
        for i in 0..2 {
            for j in 0..20 {
                if (8..=12).contains(&j) {
                    data[[i, j]] = 1.0; // Peak
                } else {
                    data[[i, j]] = 0.0;
                }
            }
        }

        // Class 1: no peak
        for i in 2..4 {
            for j in 0..20 {
                data[[i, j]] = 0.0;
            }
        }

        let clusterer = TimeSeriesClusterer::new();
        let config = ShapeletConfig {
            min_length: 3,
            max_length: 8,
            n_shapelets: 5,
            ..Default::default()
        };

        let result = clusterer
            .discover_shapelets(&data, &labels, &config)
            .unwrap();

        assert!(result.shapelets.len() <= config.n_shapelets);
        assert_eq!(result.information_gains.len(), result.shapelets.len());
        assert_eq!(result.n_series, 4);

        // Should find some shapelets with positive information gain
        assert!(result.information_gains.iter().any(|&gain| gain > 0.0));
    }
}
