//! Clustering algorithms for local minima in multi-start optimization
//!
//! This module provides algorithms to identify, cluster, and analyze local minima
//! found during multi-start optimization strategies. It helps distinguish between
//! unique local minima and provides insights into the optimization landscape.

use crate::error::OptimizeError;
use crate::unconstrained::{minimize, Method, Options};
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashMap;

/// Configuration for clustering local minima
#[derive(Debug, Clone)]
pub struct ClusteringOptions {
    /// Distance threshold for considering minima as belonging to the same cluster
    pub distance_threshold: f64,
    /// Relative tolerance for function values when clustering
    pub function_tolerance: f64,
    /// Maximum number of clusters to form
    pub max_clusters: Option<usize>,
    /// Clustering algorithm to use
    pub algorithm: ClusteringAlgorithm,
    /// Whether to normalize coordinates before clustering
    pub normalize_coordinates: bool,
    /// Whether to use function values in clustering
    pub use_function_values: bool,
    /// Weight for function values vs coordinates in distance calculation
    pub function_weight: f64,
}

impl Default for ClusteringOptions {
    fn default() -> Self {
        Self {
            distance_threshold: 1e-3,
            function_tolerance: 1e-6,
            max_clusters: None,
            algorithm: ClusteringAlgorithm::Hierarchical,
            normalize_coordinates: true,
            use_function_values: true,
            function_weight: 0.1,
        }
    }
}

/// Clustering algorithms available
#[derive(Debug, Clone, Copy)]
pub enum ClusteringAlgorithm {
    /// Hierarchical clustering with single linkage
    Hierarchical,
    /// K-means clustering
    KMeans,
    /// Density-based clustering (DBSCAN-like)
    Density,
    /// Custom threshold-based clustering
    Threshold,
}

/// Represents a local minimum found during optimization
#[derive(Debug, Clone)]
pub struct LocalMinimum<S> {
    /// Location of the minimum
    pub x: Array1<f64>,
    /// Function value at the minimum
    pub f: f64,
    /// Original function value (for generic type)
    pub fun_value: S,
    /// Number of optimization iterations to reach this minimum
    pub iterations: usize,
    /// Number of function evaluations
    pub func_evals: usize,
    /// Whether optimization was successful
    pub success: bool,
    /// Starting point that led to this minimum
    pub start_point: Array1<f64>,
    /// Cluster ID (assigned after clustering)
    pub cluster_id: Option<usize>,
    /// Distance to cluster centroid
    pub cluster_distance: Option<f64>,
}

/// Result of clustering analysis
#[derive(Debug, Clone)]
pub struct ClusteringResult<S> {
    /// All local minima found
    pub minima: Vec<LocalMinimum<S>>,
    /// Cluster centroids
    pub centroids: Vec<ClusterCentroid>,
    /// Number of clusters formed
    pub num_clusters: usize,
    /// Silhouette score (quality measure)
    pub silhouette_score: Option<f64>,
    /// Within-cluster sum of squares
    pub wcss: f64,
    /// Best minimum found (global optimum candidate)
    pub best_minimum: Option<LocalMinimum<S>>,
}

/// Cluster centroid information
#[derive(Debug, Clone)]
pub struct ClusterCentroid {
    /// Centroid coordinates
    pub x: Array1<f64>,
    /// Average function value in cluster
    pub f_avg: f64,
    /// Best (minimum) function value in cluster
    pub f_min: f64,
    /// Number of minima in this cluster
    pub size: usize,
    /// Cluster radius (max distance from centroid)
    pub radius: f64,
}

/// Multi-start optimization with clustering
pub fn multi_start_with_clustering<F, S>(
    fun: F,
    start_points: &[Array1<f64>],
    method: Method,
    options: Option<Options>,
    clustering_options: Option<ClusteringOptions>,
) -> Result<ClusteringResult<S>, OptimizeError>
where
    F: FnMut(&ArrayView1<f64>) -> S + Clone,
    S: Into<f64> + Clone + From<f64>,
{
    let clustering_opts = clustering_options.unwrap_or_default();
    let mut minima = Vec::new();

    // Run optimization from each starting point
    for start_point in start_points {
        let fun_clone = fun.clone();

        match minimize(
            fun_clone,
            start_point.as_slice().unwrap(),
            method,
            options.clone(),
        ) {
            Ok(result) => {
                let minimum = LocalMinimum {
                    x: result.x.clone(),
                    f: result.fun.clone().into(),
                    fun_value: result.fun,
                    iterations: result.iterations,
                    func_evals: result.func_evals,
                    success: result.success,
                    start_point: start_point.clone(),
                    cluster_id: None,
                    cluster_distance: None,
                };
                minima.push(minimum);
            }
            Err(_) => {
                // Skip failed optimizations but could log them
                continue;
            }
        }
    }

    if minima.is_empty() {
        return Err(OptimizeError::ComputationError(
            "No successful optimizations found".to_string(),
        ));
    }

    // Perform clustering
    cluster_minima(&mut minima, &clustering_opts)?;

    // Compute cluster centroids and statistics
    let centroids = compute_cluster_centroids(&minima)?;
    let num_clusters = centroids.len();
    let wcss = compute_wcss(&minima, &centroids);
    let silhouette_score = compute_silhouette_score(&minima);

    // Find best minimum
    let best_minimum = minima
        .iter()
        .filter(|m| m.success)
        .min_by(|a, b| a.f.partial_cmp(&b.f).unwrap())
        .cloned();

    Ok(ClusteringResult {
        minima,
        centroids,
        num_clusters,
        silhouette_score,
        wcss,
        best_minimum,
    })
}

/// Cluster local minima using the specified algorithm
fn cluster_minima<S>(
    minima: &mut [LocalMinimum<S>],
    options: &ClusteringOptions,
) -> Result<(), OptimizeError>
where
    S: Clone,
{
    if minima.is_empty() {
        return Ok(());
    }

    match options.algorithm {
        ClusteringAlgorithm::Hierarchical => hierarchical_clustering(minima, options),
        ClusteringAlgorithm::KMeans => kmeans_clustering(minima, options),
        ClusteringAlgorithm::Density => density_clustering(minima, options),
        ClusteringAlgorithm::Threshold => threshold_clustering(minima, options),
    }
}

/// Hierarchical clustering implementation
fn hierarchical_clustering<S>(
    minima: &mut [LocalMinimum<S>],
    options: &ClusteringOptions,
) -> Result<(), OptimizeError>
where
    S: Clone,
{
    let n = minima.len();
    if n <= 1 {
        if n == 1 {
            minima[0].cluster_id = Some(0);
        }
        return Ok(());
    }

    // Compute distance matrix
    let distances = compute_distance_matrix(minima, options);

    // Perform hierarchical clustering using single linkage
    let mut cluster_assignments = vec![None; n];
    let mut next_cluster_id = n;

    // Initialize: each point is its own cluster
    for (i, assignment) in cluster_assignments.iter_mut().enumerate().take(n) {
        *assignment = Some(i);
    }

    // Build hierarchy by merging closest clusters
    let mut active_clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while active_clusters.len() > 1 {
        let mut min_dist = f64::INFINITY;
        let mut merge_pair = (0, 1);

        // Find closest pair of clusters
        for i in 0..active_clusters.len() {
            for j in (i + 1)..active_clusters.len() {
                let cluster_dist =
                    compute_cluster_distance(&active_clusters[i], &active_clusters[j], &distances);
                if cluster_dist < min_dist {
                    min_dist = cluster_dist;
                    merge_pair = (i, j);
                }
            }
        }

        // Stop if distance threshold exceeded
        if min_dist > options.distance_threshold {
            break;
        }

        // Check max clusters constraint
        if let Some(max_clusters) = options.max_clusters {
            if active_clusters.len() <= max_clusters {
                break;
            }
        }

        // Merge clusters
        let (i, j) = merge_pair;
        let mut merged_cluster = active_clusters[i].clone();
        merged_cluster.extend(&active_clusters[j]);

        // Update cluster assignments
        for &point_idx in &merged_cluster {
            cluster_assignments[point_idx] = Some(next_cluster_id);
        }

        // Remove old clusters and add merged one
        let mut new_clusters = Vec::new();
        for (k, cluster) in active_clusters.iter().enumerate() {
            if k != i && k != j {
                new_clusters.push(cluster.clone());
            }
        }
        new_clusters.push(merged_cluster);
        active_clusters = new_clusters;
        next_cluster_id += 1;
    }

    // Assign final cluster IDs (renumber to be sequential)
    let mut cluster_map = HashMap::new();
    let mut final_cluster_id = 0;

    for cluster_id in cluster_assignments.iter().flatten() {
        if !cluster_map.contains_key(cluster_id) {
            cluster_map.insert(*cluster_id, final_cluster_id);
            final_cluster_id += 1;
        }
    }

    // Update minima with final cluster assignments
    for (i, minimum) in minima.iter_mut().enumerate() {
        if let Some(cluster_id) = cluster_assignments[i] {
            minimum.cluster_id = cluster_map.get(&cluster_id).copied();
        }
    }

    Ok(())
}

/// K-means clustering implementation
fn kmeans_clustering<S>(
    minima: &mut [LocalMinimum<S>],
    options: &ClusteringOptions,
) -> Result<(), OptimizeError>
where
    S: Clone,
{
    let n = minima.len();
    if n <= 1 {
        if n == 1 {
            minima[0].cluster_id = Some(0);
        }
        return Ok(());
    }

    // Determine number of clusters
    let k = if let Some(max_k) = options.max_clusters {
        std::cmp::min(max_k, n)
    } else {
        // Use elbow method or default to sqrt(n)
        std::cmp::min((n as f64).sqrt() as usize + 1, n)
    };

    if k >= n {
        // Each point is its own cluster
        for (i, minimum) in minima.iter_mut().enumerate() {
            minimum.cluster_id = Some(i);
        }
        return Ok(());
    }

    // Get feature vectors for clustering
    let features = extract_features(minima, options);
    let dim = features.ncols();

    // Initialize centroids randomly (or use k-means++)
    let mut centroids = initialize_centroids_plus_plus(&features, k);
    let mut assignments = vec![0; n];
    let max_iter = 100;
    let tolerance = 1e-6;

    for _iter in 0..max_iter {
        let mut changed = false;

        // Assign points to nearest centroids
        for (i, assignment) in assignments.iter_mut().enumerate().take(n) {
            let mut min_dist = f64::INFINITY;
            let mut best_cluster = 0;

            for j in 0..k {
                let dist = euclidean_distance(&features.row(i), &centroids.row(j));
                if dist < min_dist {
                    min_dist = dist;
                    best_cluster = j;
                }
            }

            if *assignment != best_cluster {
                *assignment = best_cluster;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut new_centroids = Array2::zeros((k, dim));
        let mut cluster_sizes = vec![0; k];

        for i in 0..n {
            let cluster = assignments[i];
            cluster_sizes[cluster] += 1;
            for d in 0..dim {
                new_centroids[[cluster, d]] += features[[i, d]];
            }
        }

        for j in 0..k {
            if cluster_sizes[j] > 0 {
                for d in 0..dim {
                    new_centroids[[j, d]] /= cluster_sizes[j] as f64;
                }
            }
        }

        // Check convergence
        let centroid_change = (&centroids - &new_centroids).mapv(|x| x.abs()).sum();

        centroids = new_centroids;

        if centroid_change < tolerance {
            break;
        }
    }

    // Assign cluster IDs to minima
    for (i, minimum) in minima.iter_mut().enumerate() {
        minimum.cluster_id = Some(assignments[i]);
    }

    Ok(())
}

/// Density-based clustering (DBSCAN-like)
fn density_clustering<S>(
    minima: &mut [LocalMinimum<S>],
    options: &ClusteringOptions,
) -> Result<(), OptimizeError>
where
    S: Clone,
{
    let n = minima.len();
    if n <= 1 {
        if n == 1 {
            minima[0].cluster_id = Some(0);
        }
        return Ok(());
    }

    let eps = options.distance_threshold;
    let min_pts = 2; // Minimum points to form a cluster

    let distances = compute_distance_matrix(minima, options);
    let mut cluster_assignments = vec![None; n];
    let mut visited = vec![false; n];
    let mut cluster_id = 0;

    for i in 0..n {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        // Find neighbors
        let neighbors: Vec<usize> = (0..n).filter(|&j| distances[[i, j]] <= eps).collect();

        if neighbors.len() < min_pts {
            // Mark as noise (will remain None)
            continue;
        }

        // Start new cluster
        let mut to_visit = neighbors.clone();
        cluster_assignments[i] = Some(cluster_id);

        let mut idx = 0;
        while idx < to_visit.len() {
            let point = to_visit[idx];

            if !visited[point] {
                visited[point] = true;

                let point_neighbors: Vec<usize> =
                    (0..n).filter(|&j| distances[[point, j]] <= eps).collect();

                if point_neighbors.len() >= min_pts {
                    // Add new neighbors to visit list
                    for &neighbor in &point_neighbors {
                        if !to_visit.contains(&neighbor) {
                            to_visit.push(neighbor);
                        }
                    }
                }
            }

            if cluster_assignments[point].is_none() {
                cluster_assignments[point] = Some(cluster_id);
            }

            idx += 1;
        }

        cluster_id += 1;
    }

    // Assign cluster IDs to minima
    for (i, minimum) in minima.iter_mut().enumerate() {
        minimum.cluster_id = cluster_assignments[i];
    }

    Ok(())
}

/// Simple threshold-based clustering
fn threshold_clustering<S>(
    minima: &mut [LocalMinimum<S>],
    options: &ClusteringOptions,
) -> Result<(), OptimizeError>
where
    S: Clone,
{
    let n = minima.len();
    if n <= 1 {
        if n == 1 {
            minima[0].cluster_id = Some(0);
        }
        return Ok(());
    }

    let mut cluster_assignments = vec![None; n];
    let mut cluster_id = 0;

    for i in 0..n {
        if cluster_assignments[i].is_some() {
            continue;
        }

        // Start new cluster
        cluster_assignments[i] = Some(cluster_id);

        // Find all points within threshold
        for j in (i + 1)..n {
            if cluster_assignments[j].is_some() {
                continue;
            }

            let distance = compute_distance(&minima[i], &minima[j], options);
            if distance <= options.distance_threshold {
                cluster_assignments[j] = Some(cluster_id);
            }
        }

        cluster_id += 1;
    }

    // Assign cluster IDs to minima
    for (i, minimum) in minima.iter_mut().enumerate() {
        minimum.cluster_id = cluster_assignments[i];
    }

    Ok(())
}

/// Compute distance matrix between all minima
fn compute_distance_matrix<S>(
    minima: &[LocalMinimum<S>],
    options: &ClusteringOptions,
) -> Array2<f64>
where
    S: Clone,
{
    let n = minima.len();
    let mut distances = Array2::zeros((n, n));

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = compute_distance(&minima[i], &minima[j], options);
            distances[[i, j]] = dist;
            distances[[j, i]] = dist;
        }
    }

    distances
}

/// Compute distance between two local minima
fn compute_distance<S>(
    min1: &LocalMinimum<S>,
    min2: &LocalMinimum<S>,
    options: &ClusteringOptions,
) -> f64
where
    S: Clone,
{
    // Coordinate distance
    let coord_dist = (&min1.x - &min2.x).mapv(|x| x.powi(2)).sum().sqrt();

    if !options.use_function_values {
        return coord_dist;
    }

    // Function value distance
    let func_dist = (min1.f - min2.f).abs();

    // Combined distance
    coord_dist + options.function_weight * func_dist
}

/// Compute distance between two clusters (single linkage)
fn compute_cluster_distance(
    cluster1: &[usize],
    cluster2: &[usize],
    distances: &Array2<f64>,
) -> f64 {
    let mut min_dist = f64::INFINITY;

    for &i in cluster1 {
        for &j in cluster2 {
            let dist = distances[[i, j]];
            if dist < min_dist {
                min_dist = dist;
            }
        }
    }

    min_dist
}

/// Extract feature vectors for clustering
fn extract_features<S>(minima: &[LocalMinimum<S>], options: &ClusteringOptions) -> Array2<f64>
where
    S: Clone,
{
    let n = minima.len();
    if n == 0 {
        return Array2::zeros((0, 0));
    }

    let coord_dim = minima[0].x.len();
    let func_dim = if options.use_function_values { 1 } else { 0 };
    let total_dim = coord_dim + func_dim;

    let mut features = Array2::zeros((n, total_dim));

    // Extract coordinates
    for (i, minimum) in minima.iter().enumerate() {
        for j in 0..coord_dim {
            features[[i, j]] = minimum.x[j];
        }
    }

    // Add function values if requested
    if options.use_function_values {
        for (i, minimum) in minima.iter().enumerate() {
            features[[i, coord_dim]] = minimum.f * options.function_weight;
        }
    }

    // Normalize if requested
    if options.normalize_coordinates {
        normalize_features(&mut features, coord_dim);
    }

    features
}

/// Normalize feature matrix
fn normalize_features(features: &mut Array2<f64>, coord_dim: usize) {
    let (n, _) = features.dim();
    if n == 0 {
        return;
    }

    // Normalize coordinates only
    for j in 0..coord_dim {
        let col = features.column(j);
        let min_val = col.iter().fold(f64::INFINITY, |a, &b| f64::min(a, b));
        let max_val = col.iter().fold(f64::NEG_INFINITY, |a, &b| f64::max(a, b));

        if (max_val - min_val).abs() > 1e-10 {
            for i in 0..n {
                features[[i, j]] = (features[[i, j]] - min_val) / (max_val - min_val);
            }
        }
    }
}

/// Initialize centroids using k-means++ algorithm
fn initialize_centroids_plus_plus(features: &Array2<f64>, k: usize) -> Array2<f64> {
    let (n, dim) = features.dim();
    let mut centroids = Array2::zeros((k, dim));

    if n == 0 || k == 0 {
        return centroids;
    }

    // Choose first centroid randomly
    let first_idx = 0; // In practice, should be random
    centroids.row_mut(0).assign(&features.row(first_idx));

    // Choose remaining centroids
    for c in 1..k {
        let mut distances = vec![f64::INFINITY; n];

        // Compute distance to nearest centroid for each point
        for (i, distance) in distances.iter_mut().enumerate().take(n) {
            let point = features.row(i);
            for j in 0..c {
                let centroid = centroids.row(j);
                let dist = euclidean_distance(&point, &centroid);
                *distance = distance.min(dist);
            }
        }

        // Choose next centroid (should be weighted random, using max for simplicity)
        let next_idx = distances
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        centroids.row_mut(c).assign(&features.row(next_idx));
    }

    centroids
}

/// Compute Euclidean distance between two points
fn euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    (a - b).mapv(|x| x.powi(2)).sum().sqrt()
}

/// Compute cluster centroids and statistics
fn compute_cluster_centroids<S>(
    minima: &[LocalMinimum<S>],
) -> Result<Vec<ClusterCentroid>, OptimizeError>
where
    S: Clone,
{
    if minima.is_empty() {
        return Ok(Vec::new());
    }

    // Group minima by cluster
    let mut clusters: HashMap<usize, Vec<&LocalMinimum<S>>> = HashMap::new();

    for minimum in minima {
        if let Some(cluster_id) = minimum.cluster_id {
            clusters.entry(cluster_id).or_default().push(minimum);
        }
    }

    let mut centroids = Vec::new();

    for (_, cluster_minima) in clusters {
        if cluster_minima.is_empty() {
            continue;
        }

        let dim = cluster_minima[0].x.len();
        let mut centroid_x = Array1::zeros(dim);
        let mut f_sum = 0.0;
        let mut f_min = f64::INFINITY;

        // Compute centroid coordinates and function statistics
        for minimum in &cluster_minima {
            centroid_x = &centroid_x + &minimum.x;
            f_sum += minimum.f;
            f_min = f_min.min(minimum.f);
        }

        let size = cluster_minima.len();
        centroid_x /= size as f64;
        let f_avg = f_sum / size as f64;

        // Compute cluster radius
        let mut max_radius = 0.0;
        for minimum in &cluster_minima {
            let dist = (&minimum.x - &centroid_x).mapv(|x| x.powi(2)).sum().sqrt();
            max_radius = f64::max(max_radius, dist);
        }

        centroids.push(ClusterCentroid {
            x: centroid_x,
            f_avg,
            f_min,
            size,
            radius: max_radius,
        });
    }

    Ok(centroids)
}

/// Compute within-cluster sum of squares
fn compute_wcss<S>(minima: &[LocalMinimum<S>], centroids: &[ClusterCentroid]) -> f64
where
    S: Clone,
{
    let mut wcss = 0.0;

    for minimum in minima {
        if let Some(cluster_id) = minimum.cluster_id {
            if cluster_id < centroids.len() {
                let centroid = &centroids[cluster_id];
                let dist = (&minimum.x - &centroid.x).mapv(|x| x.powi(2)).sum();
                wcss += dist;
            }
        }
    }

    wcss
}

/// Compute silhouette score for clustering quality
fn compute_silhouette_score<S>(minima: &[LocalMinimum<S>]) -> Option<f64>
where
    S: Clone,
{
    if minima.len() < 2 {
        return None;
    }

    let mut silhouette_sum = 0.0;
    let mut valid_points = 0;

    for (i, minimum) in minima.iter().enumerate() {
        if let Some(cluster_id) = minimum.cluster_id {
            // Compute intra-cluster distance
            let mut intra_sum = 0.0;
            let mut intra_count = 0;

            // Compute inter-cluster distances
            let mut min_inter = f64::INFINITY;
            let mut cluster_inter_sums: HashMap<usize, f64> = HashMap::new();
            let mut cluster_inter_counts: HashMap<usize, usize> = HashMap::new();

            for (j, other) in minima.iter().enumerate() {
                if i == j {
                    continue;
                }

                if let Some(other_cluster_id) = other.cluster_id {
                    let dist = (&minimum.x - &other.x).mapv(|x| x.powi(2)).sum().sqrt();

                    if other_cluster_id == cluster_id {
                        intra_sum += dist;
                        intra_count += 1;
                    } else {
                        *cluster_inter_sums.entry(other_cluster_id).or_insert(0.0) += dist;
                        *cluster_inter_counts.entry(other_cluster_id).or_insert(0) += 1;
                    }
                }
            }

            // Find minimum inter-cluster distance
            for (other_cluster_id, sum) in cluster_inter_sums {
                let count = cluster_inter_counts[&other_cluster_id];
                if count > 0 {
                    let avg_inter = sum / count as f64;
                    min_inter = min_inter.min(avg_inter);
                }
            }

            if intra_count > 0 && min_inter < f64::INFINITY {
                let a = intra_sum / intra_count as f64;
                let b = min_inter;
                let silhouette = (b - a) / f64::max(a, b);
                silhouette_sum += silhouette;
                valid_points += 1;
            }
        }
    }

    if valid_points > 0 {
        Some(silhouette_sum / valid_points as f64)
    } else {
        None
    }
}

/// Generate diverse starting points for multi-start optimization
pub fn generate_diverse_start_points(
    bounds: &[(f64, f64)],
    num_points: usize,
    strategy: StartPointStrategy,
) -> Vec<Array1<f64>> {
    match strategy {
        StartPointStrategy::Random => generate_random_points(bounds, num_points),
        StartPointStrategy::LatinHypercube => generate_latin_hypercube_points(bounds, num_points),
        StartPointStrategy::Grid => generate_grid_points(bounds, num_points),
        StartPointStrategy::Sobol => generate_sobol_points(bounds, num_points),
    }
}

/// Strategy for generating starting points
#[derive(Debug, Clone, Copy)]
pub enum StartPointStrategy {
    Random,
    LatinHypercube,
    Grid,
    Sobol,
}

/// Generate random starting points
fn generate_random_points(bounds: &[(f64, f64)], num_points: usize) -> Vec<Array1<f64>> {
    let dim = bounds.len();
    let mut points = Vec::new();

    for _ in 0..num_points {
        let mut point = Array1::zeros(dim);
        for (i, &(low, high)) in bounds.iter().enumerate() {
            // Simple pseudo-random (in practice, use proper RNG)
            let t = (i * num_points + points.len()) as f64 / (num_points * dim) as f64;
            let random_val = (t * 17.0).fract(); // Simple pseudo-random
            point[i] = low + random_val * (high - low);
        }
        points.push(point);
    }

    points
}

/// Generate Latin Hypercube sampling points
fn generate_latin_hypercube_points(bounds: &[(f64, f64)], num_points: usize) -> Vec<Array1<f64>> {
    let dim = bounds.len();
    let mut points = Vec::new();

    // Simple Latin Hypercube implementation
    for i in 0..num_points {
        let mut point = Array1::zeros(dim);
        for (j, &(low, high)) in bounds.iter().enumerate() {
            let segment = (i as f64 + 0.5) / num_points as f64; // Center of segment
            point[j] = low + segment * (high - low);
        }
        points.push(point);
    }

    points
}

/// Generate grid points
fn generate_grid_points(bounds: &[(f64, f64)], num_points: usize) -> Vec<Array1<f64>> {
    let dim = bounds.len();
    if dim == 0 {
        return Vec::new();
    }

    let points_per_dim = ((num_points as f64).powf(1.0 / dim as f64)).ceil() as usize;
    let mut points = Vec::new();

    // Generate grid coordinates recursively
    fn generate_grid_recursive(
        bounds: &[(f64, f64)],
        points_per_dim: usize,
        current_point: &mut Array1<f64>,
        dim_idx: usize,
        points: &mut Vec<Array1<f64>>,
    ) {
        if dim_idx >= bounds.len() {
            points.push(current_point.clone());
            return;
        }

        let (low, high) = bounds[dim_idx];
        for i in 0..points_per_dim {
            let t = if points_per_dim == 1 {
                0.5
            } else {
                i as f64 / (points_per_dim - 1) as f64
            };
            current_point[dim_idx] = low + t * (high - low);
            generate_grid_recursive(bounds, points_per_dim, current_point, dim_idx + 1, points);
        }
    }

    let mut current_point = Array1::zeros(dim);
    generate_grid_recursive(bounds, points_per_dim, &mut current_point, 0, &mut points);

    // Truncate to requested number of points
    points.truncate(num_points);
    points
}

/// Generate Sobol sequence points (simplified)
fn generate_sobol_points(bounds: &[(f64, f64)], num_points: usize) -> Vec<Array1<f64>> {
    // Simplified Sobol sequence (in practice, use proper implementation)
    let dim = bounds.len();
    let mut points = Vec::new();

    for i in 0..num_points {
        let mut point = Array1::zeros(dim);
        for (j, &(low, high)) in bounds.iter().enumerate() {
            // Van der Corput sequence for dimension j
            let mut n = i + 1;
            let base = 2 + j; // Different base for each dimension
            let mut result = 0.0;
            let mut f = 1.0 / base as f64;

            while n > 0 {
                result += (n % base) as f64 * f;
                n /= base;
                f /= base as f64;
            }

            point[j] = low + result * (high - low);
        }
        points.push(point);
    }

    points
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_simple_clustering() {
        // Create some test minima
        let mut minima = vec![
            LocalMinimum {
                x: Array1::from_vec(vec![0.0, 0.0]),
                f: 1.0,
                fun_value: 1.0,
                iterations: 10,
                func_evals: 20,
                success: true,
                start_point: Array1::from_vec(vec![1.0, 1.0]),
                cluster_id: None,
                cluster_distance: None,
            },
            LocalMinimum {
                x: Array1::from_vec(vec![0.1, 0.1]),
                f: 1.1,
                fun_value: 1.1,
                iterations: 12,
                func_evals: 24,
                success: true,
                start_point: Array1::from_vec(vec![1.1, 1.1]),
                cluster_id: None,
                cluster_distance: None,
            },
            LocalMinimum {
                x: Array1::from_vec(vec![5.0, 5.0]),
                f: 2.0,
                fun_value: 2.0,
                iterations: 15,
                func_evals: 30,
                success: true,
                start_point: Array1::from_vec(vec![5.5, 5.5]),
                cluster_id: None,
                cluster_distance: None,
            },
        ];

        let options = ClusteringOptions {
            distance_threshold: 1.0,
            algorithm: ClusteringAlgorithm::Threshold,
            ..Default::default()
        };

        threshold_clustering(&mut minima, &options).unwrap();

        // First two should be in same cluster, third in different cluster
        assert_eq!(minima[0].cluster_id, minima[1].cluster_id);
        assert_ne!(minima[0].cluster_id, minima[2].cluster_id);
    }

    #[test]
    fn test_distance_computation() {
        let min1 = LocalMinimum {
            x: Array1::from_vec(vec![0.0, 0.0]),
            f: 1.0,
            fun_value: 1.0,
            iterations: 10,
            func_evals: 20,
            success: true,
            start_point: Array1::from_vec(vec![1.0, 1.0]),
            cluster_id: None,
            cluster_distance: None,
        };

        let min2 = LocalMinimum {
            x: Array1::from_vec(vec![3.0, 4.0]),
            f: 2.0,
            fun_value: 2.0,
            iterations: 12,
            func_evals: 24,
            success: true,
            start_point: Array1::from_vec(vec![3.5, 4.5]),
            cluster_id: None,
            cluster_distance: None,
        };

        let options = ClusteringOptions::default();
        let distance = compute_distance(&min1, &min2, &options);

        // Euclidean distance is 5.0, plus function weight * |1.0 - 2.0| = 5.0 + 0.1 * 1.0 = 5.1
        assert_abs_diff_eq!(distance, 5.1, epsilon = 1e-10);
    }

    #[test]
    fn test_start_point_generation() {
        let bounds = vec![(0.0, 10.0), (-5.0, 5.0)];
        let num_points = 5;

        let random_points =
            generate_diverse_start_points(&bounds, num_points, StartPointStrategy::Random);
        assert_eq!(random_points.len(), num_points);

        for point in &random_points {
            assert!(point[0] >= 0.0 && point[0] <= 10.0);
            assert!(point[1] >= -5.0 && point[1] <= 5.0);
        }

        let grid_points =
            generate_diverse_start_points(&bounds, num_points, StartPointStrategy::Grid);
        assert_eq!(grid_points.len(), num_points);

        for point in &grid_points {
            assert!(point[0] >= 0.0 && point[0] <= 10.0);
            assert!(point[1] >= -5.0 && point[1] <= 5.0);
        }
    }
}
