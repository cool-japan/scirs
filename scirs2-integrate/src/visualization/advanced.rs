//! Advanced visualization features for scientific computing
//!
//! This module provides advanced visualization capabilities including multi-dimensional
//! data visualization, dimension reduction techniques, clustering algorithms, and animated
//! visualizations for time-series and dynamic systems.

use super::types::*;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::ODEResult;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::Rng;
use std::collections::HashMap;

/// Multi-dimensional data visualization engine
#[derive(Debug, Clone)]
pub struct MultiDimensionalVisualizer {
    /// Dimension reduction method
    pub reduction_method: DimensionReductionMethod,
    /// Target dimensions for visualization
    pub target_dimensions: usize,
    /// Clustering method for data grouping
    pub clustering_method: ClusteringMethod,
}

impl MultiDimensionalVisualizer {
    /// Create new multi-dimensional visualizer
    pub fn new() -> Self {
        Self {
            reduction_method: DimensionReductionMethod::PCA,
            target_dimensions: 2,
            clustering_method: ClusteringMethod::None,
        }
    }

    /// Visualize high-dimensional data
    pub fn visualize_high_dimensional_data(
        &self,
        data: &Array2<f64>,
        labels: Option<&Array1<usize>>,
    ) -> IntegrateResult<HighDimensionalPlot> {
        // Apply dimension reduction
        let reduced_data = self.apply_dimension_reduction(data)?;

        // Apply clustering if specified
        let cluster_labels = self.apply_clustering(&reduced_data)?;

        // Create plot data
        let x: Vec<f64> = reduced_data.column(0).to_vec();
        let y: Vec<f64> = if reduced_data.ncols() > 1 {
            reduced_data.column(1).to_vec()
        } else {
            vec![0.0; x.len()]
        };

        let z: Option<Vec<f64>> = if self.target_dimensions > 2 && reduced_data.ncols() > 2 {
            Some(reduced_data.column(2).to_vec())
        } else {
            None
        };

        let colors = if let Some(labels) = labels {
            labels.to_vec().into_iter().map(|l| l as f64).collect()
        } else if let Some(clusters) = &cluster_labels {
            clusters.iter().map(|&c| c as f64).collect()
        } else {
            (0..x.len()).map(|i| i as f64).collect()
        };

        let mut metadata = PlotMetadata::default();
        metadata.title = "High-Dimensional Data Visualization".to_string();
        metadata.xlabel = format!("{:?} Component 1", self.reduction_method);
        metadata.ylabel = format!("{:?} Component 2", self.reduction_method);

        Ok(HighDimensionalPlot {
            x,
            y,
            z,
            colors,
            cluster_labels,
            original_dimensions: data.ncols(),
            reduced_dimensions: self.target_dimensions,
            reduction_method: self.reduction_method,
            metadata,
        })
    }

    /// Apply dimension reduction to data
    fn apply_dimension_reduction(&self, data: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        match self.reduction_method {
            DimensionReductionMethod::PCA => self.apply_pca(data),
            DimensionReductionMethod::TSNE => self.apply_tsne(data),
            DimensionReductionMethod::UMAP => self.apply_umap(data),
            DimensionReductionMethod::LDA => self.apply_lda(data),
            DimensionReductionMethod::MDS => self.apply_mds(data),
        }
    }

    /// Apply Principal Component Analysis
    fn apply_pca(&self, data: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        let (n_samples, n_features) = data.dim();

        // Center the data
        let mean = data.mean_axis(Axis(0)).unwrap();
        let centered_data = data - &mean.insert_axis(Axis(0));

        // Compute covariance matrix
        let cov_matrix = centered_data.t().dot(&centered_data) / (n_samples - 1) as f64;

        // Simplified eigenvalue decomposition (for small matrices)
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&cov_matrix)?;

        // Sort by eigenvalue magnitude (descending)
        let mut eigenvalue_indices: Vec<usize> = (0..eigenvalues.len()).collect();
        eigenvalue_indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Project data onto principal components
        let n_components = self.target_dimensions.min(n_features);
        let mut projected_data = Array2::zeros((n_samples, n_components));

        for (i, &idx) in eigenvalue_indices.iter().take(n_components).enumerate() {
            let component = eigenvectors.column(idx);
            let projection = centered_data.dot(&component);
            projected_data.column_mut(i).assign(&projection);
        }

        Ok(projected_data)
    }

    /// Apply t-SNE (simplified implementation)
    fn apply_tsne(&self, data: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        // Simplified t-SNE implementation - in practice would use proper t-SNE algorithm
        let (n_samples, _) = data.dim();
        let mut rng = scirs2_core::random::rng();

        // Random initialization
        let mut embedding = Array2::zeros((n_samples, self.target_dimensions));
        for i in 0..n_samples {
            for j in 0..self.target_dimensions {
                embedding[[i, j]] = rng.random::<f64>() * 2.0 - 1.0;
            }
        }

        // For this simplified version, return random initialization
        // In practice, would implement full t-SNE gradient descent
        Ok(embedding)
    }

    /// Apply UMAP (simplified implementation)
    fn apply_umap(&self, data: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        // Simplified UMAP - in practice would use proper UMAP algorithm
        // For now, fall back to PCA
        self.apply_pca(data)
    }

    /// Apply Linear Discriminant Analysis
    fn apply_lda(&self, data: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        // Simplified LDA - would need class labels for proper implementation
        // For now, fall back to PCA
        self.apply_pca(data)
    }

    /// Apply Multidimensional Scaling
    fn apply_mds(&self, data: &Array2<f64>) -> IntegrateResult<Array2<f64>> {
        let n_samples = data.nrows();

        // Compute distance matrix
        let mut distance_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                distance_matrix[[i, j]] = dist;
                distance_matrix[[j, i]] = dist;
            }
        }

        // Classical MDS using eigendecomposition
        let squared_distances = distance_matrix.mapv(|d| d * d);

        // Double centering
        let row_means = squared_distances.mean_axis(Axis(1)).unwrap();
        let col_means = squared_distances.mean_axis(Axis(0)).unwrap();
        let grand_mean = squared_distances.iter().sum::<f64>() / squared_distances.len() as f64;

        let mut b_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                b_matrix[[i, j]] =
                    -0.5 * (squared_distances[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        // Eigendecomposition of B matrix
        let (eigenvalues, eigenvectors) = self.compute_eigendecomposition(&b_matrix)?;

        // Sort eigenvalues in descending order
        let mut eigenvalue_indices: Vec<usize> = (0..eigenvalues.len()).collect();
        eigenvalue_indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        // Construct embedding
        let n_components = self.target_dimensions.min(n_samples);
        let mut embedding = Array2::zeros((n_samples, n_components));

        for (i, &idx) in eigenvalue_indices.iter().take(n_components).enumerate() {
            if eigenvalues[idx] > 0.0 {
                let scale = eigenvalues[idx].sqrt();
                for j in 0..n_samples {
                    embedding[[j, i]] = eigenvectors[[j, idx]] * scale;
                }
            }
        }

        Ok(embedding)
    }

    /// Apply clustering to reduced data
    fn apply_clustering(&self, data: &Array2<f64>) -> IntegrateResult<Option<Vec<usize>>> {
        match self.clustering_method {
            ClusteringMethod::KMeans { k } => Ok(Some(self.kmeans_clustering(data, k)?)),
            ClusteringMethod::DBSCAN { eps, min_samples } => {
                Ok(Some(self.dbscan_clustering(data, eps, min_samples)?))
            }
            ClusteringMethod::Hierarchical { n_clusters } => {
                Ok(Some(self.hierarchical_clustering(data, n_clusters)?))
            }
            ClusteringMethod::None => Ok(None),
        }
    }

    /// K-means clustering implementation
    fn kmeans_clustering(&self, data: &Array2<f64>, k: usize) -> IntegrateResult<Vec<usize>> {
        let mut rng = scirs2_core::random::rng();
        let (n_samples, n_features) = data.dim();

        // Initialize centroids randomly
        let mut centroids = Array2::zeros((k, n_features));
        for i in 0..k {
            for j in 0..n_features {
                let min_val = data.column(j).iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max_val = data
                    .column(j)
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                centroids[[i, j]] = min_val + rng.random::<f64>() * (max_val - min_val);
            }
        }

        let mut labels = vec![0; n_samples];
        let max_iterations = 100;

        for _iteration in 0..max_iterations {
            let mut changed = false;

            // Assign points to nearest centroids
            for i in 0..n_samples {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for j in 0..k {
                    let distance = self.euclidean_distance(&data.row(i), &centroids.row(j));
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = j;
                    }
                }

                if labels[i] != best_cluster {
                    labels[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut cluster_counts = vec![0; k];
            centroids.fill(0.0);

            for i in 0..n_samples {
                let cluster = labels[i];
                cluster_counts[cluster] += 1;
                for j in 0..n_features {
                    centroids[[cluster, j]] += data[[i, j]];
                }
            }

            for i in 0..k {
                if cluster_counts[i] > 0 {
                    for j in 0..n_features {
                        centroids[[i, j]] /= cluster_counts[i] as f64;
                    }
                }
            }
        }

        Ok(labels)
    }

    /// DBSCAN clustering implementation (simplified)
    fn dbscan_clustering(
        &self,
        data: &Array2<f64>,
        eps: f64,
        min_samples: usize,
    ) -> IntegrateResult<Vec<usize>> {
        let n_samples = data.nrows();
        let mut labels = vec![usize::MAX; n_samples]; // MAX means unclassified
        let mut cluster_id = 0;

        for i in 0..n_samples {
            if labels[i] != usize::MAX {
                continue; // Already processed
            }

            let neighbors = self.find_neighbors(data, i, eps);

            if neighbors.len() < min_samples {
                labels[i] = usize::MAX - 1; // Mark as noise
            } else {
                self.expand_cluster(
                    data,
                    i,
                    &neighbors,
                    cluster_id,
                    eps,
                    min_samples,
                    &mut labels,
                );
                cluster_id += 1;
            }
        }

        // Convert noise points to cluster 0 and increment others
        for label in &mut labels {
            if *label == usize::MAX - 1 {
                *label = 0; // Noise cluster
            } else if *label != usize::MAX {
                *label += 1; // Shift cluster IDs
            }
        }

        Ok(labels)
    }

    /// Hierarchical clustering implementation (simplified)
    fn hierarchical_clustering(
        &self,
        data: &Array2<f64>,
        n_clusters: usize,
    ) -> IntegrateResult<Vec<usize>> {
        let n_samples = data.nrows();

        // Initialize each point as its own cluster
        let mut clusters: Vec<Vec<usize>> = (0..n_samples).map(|i| vec![i]).collect();

        // Compute initial distance matrix
        let mut distance_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let dist = self.euclidean_distance(&data.row(i), &data.row(j));
                distance_matrix[[i, j]] = dist;
                distance_matrix[[j, i]] = dist;
            }
        }

        // Agglomerative clustering
        while clusters.len() > n_clusters {
            let mut min_distance = f64::INFINITY;
            let mut merge_i = 0;
            let mut merge_j = 1;

            // Find closest clusters
            for i in 0..clusters.len() {
                for j in i + 1..clusters.len() {
                    let dist = self.cluster_distance(&clusters[i], &clusters[j], &distance_matrix);
                    if dist < min_distance {
                        min_distance = dist;
                        merge_i = i;
                        merge_j = j;
                    }
                }
            }

            // Merge clusters
            let mut merged_cluster = clusters[merge_i].clone();
            merged_cluster.extend(&clusters[merge_j]);

            // Remove old clusters and add merged cluster
            clusters.remove(merge_j); // Remove j first (higher index)
            clusters.remove(merge_i);
            clusters.push(merged_cluster);
        }

        // Assign labels
        let mut labels = vec![0; n_samples];
        for (cluster_id, cluster) in clusters.iter().enumerate() {
            for &point_id in cluster {
                labels[point_id] = cluster_id;
            }
        }

        Ok(labels)
    }

    /// Helper functions
    fn euclidean_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<f64>,
        b: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn compute_eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> IntegrateResult<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();

        // Simplified eigendecomposition using power iteration
        let mut eigenvalues = Array1::zeros(n.min(self.target_dimensions));
        let mut eigenvectors = Array2::zeros((n, n.min(self.target_dimensions)));
        let mut remaining_matrix = matrix.clone();

        for k in 0..n.min(self.target_dimensions) {
            // Power iteration for largest eigenvalue
            let mut v = Array1::from_elem(n, 1.0 / (n as f64).sqrt());
            let mut eigenvalue = 0.0;

            for _ in 0..100 {
                let v_new = remaining_matrix.dot(&v);
                eigenvalue = v.dot(&v_new);
                let norm = v_new.iter().map(|&x| x * x).sum::<f64>().sqrt();
                if norm > 1e-10 {
                    v = v_new / norm;
                }
            }

            eigenvalues[k] = eigenvalue;
            eigenvectors.column_mut(k).assign(&v);

            // Deflate matrix
            if eigenvalue.abs() > 1e-10 {
                for i in 0..n {
                    for j in 0..n {
                        remaining_matrix[[i, j]] -= eigenvalue * v[i] * v[j];
                    }
                }
            }
        }

        // Extend to full matrices
        let mut full_eigenvalues = Array1::zeros(n);
        let mut full_eigenvectors = Array2::zeros((n, n));

        for i in 0..n.min(self.target_dimensions) {
            full_eigenvalues[i] = eigenvalues[i];
            full_eigenvectors
                .column_mut(i)
                .assign(&eigenvectors.column(i));
        }

        // Fill remaining with identity
        for i in n.min(self.target_dimensions)..n {
            let mut v = Array1::zeros(n);
            v[i] = 1.0;
            full_eigenvectors.column_mut(i).assign(&v);
        }

        Ok((full_eigenvalues, full_eigenvectors))
    }

    fn find_neighbors(&self, data: &Array2<f64>, point: usize, eps: f64) -> Vec<usize> {
        let mut neighbors = Vec::new();
        for i in 0..data.nrows() {
            if i != point {
                let dist = self.euclidean_distance(&data.row(point), &data.row(i));
                if dist <= eps {
                    neighbors.push(i);
                }
            }
        }
        neighbors
    }

    fn expand_cluster(
        &self,
        data: &Array2<f64>,
        point: usize,
        neighbors: &[usize],
        cluster_id: usize,
        eps: f64,
        min_samples: usize,
        labels: &mut [usize],
    ) {
        labels[point] = cluster_id;
        let mut seed_set = neighbors.to_vec();
        let mut i = 0;

        while i < seed_set.len() {
            let q = seed_set[i];

            if labels[q] == usize::MAX - 1 {
                // Change noise to border point
                labels[q] = cluster_id;
            } else if labels[q] == usize::MAX {
                // Unclassified
                labels[q] = cluster_id;
                let q_neighbors = self.find_neighbors(data, q, eps);

                if q_neighbors.len() >= min_samples {
                    for &neighbor in &q_neighbors {
                        if !seed_set.contains(&neighbor) {
                            seed_set.push(neighbor);
                        }
                    }
                }
            }

            i += 1;
        }
    }

    fn cluster_distance(
        &self,
        cluster1: &[usize],
        cluster2: &[usize],
        distance_matrix: &Array2<f64>,
    ) -> f64 {
        // Single linkage (minimum distance)
        let mut min_distance = f64::INFINITY;

        for &i in cluster1 {
            for &j in cluster2 {
                let dist = distance_matrix[[i, j]];
                if dist < min_distance {
                    min_distance = dist;
                }
            }
        }

        min_distance
    }
}

impl Default for MultiDimensionalVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Animated visualization for time-series or dynamic systems
#[derive(Debug, Clone)]
pub struct AnimatedVisualizer {
    /// Frame data
    pub frames: Vec<PhaseSpacePlot>,
    /// Animation settings
    pub animation_settings: AnimationSettings,
    /// Current frame index
    pub current_frame: usize,
}

impl AnimatedVisualizer {
    /// Create new animated visualizer
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            animation_settings: AnimationSettings::default(),
            current_frame: 0,
        }
    }

    /// Create animation from ODE solution
    pub fn create_animation_from_ode<F: crate::common::IntegrateFloat>(
        &mut self,
        ode_result: &ODEResult<F>,
        x_index: usize,
        y_index: usize,
        frames_per_time_unit: usize,
    ) -> IntegrateResult<()> {
        let n_points = ode_result.t.len();
        if n_points == 0 {
            return Err(IntegrateError::ValueError("Empty ODE result".to_string()));
        }

        let n_vars = ode_result.y[0].len();
        if x_index >= n_vars || y_index >= n_vars {
            return Err(IntegrateError::ValueError(
                "Variable index out of bounds".to_string(),
            ));
        }

        // Create frames with progressive trajectory buildup
        self.frames.clear();
        let frame_step =
            (n_points as f64 / (frames_per_time_unit * n_points) as f64).max(1.0) as usize;

        for frame_end in (frame_step..=n_points).step_by(frame_step) {
            let x: Vec<f64> = (0..frame_end)
                .map(|i| ode_result.y[i][x_index].to_f64().unwrap_or(0.0))
                .collect();

            let y: Vec<f64> = (0..frame_end)
                .map(|i| ode_result.y[i][y_index].to_f64().unwrap_or(0.0))
                .collect();

            let colors: Vec<f64> = (0..frame_end)
                .map(|i| ode_result.t[i].to_f64().unwrap_or(0.0))
                .collect();

            let mut metadata = PlotMetadata::default();
            metadata.title = format!("Animation Frame {}", self.frames.len() + 1);
            metadata.xlabel = format!("Variable {x_index}");
            metadata.ylabel = format!("Variable {y_index}");

            self.frames.push(PhaseSpacePlot {
                x,
                y,
                colors: Some(colors),
                metadata,
            });
        }

        Ok(())
    }

    /// Get current frame
    pub fn current_frame(&self) -> Option<&PhaseSpacePlot> {
        self.frames.get(self.current_frame)
    }

    /// Advance to next frame
    pub fn next_frame(&mut self) -> bool {
        if self.current_frame + 1 < self.frames.len() {
            self.current_frame += 1;
            true
        } else if self.animation_settings.loop_animation {
            self.current_frame = 0;
            true
        } else {
            false
        }
    }

    /// Go to previous frame
    pub fn previous_frame(&mut self) -> bool {
        if self.current_frame > 0 {
            self.current_frame -= 1;
            true
        } else if self.animation_settings.loop_animation {
            self.current_frame = self.frames.len().saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Reset to first frame
    pub fn reset(&mut self) {
        self.current_frame = 0;
    }
}

impl Default for AnimatedVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Create multi-dimensional visualization
pub fn advanced_visualization(
    data: &Array2<f64>,
    method: DimensionReductionMethod,
    target_dims: usize,
) -> IntegrateResult<HighDimensionalPlot> {
    let visualizer = MultiDimensionalVisualizer {
        reduction_method: method,
        target_dimensions: target_dims,
        clustering_method: ClusteringMethod::None,
    };

    visualizer.visualize_high_dimensional_data(data, None)
}

/// Create interactive 3D visualization
pub fn advanced_interactive_3d(
    data: &Array2<f64>,
    labels: Option<&Array1<usize>>,
) -> IntegrateResult<HighDimensionalPlot> {
    let visualizer = MultiDimensionalVisualizer {
        reduction_method: DimensionReductionMethod::PCA,
        target_dimensions: 3,
        clustering_method: ClusteringMethod::KMeans { k: 3 },
    };

    visualizer.visualize_high_dimensional_data(data, labels)
}
