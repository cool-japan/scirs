//! Sparse distance matrix support for large datasets
//!
//! This module provides efficient representations and algorithms for working
//! with sparse distance matrices, particularly useful for high-dimensional
//! data where most pairwise distances are zero or very large.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::hierarchy::{LinkageMethod, Metric};

/// Sparse distance matrix using coordinate format (COO)
///
/// Stores only non-zero distances to save memory for sparse datasets.
#[derive(Debug, Clone)]
pub struct SparseDistanceMatrix<F: Float> {
    /// Row indices of non-zero distances
    rows: Vec<usize>,
    /// Column indices of non-zero distances  
    cols: Vec<usize>,
    /// Non-zero distance values
    data: Vec<F>,
    /// Number of samples (matrix dimension)
    n_samples: usize,
    /// Default value for unspecified distances (typically 0.0 or infinity)
    default_value: F,
}

impl<F: Float + FromPrimitive> SparseDistanceMatrix<F> {
    /// Create a new sparse distance matrix
    pub fn new(n_samples: usize, default_value: F) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            data: Vec::new(),
            n_samples,
            default_value,
        }
    }

    /// Create a sparse distance matrix from a dense matrix, keeping only values above threshold
    pub fn from_dense(dense: ArrayView2<F>, threshold: F) -> Self {
        let n_samples = dense.shape()[0];
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut data = Vec::new();

        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let distance = dense[[i, j]];
                if distance > threshold {
                    rows.push(i);
                    cols.push(j);
                    data.push(distance);
                }
            }
        }

        Self {
            rows,
            cols,
            data,
            n_samples,
            default_value: F::zero(),
        }
    }

    /// Add a distance entry to the sparse matrix
    pub fn add_distance(&mut self, i: usize, j: usize, distance: F) -> Result<()> {
        if i >= self.n_samples || j >= self.n_samples {
            return Err(ClusteringError::InvalidInput("Index out of bounds".into()));
        }

        // Ensure i < j for upper triangular storage
        let (row, col) = if i < j { (i, j) } else { (j, i) };

        // Check if this edge already exists
        for idx in 0..self.rows.len() {
            if self.rows[idx] == row && self.cols[idx] == col {
                // Update existing entry with the shorter distance
                if distance < self.data[idx] {
                    self.data[idx] = distance;
                }
                return Ok(());
            }
        }

        // Add new entry
        self.rows.push(row);
        self.cols.push(col);
        self.data.push(distance);

        Ok(())
    }

    /// Get the distance between two points
    pub fn get_distance(&self, i: usize, j: usize) -> F {
        if i == j {
            return F::zero();
        }

        let (row, col) = if i < j { (i, j) } else { (j, i) };

        // Linear search through stored values (could be optimized with sorted storage)
        for idx in 0..self.rows.len() {
            if self.rows[idx] == row && self.cols[idx] == col {
                return self.data[idx];
            }
        }

        self.default_value
    }

    /// Get all neighbors within a given distance threshold
    pub fn neighbors_within_distance(&self, point: usize, maxdistance: F) -> Vec<(usize, F)> {
        let mut neighbors = Vec::new();

        // Check all stored distances involving this point
        for idx in 0..self.rows.len() {
            let (neighbor, distance) = if self.rows[idx] == point {
                (self.cols[idx], self.data[idx])
            } else if self.cols[idx] == point {
                (self.rows[idx], self.data[idx])
            } else {
                continue;
            };

            if distance <= maxdistance {
                neighbors.push((neighbor, distance));
            }
        }

        neighbors
    }

    /// Get the k nearest neighbors for a point
    pub fn k_nearest_neighbors(&self, point: usize, k: usize) -> Vec<(usize, F)> {
        let mut all_neighbors = Vec::new();

        // Collect all neighbors
        for idx in 0..self.rows.len() {
            let (neighbor, distance) = if self.rows[idx] == point {
                (self.cols[idx], self.data[idx])
            } else if self.cols[idx] == point {
                (self.rows[idx], self.data[idx])
            } else {
                continue;
            };

            all_neighbors.push((neighbor, distance));
        }

        // Sort by distance and take k nearest
        all_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("Operation failed"));
        all_neighbors.truncate(k);

        all_neighbors
    }

    /// Convert to a dense distance matrix (use with caution for large matrices)
    pub fn to_dense(&self) -> Array2<F> {
        let mut dense = Array2::from_elem((self.n_samples, self.n_samples), self.default_value);

        // Set diagonal to zero
        for i in 0..self.n_samples {
            dense[[i, i]] = F::zero();
        }

        // Fill in stored distances (both upper and lower triangular)
        for idx in 0..self.rows.len() {
            let i = self.rows[idx];
            let j = self.cols[idx];
            let distance = self.data[idx];

            dense[[i, j]] = distance;
            dense[[j, i]] = distance;
        }

        dense
    }

    /// Get the number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Get the sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f64 {
        let total_entries = self.n_samples * (self.n_samples - 1) / 2;
        1.0 - (self.nnz() as f64 / total_entries as f64)
    }

    /// Get the number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
}

/// Sparse hierarchical clustering using minimal spanning tree approach
///
/// This algorithm is particularly efficient for sparse distance matrices
/// where most distances are infinite or very large.
pub struct SparseHierarchicalClustering<F: Float> {
    sparse_matrix: SparseDistanceMatrix<F>,
    linkage_method: LinkageMethod,
}

impl<F: Float + FromPrimitive + Debug + PartialOrd> SparseHierarchicalClustering<F> {
    /// Create a new sparse hierarchical clustering instance
    pub fn new(sparse_matrix: SparseDistanceMatrix<F>, linkage_method: LinkageMethod) -> Self {
        Self {
            sparse_matrix,
            linkage_method,
        }
    }

    /// Perform hierarchical clustering using Prim's algorithm for MST
    pub fn fit(&self) -> Result<Array2<F>> {
        let n_samples = self.sparse_matrix.n_samples();

        if n_samples < 2 {
            return Err(ClusteringError::InvalidInput(
                "Need at least 2 samples for clustering".into(),
            ));
        }

        // Build a minimal spanning tree
        let mst_edges = self.minimum_spanning_tree()?;

        // Convert MST to linkage matrix based on the chosen method
        self.mst_to_linkage(mst_edges)
    }

    /// Build minimal spanning tree using Prim's algorithm
    fn minimum_spanning_tree(&self) -> Result<Vec<(usize, usize, F)>> {
        let n_samples = self.sparse_matrix.n_samples();
        let mut mst_edges = Vec::new();
        let mut visited = vec![false; n_samples];
        let mut min_edge: HashMap<usize, (usize, F)> = HashMap::new();

        // Start with vertex 0
        visited[0] = true;

        // Initialize edges from vertex 0
        for neighbor_idx in 0..self.sparse_matrix.rows.len() {
            let (i, j) = (
                self.sparse_matrix.rows[neighbor_idx],
                self.sparse_matrix.cols[neighbor_idx],
            );
            let distance = self.sparse_matrix.data[neighbor_idx];

            if i == 0 && !visited[j] {
                min_edge.insert(j, (i, distance));
            } else if j == 0 && !visited[i] {
                min_edge.insert(i, (j, distance));
            }
        }

        // Build MST one edge at a time
        for _ in 1..n_samples {
            // Find minimum edge to unvisited vertex
            let mut min_dist = F::infinity();
            let mut min_vertex = 0;
            let mut min_parent = 0;

            for (&vertex, &(parent, distance)) in &min_edge {
                if !visited[vertex] && distance < min_dist {
                    min_dist = distance;
                    min_vertex = vertex;
                    min_parent = parent;
                }
            }

            if min_dist == F::infinity() {
                // Disconnected graph - use default distance
                min_dist = self.sparse_matrix.default_value;
            }

            // Add edge to MST
            mst_edges.push((min_parent, min_vertex, min_dist));
            visited[min_vertex] = true;

            // Update edges from the newly added vertex
            for neighbor_idx in 0..self.sparse_matrix.rows.len() {
                let (i, j) = (
                    self.sparse_matrix.rows[neighbor_idx],
                    self.sparse_matrix.cols[neighbor_idx],
                );
                let distance = self.sparse_matrix.data[neighbor_idx];

                let (from_vertex, to_vertex) = if i == min_vertex && !visited[j] {
                    (i, j)
                } else if j == min_vertex && !visited[i] {
                    (j, i)
                } else {
                    continue;
                };

                // Update minimum edge to to_vertex if this is better
                match min_edge.get(&to_vertex) {
                    Some(&(_, current_dist)) if distance < current_dist => {
                        min_edge.insert(to_vertex, (from_vertex, distance));
                    }
                    None => {
                        min_edge.insert(to_vertex, (from_vertex, distance));
                    }
                    _ => {}
                }
            }
        }

        Ok(mst_edges)
    }

    /// Convert MST edges to linkage matrix format
    fn mst_to_linkage(&self, mut mst_edges: Vec<(usize, usize, F)>) -> Result<Array2<F>> {
        let n_samples = self.sparse_matrix.n_samples();

        // Sort _edges by distance for single linkage, or process in MST order
        match self.linkage_method {
            LinkageMethod::Single => {
                // For single linkage, MST _edges directly give the dendrogram
                mst_edges.sort_by(|a, b| a.2.partial_cmp(&b.2).expect("Operation failed"));
            }
            _ => {
                // For other methods, we would need more complex processing
                // For now, treat as single linkage
            }
        }

        let mut linkage_matrix = Array2::zeros((n_samples - 1, 4));
        let mut cluster_map: HashMap<usize, usize> = HashMap::new();
        let mut next_cluster_id = n_samples;

        // Initialize cluster map (each point is its own cluster)
        for i in 0..n_samples {
            cluster_map.insert(i, i);
        }

        for (step, (i, j, distance)) in mst_edges.iter().enumerate() {
            let cluster_i = cluster_map[i];
            let cluster_j = cluster_map[j];

            // Record merge in linkage matrix
            linkage_matrix[[step, 0]] = F::from(cluster_i).expect("Failed to convert to float");
            linkage_matrix[[step, 1]] = F::from(cluster_j).expect("Failed to convert to float");
            linkage_matrix[[step, 2]] = *distance;
            linkage_matrix[[step, 3]] = F::from(2).expect("Failed to convert constant to float"); // Size starts at 2, would need to track actual sizes

            // Update cluster mapping
            cluster_map.insert(*i, next_cluster_id);
            cluster_map.insert(*j, next_cluster_id);
            next_cluster_id += 1;
        }

        Ok(linkage_matrix)
    }
}

/// Build a sparse k-nearest neighbor graph from dense data
#[allow(dead_code)]
pub fn sparse_knn_graph<F>(
    data: ArrayView2<F>,
    k: usize,
    metric: Metric,
) -> Result<SparseDistanceMatrix<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if k >= n_samples {
        return Err(ClusteringError::InvalidInput(
            "k must be less than number of samples".into(),
        ));
    }

    let mut sparse_matrix = SparseDistanceMatrix::new(n_samples, F::infinity());

    // For each point, find its k nearest neighbors
    for i in 0..n_samples {
        let mut distances: Vec<(usize, F)> = Vec::new();

        // Calculate distances to all other points
        for j in 0..n_samples {
            if i == j {
                continue;
            }

            let dist = match metric {
                Metric::Euclidean => {
                    let mut sum = F::zero();
                    for k in 0..n_features {
                        let diff = data[[i, k]] - data[[j, k]];
                        sum = sum + diff * diff;
                    }
                    sum.sqrt()
                }
                Metric::Manhattan => {
                    let mut sum = F::zero();
                    for k in 0..n_features {
                        let diff = (data[[i, k]] - data[[j, k]]).abs();
                        sum = sum + diff;
                    }
                    sum
                }
                Metric::Chebyshev => {
                    let mut max_diff = F::zero();
                    for k in 0..n_features {
                        let diff = (data[[i, k]] - data[[j, k]]).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                    max_diff
                }
                Metric::Cosine => {
                    let mut dot = F::zero();
                    let mut norm_i = F::zero();
                    let mut norm_j = F::zero();
                    for k in 0..n_features {
                        let vi = data[[i, k]];
                        let vj = data[[j, k]];
                        dot = dot + vi * vj;
                        norm_i = norm_i + vi * vi;
                        norm_j = norm_j + vj * vj;
                    }
                    let norm_prod = (norm_i * norm_j).sqrt();
                    if norm_prod
                        < F::from_f64(1e-10).ok_or_else(|| {
                            ClusteringError::InvalidInput("float conversion failed".into())
                        })?
                    {
                        F::one()
                    } else {
                        F::one() - dot / norm_prod
                    }
                }
                Metric::Correlation => {
                    let n_f = F::from_usize(n_features).ok_or_else(|| {
                        ClusteringError::InvalidInput("float conversion failed".into())
                    })?;
                    let mut mean_i = F::zero();
                    let mut mean_j = F::zero();
                    for k in 0..n_features {
                        mean_i = mean_i + data[[i, k]];
                        mean_j = mean_j + data[[j, k]];
                    }
                    mean_i = mean_i / n_f;
                    mean_j = mean_j / n_f;

                    let mut numerator = F::zero();
                    let mut denom_i = F::zero();
                    let mut denom_j = F::zero();
                    for k in 0..n_features {
                        let di = data[[i, k]] - mean_i;
                        let dj = data[[j, k]] - mean_j;
                        numerator = numerator + di * dj;
                        denom_i = denom_i + di * di;
                        denom_j = denom_j + dj * dj;
                    }
                    let denom = (denom_i * denom_j).sqrt();
                    if denom
                        < F::from_f64(1e-10).ok_or_else(|| {
                            ClusteringError::InvalidInput("float conversion failed".into())
                        })?
                    {
                        F::zero()
                    } else {
                        F::one() - numerator / denom
                    }
                }
            };

            distances.push((j, dist));
        }

        // Sort by distance and keep k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("Operation failed"));
        distances.truncate(k);

        // Add to sparse matrix
        for (neighbor, distance) in distances {
            sparse_matrix.add_distance(i, neighbor, distance)?;
        }
    }

    Ok(sparse_matrix)
}

/// Build a sparse epsilon-neighborhood graph from dense data
#[allow(dead_code)]
pub fn sparse_epsilon_graph<F>(
    data: ArrayView2<F>,
    epsilon: F,
    metric: Metric,
) -> Result<SparseDistanceMatrix<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    let mut sparse_matrix = SparseDistanceMatrix::new(n_samples, F::infinity());

    // For each pair of points, check if within epsilon distance
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let dist = match metric {
                Metric::Euclidean => {
                    let mut sum = F::zero();
                    for k in 0..n_features {
                        let diff = data[[i, k]] - data[[j, k]];
                        sum = sum + diff * diff;
                    }
                    sum.sqrt()
                }
                Metric::Manhattan => {
                    let mut sum = F::zero();
                    for k in 0..n_features {
                        let diff = (data[[i, k]] - data[[j, k]]).abs();
                        sum = sum + diff;
                    }
                    sum
                }
                Metric::Chebyshev => {
                    let mut max_diff = F::zero();
                    for k in 0..n_features {
                        let diff = (data[[i, k]] - data[[j, k]]).abs();
                        if diff > max_diff {
                            max_diff = diff;
                        }
                    }
                    max_diff
                }
                Metric::Cosine => {
                    let mut dot = F::zero();
                    let mut norm_i = F::zero();
                    let mut norm_j = F::zero();
                    for k in 0..n_features {
                        let vi = data[[i, k]];
                        let vj = data[[j, k]];
                        dot = dot + vi * vj;
                        norm_i = norm_i + vi * vi;
                        norm_j = norm_j + vj * vj;
                    }
                    let norm_prod = (norm_i * norm_j).sqrt();
                    if norm_prod
                        < F::from_f64(1e-10).ok_or_else(|| {
                            ClusteringError::InvalidInput("float conversion failed".into())
                        })?
                    {
                        F::one()
                    } else {
                        F::one() - dot / norm_prod
                    }
                }
                Metric::Correlation => {
                    let n_f = F::from_usize(n_features).ok_or_else(|| {
                        ClusteringError::InvalidInput("float conversion failed".into())
                    })?;
                    let mut mean_i = F::zero();
                    let mut mean_j = F::zero();
                    for k in 0..n_features {
                        mean_i = mean_i + data[[i, k]];
                        mean_j = mean_j + data[[j, k]];
                    }
                    mean_i = mean_i / n_f;
                    mean_j = mean_j / n_f;

                    let mut numerator = F::zero();
                    let mut denom_i = F::zero();
                    let mut denom_j = F::zero();
                    for k in 0..n_features {
                        let di = data[[i, k]] - mean_i;
                        let dj = data[[j, k]] - mean_j;
                        numerator = numerator + di * dj;
                        denom_i = denom_i + di * di;
                        denom_j = denom_j + dj * dj;
                    }
                    let denom = (denom_i * denom_j).sqrt();
                    if denom
                        < F::from_f64(1e-10).ok_or_else(|| {
                            ClusteringError::InvalidInput("float conversion failed".into())
                        })?
                    {
                        F::zero()
                    } else {
                        F::one() - numerator / denom
                    }
                }
            };

            if dist <= epsilon {
                sparse_matrix.add_distance(i, j, dist)?;
            }
        }
    }

    Ok(sparse_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_sparse_distance_matrix_creation() {
        let sparse_matrix = SparseDistanceMatrix::<f64>::new(5, 0.0);
        assert_eq!(sparse_matrix.n_samples(), 5);
        assert_eq!(sparse_matrix.nnz(), 0);
        assert_eq!(sparse_matrix.sparsity(), 1.0);
    }

    #[test]
    fn test_sparse_distance_matrix_add_distance() {
        let mut sparse_matrix = SparseDistanceMatrix::new(3, 0.0);

        sparse_matrix
            .add_distance(0, 1, 2.0)
            .expect("Operation failed");
        sparse_matrix
            .add_distance(1, 2, 3.0)
            .expect("Operation failed");

        assert_eq!(sparse_matrix.get_distance(0, 1), 2.0);
        assert_eq!(sparse_matrix.get_distance(1, 0), 2.0); // Symmetric
        assert_eq!(sparse_matrix.get_distance(1, 2), 3.0);
        assert_eq!(sparse_matrix.get_distance(0, 2), 0.0); // Default value
        assert_eq!(sparse_matrix.nnz(), 2);
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense =
            Array2::from_shape_vec((3, 3), vec![0.0, 1.0, 5.0, 1.0, 0.0, 2.0, 5.0, 2.0, 0.0])
                .expect("Operation failed");

        let sparse = SparseDistanceMatrix::from_dense(dense.view(), 1.5);

        // Should include distances > 1.5: (0,2)=5.0 and (1,2)=2.0
        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.get_distance(0, 2), 5.0);
        assert_eq!(sparse.get_distance(1, 2), 2.0);
        assert_eq!(sparse.get_distance(0, 1), 0.0); // Below threshold
    }

    #[test]
    fn test_neighbors_within_distance() {
        let mut sparse_matrix = SparseDistanceMatrix::new(4, f64::INFINITY);

        sparse_matrix
            .add_distance(0, 1, 1.0)
            .expect("Operation failed");
        sparse_matrix
            .add_distance(0, 2, 2.5)
            .expect("Operation failed");
        sparse_matrix
            .add_distance(0, 3, 0.5)
            .expect("Operation failed");

        let neighbors = sparse_matrix.neighbors_within_distance(0, 2.0);

        // Should find neighbors at distances 1.0 and 0.5 (both <= 2.0)
        assert_eq!(neighbors.len(), 2);

        let mut neighbor_distances: Vec<f64> = neighbors.iter().map(|(_, d)| *d).collect();
        neighbor_distances.sort_by(|a, b| a.partial_cmp(b).expect("Operation failed"));
        assert_eq!(neighbor_distances, vec![0.5, 1.0]);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let mut sparse_matrix = SparseDistanceMatrix::new(5, f64::INFINITY);

        sparse_matrix
            .add_distance(0, 1, 3.0)
            .expect("Operation failed");
        sparse_matrix
            .add_distance(0, 2, 1.0)
            .expect("Operation failed");
        sparse_matrix
            .add_distance(0, 3, 2.0)
            .expect("Operation failed");
        sparse_matrix
            .add_distance(0, 4, 4.0)
            .expect("Operation failed");

        let knn = sparse_matrix.k_nearest_neighbors(0, 2);

        // Should get 2 nearest neighbors: points 2 (dist=1.0) and 3 (dist=2.0)
        assert_eq!(knn.len(), 2);
        assert_eq!(knn[0], (2, 1.0)); // Nearest
        assert_eq!(knn[1], (3, 2.0)); // Second nearest
    }

    #[test]
    fn test_sparse_knn_graph() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0])
            .expect("Operation failed");

        let sparse_graph =
            sparse_knn_graph(data.view(), 2, Metric::Euclidean).expect("Operation failed");

        // Each point should have 2 neighbors
        // Total edges = 4 points * 2 neighbors = 8, but some may be duplicates when symmetrized
        assert!(sparse_graph.nnz() > 0);
        assert!(sparse_graph.sparsity() > 0.0);
    }

    #[test]
    fn test_sparse_epsilon_graph() {
        let data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 5.0, 5.0])
            .expect("Operation failed");

        let sparse_graph =
            sparse_epsilon_graph(data.view(), 1.0, Metric::Euclidean).expect("Operation failed");

        // Points (0,0), (0.5,0), and (0,0.5) should be connected
        // Point (5,5) should be isolated
        assert!(sparse_graph.nnz() >= 3); // At least the close connections

        // Check specific connections
        assert!(sparse_graph.get_distance(0, 1) <= 1.0);
        assert!(sparse_graph.get_distance(0, 2) <= 1.0);
    }

    #[test]
    fn test_to_dense() {
        let mut sparse_matrix = SparseDistanceMatrix::new(3, f64::INFINITY);
        sparse_matrix
            .add_distance(0, 1, 2.0)
            .expect("Operation failed");
        sparse_matrix
            .add_distance(1, 2, 3.0)
            .expect("Operation failed");

        let dense = sparse_matrix.to_dense();

        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense[[0, 1]], 2.0);
        assert_eq!(dense[[1, 0]], 2.0); // Symmetric
        assert_eq!(dense[[1, 2]], 3.0);
        assert_eq!(dense[[2, 1]], 3.0); // Symmetric
        assert_eq!(dense[[0, 0]], 0.0); // Diagonal
        assert_eq!(dense[[0, 2]], f64::INFINITY); // Unconnected
    }

    #[test]
    fn test_sparse_knn_graph_chebyshev() {
        // data: 4 points in 2D
        let data = Array2::from_shape_vec((4, 2), vec![0.0_f64, 0.0, 1.0, 0.0, 0.0, 1.0, 5.0, 5.0])
            .expect("Shape error");

        let graph = sparse_knn_graph(data.view(), 2, Metric::Chebyshev)
            .expect("sparse_knn_graph Chebyshev failed");
        assert!(graph.nnz() > 0, "Chebyshev KNN graph should have edges");
        // Chebyshev distance between (0,0) and (1,0) is 1.0; should be connected.
        assert!(
            graph.get_distance(0, 1) > 0.0,
            "points 0 and 1 should be neighbours"
        );
    }

    #[test]
    fn test_sparse_knn_graph_cosine() {
        // Identical direction vectors → cosine distance 0.
        let data = Array2::from_shape_vec((4, 2), vec![1.0_f64, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0])
            .expect("Shape error");

        let graph = sparse_knn_graph(data.view(), 2, Metric::Cosine)
            .expect("sparse_knn_graph Cosine failed");
        assert!(graph.nnz() > 0, "Cosine KNN graph should have edges");
        // (1,0) and (2,0) have cosine distance 0 → nearest neighbour.
        assert_eq!(
            graph.get_distance(0, 1),
            0.0,
            "parallel vectors have cosine distance 0"
        );
    }

    #[test]
    fn test_sparse_knn_graph_correlation() {
        // Perfect positive correlation → correlation distance ≈ 0.
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0_f64, 2.0, 3.0, 4.0, // row 0
                2.0, 4.0, 6.0, 8.0, // row 1 — perfect positive correlation with row 0
                4.0, 3.0, 2.0, 1.0, // row 2 — perfect negative correlation with row 0
            ],
        )
        .expect("Shape error");

        let graph = sparse_knn_graph(data.view(), 2, Metric::Correlation)
            .expect("sparse_knn_graph Correlation failed");
        assert!(graph.nnz() > 0, "Correlation KNN graph should have edges");
        // Rows 0 and 1 are perfectly correlated → distance ≈ 0.
        let d01 = graph.get_distance(0, 1);
        assert!(
            d01 < 1e-9,
            "perfectly correlated rows have correlation distance ≈ 0, got {d01}"
        );
    }

    #[test]
    fn test_sparse_epsilon_graph_chebyshev() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0_f64, 0.0, 0.8, 0.0, 5.0, 5.0])
            .expect("Shape error");

        // epsilon = 1.0 under Chebyshev: distance(0,1) = max(0.8, 0.0) = 0.8 < 1.0 → connected.
        let graph = sparse_epsilon_graph(data.view(), 1.0, Metric::Chebyshev)
            .expect("sparse_epsilon_graph Chebyshev failed");
        assert!(
            graph.get_distance(0, 1) > 0.0,
            "points 0 and 1 should be connected under Chebyshev"
        );
        // Point 2 is far away → not connected to 0.
        assert_eq!(
            graph.get_distance(0, 2),
            f64::INFINITY,
            "distant point should be disconnected"
        );
    }

    #[test]
    fn test_sparse_epsilon_graph_cosine() {
        // Parallel vectors have cosine distance 0 → within any epsilon > 0.
        let data = Array2::from_shape_vec((3, 2), vec![1.0_f64, 0.0, 2.0, 0.0, 0.0, 1.0])
            .expect("Shape error");

        let graph = sparse_epsilon_graph(data.view(), 0.5, Metric::Cosine)
            .expect("sparse_epsilon_graph Cosine failed");
        // Rows 0 and 1 are parallel (distance 0) → connected.
        assert!(
            graph.get_distance(0, 1) < 0.5,
            "parallel vectors connected under cosine epsilon graph"
        );
        // Row 2 is orthogonal to rows 0/1 (distance 1.0) → not connected for epsilon 0.5.
        assert_eq!(
            graph.get_distance(0, 2),
            f64::INFINITY,
            "orthogonal vector should be disconnected"
        );
    }

    #[test]
    fn test_sparse_epsilon_graph_correlation() {
        // Perfectly correlated rows: distance ≈ 0 → within epsilon 0.01.
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0_f64, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 4.0, 3.0, 2.0, 1.0,
            ],
        )
        .expect("Shape error");

        let graph = sparse_epsilon_graph(data.view(), 0.01, Metric::Correlation)
            .expect("sparse_epsilon_graph Correlation failed");
        // Rows 0 and 1 are perfectly correlated → distance ≈ 0 < 0.01.
        let d01 = graph.get_distance(0, 1);
        assert!(
            d01 < 0.01,
            "perfectly correlated rows connected under correlation epsilon graph, got {d01}"
        );
    }
}
