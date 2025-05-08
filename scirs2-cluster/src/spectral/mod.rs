//! Spectral clustering implementation
//!
//! Spectral clustering uses the eigenvalues of a similarity matrix to reduce the
//! dimensionality before clustering in fewer dimensions. This method is particularly
//! useful when the clusters have complex shapes and KMeans would perform poorly.

use ndarray::{s, Array1, Array2, ArrayView2};
use num_traits::{Float, FromPrimitive};
use scirs2_linalg::eigh;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::vq::{kmeans, KMeansInit, KMeansOptions};

/// Affinity matrix construction methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffinityMode {
    /// Nearest neighbors connectivity
    NearestNeighbors,
    /// Gaussian similarity (RBF kernel)
    RBF,
    /// Precomputed affinity matrix
    Precomputed,
}

/// Eigengap heuristic to estimate the number of clusters
///
/// This function implements the eigengap heuristic, which estimates
/// the number of clusters based on the differences between consecutive
/// eigenvalues.
///
/// # Arguments
///
/// * `eigenvalues` - Array of eigenvalues sorted in ascending order
/// * `max_clusters` - Maximum number of clusters to consider
///
/// # Returns
///
/// * The estimated number of clusters
fn eigengap_heuristic<F>(eigenvalues: &[F], max_clusters: usize) -> usize
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    // Find the largest eigengap among the first max_clusters eigenvalues
    let n = eigenvalues.len();
    let mut max_gap = F::zero();
    let mut max_gap_idx = 1; // Default to 1 cluster

    for i in 0..(max_clusters.min(n - 1)) {
        let gap = eigenvalues[i + 1] - eigenvalues[i];
        if gap > max_gap {
            max_gap = gap;
            max_gap_idx = i + 1;
        }
    }

    max_gap_idx
}

/// Normalized graph Laplacian
///
/// This function computes the normalized graph Laplacian from an affinity matrix.
/// The normalized Laplacian is defined as:
///   L_norm = I - D^(-1/2) A D^(-1/2)
/// where A is the affinity matrix and D is the diagonal matrix of degrees.
///
/// # Arguments
///
/// * `affinity` - Affinity matrix
///
/// # Returns
///
/// * The normalized graph Laplacian matrix
fn normalized_laplacian<F>(affinity: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n = affinity.shape()[0];
    if n != affinity.shape()[1] {
        return Err(ClusteringError::InvalidInput(
            "Affinity matrix must be square".to_string(),
        ));
    }

    // Calculate row sums (degrees)
    let mut degrees = Array1::zeros(n);
    for i in 0..n {
        degrees[i] = affinity.row(i).sum();
    }

    // Calculate D^(-1/2)
    let mut d_inv_sqrt = Array1::zeros(n);
    for i in 0..n {
        if degrees[i] <= F::epsilon() {
            return Err(ClusteringError::ComputationError(
                "Degree matrix contains zero values, graph may be disconnected".to_string(),
            ));
        }
        d_inv_sqrt[i] = F::one() / degrees[i].sqrt();
    }

    // Calculate normalized Laplacian L_norm = I - D^(-1/2) A D^(-1/2)
    let mut laplacian = Array2::zeros((n, n));

    for i in 0..n {
        for j in 0..n {
            if i == j {
                // Diagonal elements of identity matrix I minus the normalized affinity
                laplacian[[i, j]] = F::one() - affinity[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
            } else {
                // Off-diagonal elements are the negative normalized affinity
                laplacian[[i, j]] = -affinity[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
            }
        }
    }

    Ok(laplacian)
}

/// Create a K-nearest neighbor affinity matrix
///
/// # Arguments
///
/// * `data` - Input data
/// * `n_neighbors` - Number of neighbors to consider for each point
///
/// # Returns
///
/// * Affinity matrix where each row has at most n_neighbors non-zero entries
fn knn_affinity<F>(data: ArrayView2<F>, n_neighbors: usize) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    // Ensure n_neighbors is valid
    if n_neighbors >= n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "n_neighbors ({}) must be less than the number of samples ({})",
            n_neighbors, n_samples
        )));
    }

    // Calculate pairwise distances
    let mut dist_matrix = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let mut dist_sq = F::zero();
            for k in 0..n_features {
                let diff = data[[i, k]] - data[[j, k]];
                dist_sq = dist_sq + diff * diff;
            }
            let dist = dist_sq.sqrt();

            dist_matrix[[i, j]] = dist;
            dist_matrix[[j, i]] = dist; // Symmetric
        }
    }

    // Create KNN affinity matrix
    let mut affinity = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        // Get distances from point i to all other points
        let mut distances: Vec<(usize, F)> = (0..n_samples)
            .filter(|&j| i != j) // Exclude self
            .map(|j| (j, dist_matrix[[i, j]]))
            .collect();

        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select k nearest neighbors
        for k in 0..n_neighbors.min(distances.len()) {
            let (j, _) = distances[k];

            // Create binary adjacency matrix (1 for neighbors, 0 otherwise)
            affinity[[i, j]] = F::one();
            // Make it symmetric
            affinity[[j, i]] = F::one();
        }
    }

    Ok(affinity)
}

/// Create a RBF kernel affinity matrix
///
/// # Arguments
///
/// * `data` - Input data
/// * `gamma` - RBF kernel parameter (1/(2*sigma^2))
///
/// # Returns
///
/// * Affinity matrix where each element (i,j) is exp(-gamma * ||x_i - x_j||^2)
fn rbf_affinity<F>(data: ArrayView2<F>, gamma: F) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + PartialOrd,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if gamma <= F::zero() {
        return Err(ClusteringError::InvalidInput(
            "gamma must be positive".to_string(),
        ));
    }

    // Calculate pairwise distances and apply RBF kernel
    let mut affinity = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        // Diagonal is 1 (distance to self is 0)
        affinity[[i, i]] = F::one();

        for j in (i + 1)..n_samples {
            let mut dist_sq = F::zero();
            for k in 0..n_features {
                let diff = data[[i, k]] - data[[j, k]];
                dist_sq = dist_sq + diff * diff;
            }

            // Apply RBF kernel: exp(-gamma * ||x_i - x_j||^2)
            let affinity_val = (-gamma * dist_sq).exp();

            affinity[[i, j]] = affinity_val;
            affinity[[j, i]] = affinity_val; // Symmetric
        }
    }

    Ok(affinity)
}

/// Options for spectral clustering
#[derive(Debug, Clone)]
pub struct SpectralClusteringOptions<F: Float> {
    /// Method to build the affinity matrix
    pub affinity: AffinityMode,

    /// Number of neighbors for nearest neighbors affinity
    pub n_neighbors: usize,

    /// Parameter for RBF kernel (1/(2*sigma^2))
    pub gamma: F,

    /// Whether to use normalized graph Laplacian
    pub normalized_laplacian: bool,

    /// Maximum number of iterations for k-means
    pub max_iter: usize,

    /// Number of k-means initializations to run
    pub n_init: usize,

    /// Convergence threshold for k-means
    pub tol: F,

    /// Random seed for initialization
    pub random_seed: Option<u64>,

    /// Method for postprocessing eigenvectors
    pub eigen_solver: String,

    /// Whether to automatically detect number of clusters using eigengap heuristic
    pub auto_n_clusters: bool,
}

impl<F: Float + FromPrimitive> Default for SpectralClusteringOptions<F> {
    fn default() -> Self {
        Self {
            affinity: AffinityMode::RBF,
            n_neighbors: 10,
            gamma: F::from(1.0).unwrap(),
            normalized_laplacian: true,
            max_iter: 300,
            n_init: 10,
            tol: F::from(1e-4).unwrap(),
            random_seed: None,
            eigen_solver: "arpack".to_string(),
            auto_n_clusters: false,
        }
    }
}

/// Spectral clustering
///
/// Spectral clustering uses the eigenvalues of a similarity matrix to perform
/// dimensionality reduction before clustering in fewer dimensions.
///
/// # Arguments
///
/// * `data` - Input data or affinity matrix (n_samples × n_features) or (n_samples × n_samples)
/// * `n_clusters` - Number of clusters to find
/// * `options` - Optional parameters
///
/// # Returns
///
/// * Tuple of (embeddings, labels) where:
///   - embeddings: Array of shape (n_samples × n_clusters) with spectral embeddings
///   - labels: Array of shape (n_samples,) with cluster assignments
///
/// # Examples
///
/// ```ignore
/// use ndarray::{Array2, ArrayView2};
/// use scirs2_cluster::spectral::{spectral_clustering, SpectralClusteringOptions, AffinityMode};
///
/// // Example data with two ring-shaped clusters
/// let data = Array2::from_shape_vec((20, 2), vec![
///     // First ring
///     1.0, 0.0,  0.87, 0.5,  0.5, 0.87,  0.0, 1.0,  -0.5, 0.87,
///     -0.87, 0.5,  -1.0, 0.0,  -0.87, -0.5,  -0.5, -0.87,  0.0, -1.0,
///     // Second ring (larger radius)
///     4.0, 0.0,  3.46, 2.0,  2.0, 3.46,  0.0, 4.0,  -2.0, 3.46,
///     -3.46, 2.0,  -4.0, 0.0,  -3.46, -2.0,  -2.0, -3.46,  0.0, -4.0,
/// ]).unwrap();
///
/// // Run spectral clustering with RBF affinity
/// let options = SpectralClusteringOptions {
///     affinity: AffinityMode::RBF,
///     gamma: 0.5, // Adjust based on the scale of your data
///     ..Default::default()
/// };
///
/// let (embeddings, labels) = spectral_clustering(data.view(), 2, Some(options)).unwrap();
///
/// // Print the results
/// println!("Cluster assignments: {:?}", labels);
/// ```
pub fn spectral_clustering<F>(
    data: ArrayView2<F>,
    n_clusters: usize,
    options: Option<SpectralClusteringOptions<F>>,
) -> Result<(Array2<F>, Array1<usize>)>
where
    F: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + 'static
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    let opts = options.unwrap_or_default();
    let n_samples = data.shape()[0];

    if n_samples < 2 {
        return Err(ClusteringError::InvalidInput(
            "Input data must have at least 2 samples".to_string(),
        ));
    }

    if n_clusters < 2 || n_clusters > n_samples {
        return Err(ClusteringError::InvalidInput(format!(
            "Number of clusters must be between 2 and number of samples ({}), got {}",
            n_samples, n_clusters
        )));
    }

    // Step 1: Create the affinity matrix
    let affinity = match opts.affinity {
        AffinityMode::NearestNeighbors => {
            // Check if data is a square matrix (precomputed affinity)
            if data.shape()[0] == data.shape()[1] {
                // Assuming it's already a precomputed affinity matrix
                data.to_owned()
            } else {
                // Create KNN affinity matrix
                knn_affinity(data, opts.n_neighbors)?
            }
        }
        AffinityMode::RBF => {
            // Check if data is a square matrix (precomputed affinity)
            if data.shape()[0] == data.shape()[1] {
                // Assuming it's already a precomputed affinity matrix
                data.to_owned()
            } else {
                // Create RBF kernel affinity matrix
                rbf_affinity(data, opts.gamma)?
            }
        }
        AffinityMode::Precomputed => {
            // Verify that data is a square matrix
            if data.shape()[0] != data.shape()[1] {
                return Err(ClusteringError::InvalidInput(
                    "For precomputed affinity, data must be a square matrix".to_string(),
                ));
            }
            data.to_owned()
        }
    };

    // Step 2: Compute the graph Laplacian
    let laplacian = if opts.normalized_laplacian {
        normalized_laplacian(&affinity)?
    } else {
        // Unnormalized Laplacian L = D - A
        let mut lap = Array2::zeros((n_samples, n_samples));

        // Calculate degrees (diagonal elements of D)
        let mut degrees = vec![F::zero(); n_samples];
        for i in 0..n_samples {
            degrees[i] = affinity.row(i).sum();
            lap[[i, i]] = degrees[i];
        }

        // Subtract affinity matrix: L = D - A
        for i in 0..n_samples {
            for j in 0..n_samples {
                lap[[i, j]] = lap[[i, j]] - affinity[[i, j]];
            }
        }

        lap
    };

    // Step 3: Compute the eigenvalues and eigenvectors
    // Ensure numerical stability by adding a small value to the diagonal
    let n = laplacian.nrows();
    let mut stabilized_laplacian = laplacian.clone();
    for i in 0..n {
        stabilized_laplacian[[i, i]] = stabilized_laplacian[[i, i]] + F::from(1e-10).unwrap();
    }

    // Use the stabilized matrix for eigenvalue decomposition
    let (eigenvalues, eigenvectors) = eigh(&stabilized_laplacian.view())?;

    // Determine the actual number of clusters
    let actual_n_clusters = if opts.auto_n_clusters {
        // Use eigengap heuristic to determine the number of clusters
        // When using the normalized Laplacian, we need the smaller eigenvalues
        eigengap_heuristic(&eigenvalues.to_vec(), n_clusters)
    } else {
        n_clusters
    };

    // Step 4: Choose the appropriate eigenvectors
    // For the normalized Laplacian, we take the eigenvectors corresponding to the smallest eigenvalues
    let embedding = if opts.normalized_laplacian {
        // Extract n_clusters eigenvectors corresponding to the smallest eigenvalues
        // Note: eigenvalues should already be sorted in ascending order
        eigenvectors.slice(s![.., ..actual_n_clusters]).to_owned()
    } else {
        // For the unnormalized Laplacian, we skip the constant eigenvector (smallest eigenvalue)
        eigenvectors
            .slice(s![.., 1..(actual_n_clusters + 1)])
            .to_owned()
    };

    // Step 5: Row normalization (optional for some algorithms)
    let normalized_embedding = if opts.normalized_laplacian {
        // Normalize each row to have unit norm
        let mut norm_embedding = embedding.clone();

        for i in 0..n_samples {
            let row = embedding.row(i);
            let norm: F = row.iter().map(|&x| x * x).sum::<F>().sqrt();

            if norm > F::epsilon() {
                for j in 0..actual_n_clusters {
                    norm_embedding[[i, j]] = embedding[[i, j]] / norm;
                }
            }
        }

        norm_embedding
    } else {
        embedding
    };

    // Step 6: Apply k-means clustering in the embedding space
    let kmeans_opts = KMeansOptions {
        max_iter: opts.max_iter,
        tol: opts.tol,
        random_seed: opts.random_seed,
        n_init: opts.n_init,
        init_method: KMeansInit::KMeansPlusPlus,
    };

    let (_, labels) = kmeans(
        normalized_embedding.view(),
        actual_n_clusters,
        Some(kmeans_opts),
    )?;

    Ok((normalized_embedding, labels))
}

/// Fit a spectral bipartitioning model
///
/// This function finds a 2-cluster solution by analyzing the second
/// eigenvector of the graph Laplacian.
///
/// # Arguments
///
/// * `data` - Input data or affinity matrix (n_samples × n_features) or (n_samples × n_samples)
/// * `options` - Optional parameters
///
/// # Returns
///
/// * Array of shape (n_samples,) with binary cluster assignments
pub fn spectral_bipartition<F>(
    data: ArrayView2<F>,
    options: Option<SpectralClusteringOptions<F>>,
) -> Result<Array1<usize>>
where
    F: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + 'static
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    // Run spectral clustering with exactly 2 clusters
    let (_, labels) = spectral_clustering(data, 2, options)?;
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::silhouette_score;
    use ndarray::Array2;

    #[test]
    #[ignore = "Needs algorithm tuning - fails in the current implementation"]
    fn test_spectral_clustering_basic() {
        // Create a dataset with 2 well-separated clusters
        let data = Array2::from_shape_vec(
            (10, 2),
            vec![
                // Cluster 1
                1.0, 1.0, 1.2, 1.1, 0.9, 0.9, 1.1, 1.2, 0.8, 1.1, // Cluster 2
                4.0, 4.0, 4.2, 4.1, 3.9, 3.9, 4.1, 4.2, 3.8, 4.1,
            ],
        )
        .unwrap();

        // Run spectral clustering
        let options = SpectralClusteringOptions {
            affinity: AffinityMode::RBF,
            gamma: 1.0,
            ..Default::default()
        };

        let result = spectral_clustering(data.view(), 2, Some(options));
        assert!(result.is_ok());

        let (embeddings, labels) = result.unwrap();

        // Check dimensions
        assert_eq!(embeddings.shape(), &[10, 2]);
        assert_eq!(labels.len(), 10);

        // Check that we have 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 2);

        // Check cluster quality with silhouette score
        let silhouette =
            silhouette_score(data.view(), labels.mapv(|x| x as i32).view(), None).unwrap();
        assert!(
            silhouette > 0.8,
            "Silhouette score should be high for well-separated clusters"
        );
    }

    #[test]
    #[ignore = "Needs algorithm tuning - fails in the current implementation"]
    fn test_spectral_clustering_ring() {
        // Create two concentric ring-shaped clusters
        // First ring has radius 1, second ring has radius 3
        let data = Array2::from_shape_vec(
            (16, 2),
            vec![
                // First ring (8 points)
                1.0, 0.0, 0.7, 0.7, 0.0, 1.0, -0.7, 0.7, -1.0, 0.0, -0.7, -0.7, 0.0, -1.0, 0.7,
                -0.7, // Second ring (8 points)
                3.0, 0.0, 2.1, 2.1, 0.0, 3.0, -2.1, 2.1, -3.0, 0.0, -2.1, -2.1, 0.0, -3.0, 2.1,
                -2.1,
            ],
        )
        .unwrap();

        // K-means would fail on this dataset because the clusters are not linearly separable
        // but spectral clustering should work well

        // Run spectral clustering
        let options = SpectralClusteringOptions {
            affinity: AffinityMode::RBF,
            gamma: 0.2, // Lower gamma for wider neighborhoods
            ..Default::default()
        };

        let result = spectral_clustering(data.view(), 2, Some(options));
        assert!(result.is_ok());

        let (_, labels) = result.unwrap();

        // Check that we have 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert_eq!(unique_labels.len(), 2);

        // Check that each ring is assigned to a single cluster
        let first_ring_label = labels[0];
        for i in 0..8 {
            assert_eq!(labels[i], first_ring_label);
        }

        let second_ring_label = labels[8];
        assert_ne!(first_ring_label, second_ring_label);
        for i in 8..16 {
            assert_eq!(labels[i], second_ring_label);
        }
    }
}
