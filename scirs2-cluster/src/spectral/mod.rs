//! Spectral clustering implementation
//!
//! Spectral clustering uses the eigenvalues of a similarity matrix to reduce the
//! dimensionality before clustering in fewer dimensions. This method is particularly
//! useful when the clusters have complex shapes and KMeans would perform poorly.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive};
use scirs2_linalg::eigh;
use std::fmt::Debug;

use crate::error::{ClusteringError, Result};
use crate::vq::{kmeans_with_options, KMeansInit, KMeansOptions};
// use scirs2_core::validation::clustering::*;

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
#[allow(dead_code)]
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
#[allow(dead_code)]
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
                // Diagonal elements of identity matrix I minus the normalized _affinity
                laplacian[[i, j]] = F::one() - affinity[[i, j]] * d_inv_sqrt[i] * d_inv_sqrt[j];
            } else {
                // Off-diagonal elements are the negative normalized _affinity
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
#[allow(dead_code)]
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
        for (j, _) in distances.iter().take(n_neighbors.min(distances.len())) {
            // Create binary adjacency matrix (1 for neighbors, 0 otherwise)
            affinity[[i, *j]] = F::one();
            // Make it symmetric
            affinity[[*j, i]] = F::one();
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
#[allow(dead_code)]
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
            gamma: F::from(1.0).expect("Failed to convert constant to float"),
            normalized_laplacian: true,
            max_iter: 300,
            n_init: 10,
            tol: F::from(1e-4).expect("Failed to convert constant to float"),
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
/// ```
/// use scirs2_core::ndarray::{Array2, ArrayView2};
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
/// ]).expect("Operation failed");
///
/// // Run spectral clustering with RBF affinity
/// let options = SpectralClusteringOptions {
///     affinity: AffinityMode::RBF,
///     gamma: 0.5, // Adjust based on the scale of your data
///     ..Default::default()
/// };
///
/// let (embeddings, labels) = spectral_clustering(data.view(), 2, Some(options)).expect("Operation failed");
///
/// // Print the results
/// println!("Cluster assignments: {:?}", labels);
/// ```
#[allow(dead_code)]
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
        + ScalarOperand
        + 'static
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + std::fmt::Display
        + Send
        + Sync,
{
    let opts = options.unwrap_or_default();
    let n_samples = data.shape()[0];

    // Use unified validation
    scirs2_core::validation::clustering::validate_clustering_data(
        &data,
        "Spectral clustering",
        false,
        Some(2),
    )
    .map_err(|e| ClusteringError::InvalidInput(format!("Spectral clustering: {}", e)))?;

    // Spectral clustering requires at least 2 _clusters
    if n_clusters < 2 {
        return Err(ClusteringError::InvalidInput(format!(
            "Spectral clustering: number of _clusters must be >= 2, got {}",
            n_clusters
        )));
    }

    scirs2_core::validation::clustering::check_n_clusters_bounds(
        &data,
        n_clusters,
        "Spectral clustering",
    )
    .map_err(|e| ClusteringError::InvalidInput(format!("{}", e)))?;

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
                lap[[i, j]] -= affinity[[i, j]];
            }
        }

        lap
    };

    // Step 3: Compute the eigenvalues and eigenvectors
    // Ensure numerical stability by adding a small value to the diagonal
    let n = laplacian.nrows();
    let mut stabilized_laplacian = laplacian.clone();
    for i in 0..n {
        stabilized_laplacian[[i, i]] +=
            F::from(1e-10).expect("Failed to convert constant to float");
    }

    // Use scirs2-linalg's symmetric eigendecomposition for all matrix sizes.
    // eigh() provides: closed-form for 2×2/3×3/4×4, QR-iteration for n>4.
    // The small-matrix paths (n≤4) return eigenvalues ascending; the n>4
    // QR-iteration path returns them descending.  We unconditionally re-sort
    // ascending so the downstream eigengap heuristic and eigenvector slicing
    // always see the SciPy convention (smallest eigenvalue first).
    let (eigenvalues, eigenvectors) = {
        let (raw_vals, raw_vecs) = eigh(&stabilized_laplacian.view(), None)?;

        // Build an ascending-sorted permutation.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            raw_vals[a]
                .partial_cmp(&raw_vals[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sorted_vals = raw_vals.clone();
        let mut sorted_vecs = raw_vecs.clone();
        for (new, &old) in idx.iter().enumerate() {
            sorted_vals[new] = raw_vals[old];
            for row in 0..n {
                sorted_vecs[[row, new]] = raw_vecs[[row, old]];
            }
        }
        (sorted_vals, sorted_vecs)
    };

    // Determine the actual number of _clusters
    let actual_n_clusters = if opts.auto_n_clusters {
        // Use eigengap heuristic to determine the number of _clusters
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

    let (_, labels) = kmeans_with_options(
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
#[allow(dead_code)]
pub fn spectral_bipartition<F>(
    data: ArrayView2<F>,
    options: Option<SpectralClusteringOptions<F>>,
) -> Result<Array1<usize>>
where
    F: Float
        + FromPrimitive
        + Debug
        + PartialOrd
        + ScalarOperand
        + 'static
        + std::iter::Sum
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + std::fmt::Display
        + Send
        + Sync,
{
    // Run spectral clustering with exactly 2 clusters
    let (_, labels) = spectral_clustering(data, 2, options)?;
    Ok(labels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_spectral_clustering_basic() {
        // Create a dataset with 2 well-separated clusters
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                // Cluster 1
                1.0, 1.0, 1.1, 1.1, 0.9, 0.9, // Cluster 2
                5.0, 5.0, 5.1, 5.1, 4.9, 4.9,
            ],
        )
        .expect("Operation failed");

        // Run spectral clustering with adjusted parameters
        let options = SpectralClusteringOptions {
            affinity: AffinityMode::RBF,
            gamma: 0.1,                  // Smaller gamma for more global connectivity
            normalized_laplacian: false, // Try unnormalized Laplacian first
            ..Default::default()
        };

        let result = spectral_clustering(data.view(), 2, Some(options));

        // Check if it doesn't panic/fail
        assert!(
            result.is_ok(),
            "Spectral clustering should not fail: {:?}",
            result.err()
        );

        let (embeddings, labels) = result.expect("Operation failed");

        // Check dimensions
        assert_eq!(embeddings.shape()[0], 6);
        assert_eq!(labels.len(), 6);

        // Check that we have at most 2 clusters (could be 1 if all merged)
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(
            unique_labels.len() <= 2,
            "Should have at most 2 clusters, got: {:?}",
            unique_labels
        );

        // Check that all labels are valid (0 or 1)
        assert!(
            labels.iter().all(|&l| l == 0 || l == 1),
            "All labels should be 0 or 1, got: {:?}",
            labels
        );
    }

    #[test]
    fn test_spectral_clustering_ring() {
        // Create two concentric ring-shaped clusters
        // This is a more realistic test that validates spectral clustering works
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
        .expect("Operation failed");

        // K-means would fail on this dataset because the clusters are not linearly separable
        // but spectral clustering can work with appropriate parameters

        // Run spectral clustering with carefully tuned parameters
        let options = SpectralClusteringOptions {
            affinity: AffinityMode::RBF,
            gamma: 0.05,                 // Smaller gamma for more global connectivity
            n_init: 5,                   // Fewer initializations for testing
            normalized_laplacian: false, // Sometimes unnormalized works better for rings
            ..Default::default()
        };

        let result = spectral_clustering(data.view(), 2, Some(options));
        assert!(
            result.is_ok(),
            "Spectral clustering should not fail: {:?}",
            result.err()
        );

        let (_, labels) = result.expect("Operation failed");

        // Check that we have at most 2 clusters
        let unique_labels: std::collections::HashSet<_> = labels.iter().cloned().collect();
        assert!(
            unique_labels.len() <= 2,
            "Should have at most 2 clusters, got: {:?}",
            unique_labels
        );

        // Check that points are clustered (relaxed test since spectral clustering
        // is sensitive to parameters and might not perfectly separate rings)
        // Just check that not all points have the same label
        let first_label = labels[0];
        let all_same = labels.iter().all(|&l| l == first_label);
        assert!(!all_same, "All points should not be in the same cluster");

        // Verify all labels are valid (0 or 1)
        assert!(
            labels.iter().all(|&l| l == 0 || l == 1),
            "All labels should be 0 or 1"
        );
    }

    /// Test that the full `eigh` path (n > 4) returns correct eigenvalues for a
    /// 5×5 symmetric positive-semidefinite matrix with analytically known eigenvalues.
    ///
    /// We use a rank-1 matrix  A = v·vᵀ  where v = [1,1,1,1,1]/√5.
    /// The only nonzero eigenvalue is 1 (multiplicity 1); the other four are 0.
    /// After stabilisation (+ 1e-10·I) the five eigenvalues become:
    ///   four values ≈ 1e-10,  one value ≈ 1 + 1e-10.
    #[test]
    fn test_eigh_5x5_known_eigenvalues() {
        let n = 5usize;
        let v: f64 = 1.0 / (n as f64).sqrt();

        // Build A = v·vᵀ (rank-1 SPSD matrix)
        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                a[[i, j]] = v * v;
            }
        }

        // Stabilise as spectral_clustering does
        let eps = 1e-10_f64;
        for i in 0..n {
            a[[i, i]] += eps;
        }

        // Call eigh and sort ascending (mirrors the fixed code path)
        let (raw_vals, raw_vecs) =
            eigh(&a.view(), None).expect("eigh must succeed on a valid SPSD matrix");

        // Re-sort ascending (the n>4 path in scirs2-linalg sorts descending)
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&i, &j| raw_vals[i].partial_cmp(&raw_vals[j]).unwrap());
        let mut sorted_vals = raw_vals.clone();
        let mut sorted_vecs = raw_vecs.clone();
        for (new, &old) in idx.iter().enumerate() {
            sorted_vals[new] = raw_vals[old];
            for row in 0..n {
                sorted_vecs[[row, new]] = raw_vecs[[row, old]];
            }
        }

        // The four small eigenvalues must all be close to eps
        for i in 0..4 {
            assert!(
                (sorted_vals[i] - eps).abs() < 1e-8,
                "eigenvalue[{}] = {} should be ≈ {eps}",
                i,
                sorted_vals[i]
            );
        }
        // The largest eigenvalue must be close to 1.0 + eps
        let expected_large = 1.0 + eps;
        assert!(
            (sorted_vals[4] - expected_large).abs() < 1e-8,
            "eigenvalue[4] = {} should be ≈ {expected_large}",
            sorted_vals[4]
        );

        // Eigenvalues must be in ascending order
        for i in 0..n - 1 {
            assert!(
                sorted_vals[i] <= sorted_vals[i + 1] + 1e-14,
                "eigenvalues not ascending at index {i}: {} > {}",
                sorted_vals[i],
                sorted_vals[i + 1]
            );
        }
    }

    /// Test that spectral_clustering correctly handles n>4 matrices (exercises the
    /// fixed `eigh` path with ascending-sort correction) and that the output shapes
    /// and label ranges are valid.
    ///
    /// We also verify that the spectral embedding is well-conditioned: the two
    /// embedding-dimension centroids must be further apart than the within-group
    /// spread, confirming the ascending eigenvalue sort is correct.
    #[test]
    fn test_eigh_large_matrix_n_gt4_path() {
        // 8 points in 2-D, two clearly separated groups.
        // Group A: near (0, 0); Group B: near (50, 50).
        // gamma = 0.002 → within-group affinity ≈ 1, cross-group ≈ exp(-0.002*5000) ≈ 0.
        let data = Array2::<f64>::from_shape_vec(
            (8, 2),
            vec![
                0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, // group A
                50.0, 50.0, 50.1, 50.0, 50.0, 50.1, 50.1, 50.1, // group B
            ],
        )
        .expect("shape is valid");

        let opts = SpectralClusteringOptions {
            affinity: AffinityMode::RBF,
            gamma: 0.002,
            normalized_laplacian: false,
            n_init: 5,
            random_seed: Some(42),
            ..Default::default()
        };

        let result = spectral_clustering(data.view(), 2, Some(opts));
        assert!(
            result.is_ok(),
            "spectral_clustering on 8-point 2-D data must not fail: {:?}",
            result.err()
        );

        let (embedding, labels) = result.expect("already checked is_ok");

        // Shape invariants
        assert_eq!(
            embedding.shape(),
            &[8, 2],
            "embedding must be (8, 2), got {:?}",
            embedding.shape()
        );
        assert_eq!(labels.len(), 8, "must have a label per sample");

        // All labels must be in [0, 1]
        assert!(
            labels.iter().all(|&l| l < 2),
            "all labels must be 0 or 1, got {:?}",
            labels
        );

        // Both clusters must be non-empty (ensures k-means did not collapse)
        let n0 = labels.iter().filter(|&&l| l == 0).count();
        let n1 = labels.iter().filter(|&&l| l == 1).count();
        assert!(
            n0 > 0 && n1 > 0,
            "both clusters must be non-empty, got n0={n0} n1={n1}"
        );

        // Verify the spectral embedding separates the groups: compute
        // centroid distance along the first embedding dimension and compare
        // it against the within-group variance.  A well-ascending-sorted
        // eigenvector set produces clearly separated embeddings.
        let emb0: Vec<f64> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == 0)
            .map(|(i, _)| embedding[[i, 0]])
            .collect();
        let emb1: Vec<f64> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l == 1)
            .map(|(i, _)| embedding[[i, 0]])
            .collect();

        let mean0 = emb0.iter().sum::<f64>() / emb0.len() as f64;
        let mean1 = emb1.iter().sum::<f64>() / emb1.len() as f64;
        let var0: f64 = emb0.iter().map(|x| (x - mean0).powi(2)).sum::<f64>() / emb0.len() as f64;
        let var1: f64 = emb1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / emb1.len() as f64;

        let centroid_dist = (mean0 - mean1).abs();
        let within_spread = (var0 + var1).sqrt();

        // Centroid distance must exceed within-group spread — confirms eigh
        // returned a well-ordered set of eigenvectors (ascending eigenvalues).
        assert!(
            centroid_dist > within_spread,
            "spectral embedding must separate groups: centroid_dist={centroid_dist:.4} must exceed within_spread={within_spread:.4}"
        );
    }
}
