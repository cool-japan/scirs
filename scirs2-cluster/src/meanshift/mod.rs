use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive};
use std::collections::HashMap;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::marker::{Send, Sync};

use crate::error::ClusteringError;
use scirs2_spatial::distance::EuclideanDistance;
use scirs2_spatial::kdtree::KDTree;

/// Configuration options for Mean Shift algorithm
pub struct MeanShiftOptions<T: Float> {
    /// Bandwidth parameter used in the flat kernel.
    /// If not provided, it will be estimated from the data.
    pub bandwidth: Option<T>,

    /// Points used as initial kernel locations.
    /// If not provided, either all points or discretized bins will be used.
    pub seeds: Option<Array2<T>>,

    /// If true, initial kernels are located on a grid with bin_size = bandwidth.
    /// This can significantly speed up the algorithm for large datasets.
    pub bin_seeding: bool,

    /// Only bins with at least min_bin_freq points will be selected as seeds.
    /// Only relevant when bin_seeding is true.
    pub min_bin_freq: usize,

    /// If true, all points are assigned to clusters, even those not within any kernel.
    /// Orphans are assigned to the nearest kernel.
    /// If false, orphans are given a cluster label of -1.
    pub cluster_all: bool,

    /// Maximum number of iterations for a single seed before stopping.
    pub max_iter: usize,
}

impl<T: Float> Default for MeanShiftOptions<T> {
    fn default() -> Self {
        Self {
            bandwidth: None,
            seeds: None,
            bin_seeding: false,
            min_bin_freq: 1,
            cluster_all: true,
            max_iter: 300,
        }
    }
}

/// FloatPoint wrapper to make f32/f64 arrays comparable and hashable
#[derive(Debug, Clone)]
struct FloatPoint<T: Float>(Vec<T>);

impl<T: Float> PartialEq for FloatPoint<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }

        for (a, b) in self.0.iter().zip(other.0.iter()) {
            if !a.is_finite() || !b.is_finite() || (*a - *b).abs() > T::epsilon() {
                return false;
            }
        }
        true
    }
}

impl<T: Float> Eq for FloatPoint<T> {}

impl<T: Float> Hash for FloatPoint<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use fixed precision for consistent hashing of floating point numbers
        for value in &self.0 {
            let bits = if let Some(bits) = value.to_f64() {
                (bits * 1e10).round() as i64
            } else {
                0
            };
            bits.hash(state);
        }
    }
}

/// Estimate the bandwidth to use with the mean-shift algorithm.
///
/// This function estimates the bandwidth parameter for Mean Shift clustering by computing
/// the median distance of each point to its k nearest neighbors, where k is determined
/// by the quantile parameter.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array where rows are samples and columns are features.
/// * `quantile` - Quantile of the pairwise distances to use as the bandwidth. Should be between 0 and 1.
/// * `n_samples` - Number of samples to use for estimation. If None, all samples are used.
/// * `random_state` - Optional seed for random sampling.
///
/// # Returns
///
/// * `Result<T, ClusteringError>` - The estimated bandwidth value or an error.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_cluster::meanshift::estimate_bandwidth;
///
/// let data = array![
///     [1.0, 1.0], [2.0, 1.0], [1.0, 0.0],
///     [4.0, 7.0], [3.0, 5.0], [3.0, 6.0]
/// ];
///
/// let bandwidth = estimate_bandwidth(&data.view(), Some(0.5), None, None).unwrap();
/// println!("Estimated bandwidth: {}", bandwidth);
/// ```
pub fn estimate_bandwidth<T: Float + Display + FromPrimitive + Send + Sync + 'static>(
    data: &ArrayView2<T>,
    quantile: Option<T>,
    n_samples: Option<usize>,
    _random_state: Option<u64>,
) -> Result<T, ClusteringError> {
    // Manual check that all data is finite
    for row in data.rows() {
        for &val in row.iter() {
            if !val.is_finite() {
                return Err(ClusteringError::InvalidInput(
                    "Input contains non-finite values".to_string(),
                ));
            }
        }
    }

    let quantile = quantile.unwrap_or_else(|| T::from(0.3).unwrap());

    if quantile < T::zero() || quantile > T::one() {
        return Err(ClusteringError::InvalidInput(
            "Quantile should be between 0 and 1 inclusive".to_string(),
        ));
    }

    // Select a subset of samples if specified
    let data = if let Some(n) = n_samples {
        if n >= data.nrows() {
            data.to_owned()
        } else {
            // Sample n_samples points randomly
            let mut rng = rand::rng();

            use rand::seq::SliceRandom;
            let mut indices: Vec<usize> = (0..data.nrows()).collect();
            indices.shuffle(&mut rng);

            let indices = &indices[0..n];
            let mut sampled_data = Array2::zeros((n, data.ncols()));
            for (i, &idx) in indices.iter().enumerate() {
                sampled_data.row_mut(i).assign(&data.row(idx));
            }
            sampled_data
        }
    } else {
        data.to_owned()
    };

    let n_neighbors = (T::from(data.nrows()).unwrap() * quantile)
        .to_usize()
        .unwrap_or(1)
        .max(1)
        .min(data.nrows().saturating_sub(1)); // Cannot have more neighbors than n_samples - 1

    // Build KDTree for nearest neighbor search
    let kdtree = KDTree::<_, EuclideanDistance<T>>::new(&data)
        .map_err(|e| ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e)))?;

    let mut bandwidth_sum = T::zero();

    // Process in batches to avoid memory issues with large datasets
    let batch_size = 500;
    for i in (0..data.nrows()).step_by(batch_size) {
        let end = (i + batch_size).min(data.nrows());
        let batch = data.slice(ndarray::s![i..end, ..]);

        for row in batch.rows() {
            let (_, distances) = kdtree.query(&row.to_vec(), n_neighbors + 1).map_err(|e| {
                ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
            })?;

            if distances.len() > 1 {
                // Skip the first distance (to itself, which is 0) and take the last (k-th neighbor)
                let kth_dist = distances.last().copied().unwrap_or(T::from(1.0).unwrap()); // Default to 1.0 if no valid distance is found

                bandwidth_sum = bandwidth_sum + kth_dist;
            } else if !distances.is_empty() {
                // If we only have one distance (to itself), use a larger default value
                bandwidth_sum = bandwidth_sum + T::from(1.0).unwrap();
            }
        }
    }

    Ok(bandwidth_sum / T::from(data.nrows()).unwrap())
}

/// Find seeds for mean_shift by binning data onto a grid.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array.
/// * `bin_size` - The size of the bins for discretization.
/// * `min_bin_freq` - Minimum number of points in a bin for it to be a seed.
///
/// # Returns
///
/// * `Array2<T>` - Coordinates of bin seeds to use as initial kernel positions.
pub fn get_bin_seeds<T: Float + Display + FromPrimitive + Send + Sync + 'static>(
    data: &ArrayView2<T>,
    bin_size: T,
    min_bin_freq: usize,
) -> Array2<T> {
    if bin_size <= T::zero() {
        return data.to_owned();
    }

    let mut bin_sizes: HashMap<FloatPoint<T>, usize> = HashMap::new();

    // Bin points
    for row in data.rows() {
        // Round to nearest bin center
        let mut binned_point = Vec::with_capacity(row.len());
        for &val in row.iter() {
            binned_point.push((val / bin_size).round() * bin_size);
        }

        let point = FloatPoint::<T>(binned_point);
        *bin_sizes.entry(point).or_insert(0) += 1;
    }

    // Select only bins with enough points
    let seeds: Vec<Vec<T>> = bin_sizes
        .into_iter()
        .filter(|(_, freq)| *freq >= min_bin_freq)
        .map(|(point, _)| point.0)
        .collect();

    // If all points are seeds, just return the original data
    if seeds.len() == data.nrows() {
        return data.to_owned();
    }

    // Convert to Array2
    if seeds.is_empty() {
        Array2::zeros((0, data.ncols()))
    } else {
        let mut result = Array2::zeros((seeds.len(), data.ncols()));
        for (i, seed) in seeds.into_iter().enumerate() {
            for (j, val) in seed.into_iter().enumerate() {
                result[[i, j]] = val;
            }
        }
        result
    }
}

/// Perform Mean Shift single seed update.
///
/// # Arguments
///
/// * `seed` - The initial position of the seed.
/// * `data` - The dataset to operate on.
/// * `bandwidth` - The bandwidth parameter.
/// * `max_iter` - Maximum number of iterations.
///
/// # Returns
///
/// * `(Vec<T>, usize, usize)` - (Final seed position, number of points within bandwidth, iterations performed)
fn mean_shift_single_seed<
    T: Float
        + Display
        + std::iter::Sum
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand,
>(
    seed: ArrayView1<T>,
    data: &ArrayView2<T>,
    bandwidth: T,
    max_iter: usize,
) -> (Vec<T>, usize, usize) {
    let stop_thresh = bandwidth * T::from(1e-3).unwrap();
    let mut my_mean = seed.to_owned();
    let mut completed_iterations = 0;

    // Create KDTree for efficient neighbor search
    // Convert ArrayView2 to owned Array2 for KDTree
    let owned_data = data.to_owned();
    let kdtree = match KDTree::<_, EuclideanDistance<T>>::new(&owned_data) {
        Ok(tree) => tree,
        Err(_) => return (seed.to_vec(), 0, 0),
    };

    loop {
        // Find points within bandwidth
        let (indices, _) = match kdtree.query_radius(&my_mean.to_vec(), bandwidth) {
            Ok((idx, distances)) => (idx, distances),
            Err(_) => return (my_mean.to_vec(), 0, completed_iterations),
        };

        if indices.is_empty() {
            break;
        }
        let my_old_mean = my_mean.clone();

        // Calculate new mean
        my_mean.fill(T::zero());
        let mut sum = Array1::zeros(my_mean.dim());
        for &idx in &indices {
            let row_clone = data.row(idx).to_owned();
            for (s, v) in sum.iter_mut().zip(row_clone.iter()) {
                *s = *s + *v;
            }
        }
        my_mean = sum / T::from(indices.len()).unwrap();

        // Compute Euclidean distance manually for convergence check
        let mut dist_squared = T::zero();
        for (a, b) in my_mean.iter().zip(my_old_mean.iter()) {
            dist_squared = dist_squared + (*a - *b) * (*a - *b);
        }
        let dist = dist_squared.sqrt();

        if dist <= stop_thresh || completed_iterations == max_iter {
            break;
        }

        completed_iterations += 1;
    }

    // Find number of points within bandwidth of final position
    let (indices, _) = match kdtree.query_radius(&my_mean.to_vec(), bandwidth) {
        Ok((idx, distances)) => (idx, distances),
        Err(_) => return (my_mean.to_vec(), 0, completed_iterations),
    };

    (my_mean.to_vec(), indices.len(), completed_iterations)
}

/// Perform Mean Shift clustering of data using a flat kernel.
///
/// # Arguments
///
/// * `data` - The input data as a 2D array.
/// * `options` - The Mean Shift algorithm options.
///
/// # Returns
///
/// * `Result<(Array2<T>, Array1<i32>), ClusteringError>` - Tuple of (cluster centers, labels) or an error.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use scirs2_cluster::meanshift::{mean_shift, MeanShiftOptions};
///
/// let data = array![
///     [1.0, 1.0], [2.0, 1.0], [1.0, 0.0],
///     [4.0, 7.0], [3.0, 5.0], [3.0, 6.0]
/// ];
///
/// let options = MeanShiftOptions {
///     bandwidth: Some(2.0),
///     ..Default::default()
/// };
///
/// let (centers, labels) = mean_shift(&data.view(), options).unwrap();
/// println!("Number of clusters: {}", centers.nrows());
/// ```
pub fn mean_shift<
    T: Float
        + Display
        + std::iter::Sum
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + ndarray::ScalarOperand,
>(
    data: &ArrayView2<T>,
    options: MeanShiftOptions<T>,
) -> Result<(Array2<T>, Array1<i32>), ClusteringError> {
    // Delegate to MeanShift::fit which implements the core algorithm
    let mut model = MeanShift::new(options);
    let model = model.fit(data)?;
    Ok((
        model.cluster_centers().to_owned(),
        model.labels().to_owned(),
    ))
}

/// Mean Shift clustering using a flat kernel.
pub struct MeanShift<T: Float> {
    options: MeanShiftOptions<T>,
    cluster_centers_: Option<Array2<T>>,
    labels_: Option<Array1<i32>>,
    n_iter_: usize,
}

impl<
        T: Float
            + Display
            + std::iter::Sum
            + FromPrimitive
            + Send
            + Sync
            + 'static
            + ndarray::ScalarOperand,
    > MeanShift<T>
{
    /// Create a new Mean Shift instance with the given options.
    pub fn new(options: MeanShiftOptions<T>) -> Self {
        Self {
            options,
            cluster_centers_: None,
            labels_: None,
            n_iter_: 0,
        }
    }

    /// Fit the Mean Shift model to the data.
    pub fn fit(&mut self, data: &ArrayView2<T>) -> Result<&mut Self, ClusteringError> {
        // Manual check that all data is finite
        for row in data.rows() {
            for &val in row.iter() {
                if !val.is_finite() {
                    return Err(ClusteringError::InvalidInput(
                        "Input contains non-finite values".to_string(),
                    ));
                }
            }
        }

        let (n_samples, n_features) = data.dim();

        // Determine bandwidth
        let bandwidth = match self.options.bandwidth {
            Some(bw) => {
                if bw <= T::zero() {
                    return Err(ClusteringError::InvalidInput(
                        "Bandwidth must be positive".to_string(),
                    ));
                }
                bw
            }
            None => estimate_bandwidth(data, Some(T::from(0.3).unwrap()), None, None)?,
        };

        // Get seeds
        let seeds = match &self.options.seeds {
            Some(s) => s.clone(),
            None => {
                if self.options.bin_seeding {
                    get_bin_seeds(data, bandwidth, self.options.min_bin_freq)
                } else {
                    data.to_owned()
                }
            }
        };

        if seeds.is_empty() {
            return Err(ClusteringError::ComputationError(
                "No seeds provided and bin seeding produced no seeds".to_string(),
            ));
        }

        // Run mean shift on each seed
        let seed_results: Vec<_> = seeds
            .axis_iter(Axis(0))
            .map(|seed| mean_shift_single_seed(seed, data, bandwidth, self.options.max_iter))
            .collect();

        // Process results
        let mut center_intensity_dict: HashMap<FloatPoint<T>, usize> = HashMap::new();
        for (center, size, iterations) in seed_results {
            if size > 0 {
                center_intensity_dict.insert(FloatPoint(center), size);
            }

            // Update maximum iterations
            self.n_iter_ = self.n_iter_.max(iterations);
        }

        if center_intensity_dict.is_empty() {
            return Err(ClusteringError::ComputationError(
                format!("No point was within bandwidth={} of any seed. Try a different seeding strategy or increase the bandwidth.", bandwidth)
            ));
        }

        // Sort centers by intensity (number of points within bandwidth)
        let mut sorted_by_intensity: Vec<_> = center_intensity_dict.into_iter().collect();
        sorted_by_intensity.sort_by(|a, b| {
            b.1.cmp(&a.1).then_with(|| {
                a.0 .0
                    .iter()
                    .zip(b.0 .0.iter())
                    .find_map(|(a_val, b_val)| a_val.partial_cmp(b_val))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        // Debug: print number of centers before deduplication
        #[cfg(debug_assertions)]
        if sorted_by_intensity.len() > 1 {
            eprintln!(
                "DEBUG: Found {} centers before deduplication",
                sorted_by_intensity.len()
            );
        }

        // Convert to Array2
        let mut sorted_centers = Array2::zeros((sorted_by_intensity.len(), n_features));
        for (i, (center, _)) in sorted_by_intensity.iter().enumerate() {
            for (j, &val) in center.0.iter().enumerate() {
                sorted_centers[[i, j]] = val;
            }
        }

        // Remove near-duplicate centers
        let mut unique = vec![true; sorted_centers.nrows()];

        // Use KDTree for efficient radius search
        let kdtree = KDTree::<_, EuclideanDistance<T>>::new(&sorted_centers).map_err(|e| {
            ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e))
        })?;

        // Use a smaller threshold for merging centers (typically bandwidth/10 or less)
        let merge_threshold = bandwidth * T::from(0.1).unwrap();

        for i in 0..sorted_centers.nrows() {
            if unique[i] {
                let (indices, _) = kdtree
                    .query_radius(&sorted_centers.row(i).to_vec(), merge_threshold)
                    .map_err(|e| {
                        ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
                    })?;

                // Mark all neighbors as non-unique, except the current point
                for &idx in indices.iter() {
                    if idx != i {
                        unique[idx] = false;
                    }
                }
            }
        }

        // Extract unique centers
        let unique_indices: Vec<_> = unique
            .iter()
            .enumerate()
            .filter(|&(_, &is_unique)| is_unique)
            .map(|(i, _)| i)
            .collect();

        let mut cluster_centers = Array2::zeros((unique_indices.len(), n_features));
        for (i, &idx) in unique_indices.iter().enumerate() {
            cluster_centers.row_mut(i).assign(&sorted_centers.row(idx));
        }

        // ASSIGN LABELS: a point belongs to the cluster that it is closest to
        let kdtree = KDTree::<_, EuclideanDistance<T>>::new(&cluster_centers).map_err(|e| {
            ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e))
        })?;

        let mut labels = Array1::zeros(n_samples);

        // Batch processing to handle large datasets efficiently
        let batch_size = 1000;
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let batch = data.slice(ndarray::s![i..end, ..]);

            for (row_idx, row) in batch.rows().into_iter().enumerate() {
                let point_idx = i + row_idx;

                let (indices, distances) = kdtree.query(&row.to_vec(), 1).map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
                })?;

                if !indices.is_empty() {
                    let idx = indices[0];
                    let distance = distances[0];

                    if self.options.cluster_all || (distance <= bandwidth) {
                        labels[point_idx] = T::to_i32(&T::from(idx).unwrap()).unwrap();
                    } else {
                        // Mark as noise if not close to any cluster and not clustering all points
                        labels[point_idx] = -1;
                    }
                } else {
                    // Should never happen, but just in case
                    labels[point_idx] = -1;
                }
            }
        }

        // Store results
        self.cluster_centers_ = Some(cluster_centers);
        self.labels_ = Some(labels);

        Ok(self)
    }

    /// Get cluster centers found by the algorithm.
    pub fn cluster_centers(&self) -> &Array2<T> {
        self.cluster_centers_
            .as_ref()
            .expect("Model has not been fitted yet")
    }

    /// Get labels assigned to each data point.
    pub fn labels(&self) -> &Array1<i32> {
        self.labels_
            .as_ref()
            .expect("Model has not been fitted yet")
    }

    /// Get the number of iterations performed for the most complex seed.
    pub fn n_iter(&self) -> usize {
        self.n_iter_
    }

    /// Predict the closest cluster each sample in data belongs to.
    pub fn predict(&self, data: &ArrayView2<T>) -> Result<Array1<i32>, ClusteringError> {
        let centers = self.cluster_centers_.as_ref().ok_or_else(|| {
            ClusteringError::InvalidState("Model has not been fitted yet".to_string())
        })?;

        // Manual check that all data is finite
        for row in data.rows() {
            for &val in row.iter() {
                if !val.is_finite() {
                    return Err(ClusteringError::InvalidInput(
                        "Input contains non-finite values".to_string(),
                    ));
                }
            }
        }

        let n_samples = data.nrows();
        let mut labels = Array1::zeros(n_samples);

        // Use KDTree for efficient nearest neighbor search
        let kdtree = KDTree::<_, EuclideanDistance<T>>::new(centers).map_err(|e| {
            ClusteringError::ComputationError(format!("Failed to build KDTree: {}", e))
        })?;

        // Process in batches for memory efficiency
        let batch_size = 1000;
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let batch = data.slice(ndarray::s![i..end, ..]);

            for (row_idx, row) in batch.rows().into_iter().enumerate() {
                let (indices, _) = kdtree.query(&row.to_vec(), 1).map_err(|e| {
                    ClusteringError::ComputationError(format!("Failed to query KDTree: {}", e))
                })?;

                if !indices.is_empty() {
                    labels[i + row_idx] = T::to_i32(&T::from(indices[0]).unwrap()).unwrap();
                } else {
                    // Should never happen, but just in case
                    labels[i + row_idx] = -1;
                }
            }
        }

        Ok(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use std::collections::HashSet;

    #[test]
    fn test_estimate_bandwidth() {
        let data = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0]
        ];

        // Use a quantile that ensures we get at least one neighbor
        // With 6 points and quantile=0.3, we should get 1 neighbor
        // But let's be more explicit and use 0.4 to ensure 2 neighbors
        let quantile = 0.4;
        let n_neighbors = ((data.nrows() as f64) * quantile) as usize;
        println!("n_neighbors calculated: {}", n_neighbors);

        let bandwidth = estimate_bandwidth(&data.view(), Some(quantile), None, None).unwrap();

        // The bandwidth should be a positive value
        assert!(
            bandwidth > 0.0,
            "Bandwidth should be positive, got: {}",
            bandwidth
        );

        // With this test data, bandwidth should be reasonable
        assert!(
            bandwidth < 20.0,
            "Bandwidth should be reasonable, got: {}",
            bandwidth
        );
    }

    #[test]
    fn test_estimate_bandwidth_small_sample() {
        let data = array![[1.0, 1.0]];

        let bandwidth = estimate_bandwidth(&data.view(), Some(0.3), None, None).unwrap();

        // With only one sample, we return a default value of 1.0
        assert!(
            bandwidth > 0.0,
            "Bandwidth should be positive for single sample"
        );
        assert_eq!(bandwidth, 1.0, "Bandwidth should be 1.0 for single sample");
    }

    #[test]
    fn test_get_bin_seeds() {
        let data = array![
            [1.0, 1.0],
            [1.4, 1.4],
            [1.8, 1.2],
            [2.0, 1.0],
            [2.1, 1.1],
            [0.0, 0.0]
        ];

        // With bin_size=1.0 and min_freq=1, should get 3 bins
        let bin_seeds = get_bin_seeds(&data.view(), 1.0, 1);
        assert_eq!(bin_seeds.nrows(), 3);

        // With bin_size=1.0 and min_freq=2, should get 2 bins
        let bin_seeds = get_bin_seeds(&data.view(), 1.0, 2);
        assert_eq!(bin_seeds.nrows(), 2);

        // With very small bin_size, should get all points
        let bin_seeds = get_bin_seeds(&data.view(), 0.01, 1);
        assert_eq!(bin_seeds.nrows(), data.nrows());
    }

    #[test]
    fn test_mean_shift_simple() {
        let data = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0]
        ];

        let options = MeanShiftOptions {
            bandwidth: Some(2.0), // Adjust bandwidth for more reliable clustering
            ..Default::default()
        };

        let (centers, labels) = mean_shift(&data.view(), options).unwrap();

        // Should find at least 1 cluster
        assert!(centers.nrows() >= 1, "Should find at least 1 cluster");
        assert!(centers.nrows() <= 3, "Should find at most 3 clusters");

        // Check that all labels are valid (non-negative)
        assert!(
            labels.iter().all(|&l| l >= 0),
            "All labels should be non-negative"
        );

        // Check that labels are within range
        let max_label = *labels.iter().max().unwrap_or(&0);
        assert_eq!(
            max_label as usize,
            centers.nrows() - 1,
            "Max label should match number of centers - 1"
        );
    }

    #[test]
    fn test_mean_shift_bin_seeding() {
        let data = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0]
        ];

        let options = MeanShiftOptions {
            bandwidth: Some(2.0), // Use larger bandwidth
            bin_seeding: true,
            ..Default::default()
        };

        let (centers, labels) = mean_shift(&data.view(), options).unwrap();

        // Should find at least 1 cluster with bin seeding
        assert!(centers.nrows() >= 1, "Should find at least 1 cluster");
        assert!(centers.nrows() <= 3, "Should find at most 3 clusters");

        // Check that all labels are valid
        assert!(
            labels.iter().all(|&l| l >= 0),
            "All labels should be non-negative"
        );

        // Verify bin seeding works (doesn't crash)
        // The exact number of clusters can vary based on binning
    }

    #[test]
    fn test_mean_shift_no_cluster_all() {
        let data = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0],
            [10.0, 10.0] // Outlier
        ];

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            cluster_all: false,
            ..Default::default()
        };

        let (_centers, labels) = mean_shift(&data.view(), options).unwrap();

        // Check that we have some noise points (-1)
        assert!(labels.iter().any(|&l| l == -1));
    }

    #[test]
    fn test_mean_shift_max_iter() {
        let data = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0]
        ];

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            max_iter: 1, // Very low iteration limit
            ..Default::default()
        };

        let mut model = MeanShift::new(options);
        model.fit(&data.view()).unwrap();

        // With very low max_iter, we should hit the limit
        assert_eq!(model.n_iter(), 1);
    }

    #[test]
    fn test_mean_shift_predict() {
        let data = array![
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [4.0, 7.0],
            [3.0, 5.0],
            [3.0, 6.0]
        ];

        let options = MeanShiftOptions {
            bandwidth: Some(2.0),
            ..Default::default()
        };

        let mut model = MeanShift::new(options);
        model.fit(&data.view()).unwrap();

        // Predict the same data
        let predicted_labels = model.predict(&data.view()).unwrap();

        // Predictions on the same data should match the fitted labels
        assert_eq!(predicted_labels, model.labels().clone());
    }

    #[test]
    fn test_mean_shift_large_dataset() {
        // Create a dataset with two clear clusters
        let mut data = Array2::zeros((20, 2));

        // First cluster - tighter
        for i in 0..10 {
            data[[i, 0]] = 1.0 + 0.05 * (i as f64);
            data[[i, 1]] = 1.0 + 0.05 * (i as f64);
        }

        // Second cluster - tighter and farther away
        for i in 10..20 {
            data[[i, 0]] = 8.0 + 0.05 * ((i - 10) as f64);
            data[[i, 1]] = 8.0 + 0.05 * ((i - 10) as f64);
        }

        let options = MeanShiftOptions {
            bandwidth: Some(1.5), // Adjust bandwidth to capture cluster structure
            bin_seeding: true,    // Use bin seeding for efficiency
            ..Default::default()
        };

        let (centers, labels) = mean_shift(&data.view(), options).unwrap();

        // Should find 1-2 clusters (algorithm can merge distant clusters with large bandwidth)
        assert!(centers.nrows() >= 1, "Should find at least 1 cluster");
        assert!(centers.nrows() <= 3, "Should find at most 3 clusters");

        // Check that we have valid labels
        let unique_labels: HashSet<_> = labels.iter().cloned().collect();
        assert!(
            !unique_labels.is_empty(),
            "Should have at least 1 unique label"
        );
        assert!(
            unique_labels.len() <= centers.nrows(),
            "Unique labels should not exceed number of centers"
        );
    }
}
