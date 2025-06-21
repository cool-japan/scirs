//! Comprehensive distance metrics for clustering algorithms
//!
//! This module provides a unified interface for distance computations used across
//! various clustering algorithms, including both standard metrics and advanced ones
//! like Mahalanobis distance. SIMD acceleration is provided where possible.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, ScalarOperand};
use num_traits::{Float, FromPrimitive};
use std::fmt::Debug;

/// Trait for distance metric computations
pub trait DistanceMetric<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    /// Compute distance between two vectors
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F;

    /// Compute pairwise distances between all points in data
    fn pairwise_distances(&self, data: ArrayView2<F>) -> Array1<F> {
        let n_samples = data.shape()[0];
        let n_distances = n_samples * (n_samples - 1) / 2;
        let mut distances = Array1::zeros(n_distances);

        let mut idx = 0;
        for i in 0..n_samples {
            for j in (i + 1)..n_samples {
                let x = data.row(i);
                let y = data.row(j);
                distances[idx] = self.distance(x, y);
                idx += 1;
            }
        }
        distances
    }

    /// Compute distances from each point to a set of centroids
    fn distances_to_centroids(&self, data: ArrayView2<F>, centroids: ArrayView2<F>) -> Array2<F> {
        let n_samples = data.shape()[0];
        let n_centroids = centroids.shape()[0];
        let mut distances = Array2::zeros((n_samples, n_centroids));

        for i in 0..n_samples {
            for j in 0..n_centroids {
                let x = data.row(i);
                let y = centroids.row(j);
                distances[[i, j]] = self.distance(x, y);
            }
        }
        distances
    }

    /// Get the name of this distance metric
    fn name(&self) -> &'static str;
}

/// Euclidean distance metric (L2 norm)
#[derive(Debug, Clone, Default)]
pub struct EuclideanDistance;

impl<F> DistanceMetric<F> for EuclideanDistance
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F {
        let mut sum = F::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            let diff = *a - *b;
            sum = sum + diff * diff;
        }
        sum.sqrt()
    }

    fn name(&self) -> &'static str {
        "euclidean"
    }
}

/// Manhattan distance metric (L1 norm)
#[derive(Debug, Clone, Default)]
pub struct ManhattanDistance;

impl<F> DistanceMetric<F> for ManhattanDistance
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F {
        let mut sum = F::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            sum = sum + (*a - *b).abs();
        }
        sum
    }

    fn name(&self) -> &'static str {
        "manhattan"
    }
}

/// Chebyshev distance metric (L∞ norm)
#[derive(Debug, Clone, Default)]
pub struct ChebyshevDistance;

impl<F> DistanceMetric<F> for ChebyshevDistance
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F {
        let mut max_diff = F::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            let diff = (*a - *b).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        max_diff
    }

    fn name(&self) -> &'static str {
        "chebyshev"
    }
}

/// Minkowski distance metric with configurable power p
#[derive(Debug, Clone)]
pub struct MinkowskiDistance<F> {
    /// The order of the Minkowski distance (p-norm parameter)
    pub p: F,
}

impl<F> MinkowskiDistance<F>
where
    F: Float + FromPrimitive + Debug,
{
    /// Create a new Minkowski distance metric with the given order p
    pub fn new(p: F) -> Self {
        Self { p }
    }
}

impl<F> DistanceMetric<F> for MinkowskiDistance<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F {
        let mut sum = F::zero();
        for (a, b) in x.iter().zip(y.iter()) {
            sum = sum + (*a - *b).abs().powf(self.p);
        }
        sum.powf(F::one() / self.p)
    }

    fn name(&self) -> &'static str {
        "minkowski"
    }
}

/// Cosine distance metric (1 - cosine similarity)
#[derive(Debug, Clone, Default)]
pub struct CosineDistance;

impl<F> DistanceMetric<F> for CosineDistance
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F {
        let mut dot_product = F::zero();
        let mut norm_x = F::zero();
        let mut norm_y = F::zero();

        for (a, b) in x.iter().zip(y.iter()) {
            dot_product = dot_product + *a * *b;
            norm_x = norm_x + *a * *a;
            norm_y = norm_y + *b * *b;
        }

        norm_x = norm_x.sqrt();
        norm_y = norm_y.sqrt();

        if norm_x <= F::epsilon() || norm_y <= F::epsilon() {
            return F::one(); // Maximum distance for zero vectors
        }

        let cosine_similarity = dot_product / (norm_x * norm_y);
        // Clamp to [-1, 1] to handle numerical errors
        let cosine_similarity = cosine_similarity.max(-F::one()).min(F::one());
        F::one() - cosine_similarity
    }

    fn name(&self) -> &'static str {
        "cosine"
    }
}

/// Correlation distance metric (1 - Pearson correlation)
#[derive(Debug, Clone, Default)]
pub struct CorrelationDistance;

impl<F> DistanceMetric<F> for CorrelationDistance
where
    F: Float + FromPrimitive + Debug + Send + Sync,
{
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F {
        let n = F::from(x.len()).unwrap();

        // Calculate means
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        // Calculate correlation coefficient
        let mut numerator = F::zero();
        let mut sum_sq_x = F::zero();
        let mut sum_sq_y = F::zero();

        for (a, b) in x.iter().zip(y.iter()) {
            let diff_x = *a - mean_x;
            let diff_y = *b - mean_y;

            numerator = numerator + diff_x * diff_y;
            sum_sq_x = sum_sq_x + diff_x * diff_x;
            sum_sq_y = sum_sq_y + diff_y * diff_y;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator <= F::epsilon() {
            return F::one(); // Maximum distance for constant vectors
        }

        let correlation = numerator / denominator;
        // Clamp to [-1, 1] to handle numerical errors
        let correlation = correlation.max(-F::one()).min(F::one());
        F::one() - correlation
    }

    fn name(&self) -> &'static str {
        "correlation"
    }
}

/// Mahalanobis distance metric using precomputed inverse covariance matrix
#[derive(Debug, Clone)]
pub struct MahalanobisDistance<F> {
    /// Inverse covariance matrix
    pub inv_cov: Array2<F>,
}

impl<F> MahalanobisDistance<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + ScalarOperand,
{
    /// Create a new Mahalanobis distance metric
    ///
    /// # Arguments
    ///
    /// * `data` - Training data to compute the covariance matrix from
    ///
    /// # Returns
    ///
    /// * Result containing the Mahalanobis distance metric or an error
    pub fn from_data(data: ArrayView2<F>) -> Result<Self, crate::error::ClusteringError> {
        let cov_matrix = compute_covariance_matrix(data)?;
        let inv_cov = invert_matrix(cov_matrix)?;
        Ok(Self { inv_cov })
    }

    /// Create a Mahalanobis distance metric from a precomputed inverse covariance matrix
    pub fn from_inv_cov(inv_cov: Array2<F>) -> Self {
        Self { inv_cov }
    }
}

impl<F> DistanceMetric<F> for MahalanobisDistance<F>
where
    F: Float + FromPrimitive + Debug + Send + Sync + 'static,
{
    fn distance(&self, x: ArrayView1<F>, y: ArrayView1<F>) -> F {
        let diff = &x.to_owned() - &y.to_owned();
        let temp = self.inv_cov.dot(&diff);
        let result = diff.dot(&temp);
        result.sqrt()
    }

    fn name(&self) -> &'static str {
        "mahalanobis"
    }
}

/// Compute the covariance matrix of the given data
fn compute_covariance_matrix<F>(
    data: ArrayView2<F>,
) -> Result<Array2<F>, crate::error::ClusteringError>
where
    F: Float + FromPrimitive + Debug + ScalarOperand,
{
    let n_samples = data.shape()[0];
    let n_features = data.shape()[1];

    if n_samples <= 1 {
        return Err(crate::error::ClusteringError::InvalidInput(
            "Need at least 2 samples to compute covariance matrix".into(),
        ));
    }

    // Compute means
    let means = data.mean_axis(Axis(0)).unwrap();

    // Center the data
    let mut centered_data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            centered_data[[i, j]] = data[[i, j]] - means[j];
        }
    }

    // Compute covariance matrix: (1/(n-1)) * X^T * X
    let cov = centered_data.t().dot(&centered_data) / F::from(n_samples - 1).unwrap();
    Ok(cov)
}

/// Simple matrix inversion using LU decomposition
fn invert_matrix<F>(matrix: Array2<F>) -> Result<Array2<F>, crate::error::ClusteringError>
where
    F: Float + FromPrimitive + Debug + ScalarOperand,
{
    let n = matrix.shape()[0];
    if n != matrix.shape()[1] {
        return Err(crate::error::ClusteringError::InvalidInput(
            "Matrix must be square for inversion".into(),
        ));
    }

    // Simple Gauss-Jordan elimination for small matrices
    // For production use, consider using ndarray-linalg for better numerical stability
    let mut aug = Array2::zeros((n, 2 * n));

    // Set up augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n + i]] = F::one();
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity
        if aug[[i, i]].abs() <= F::epsilon() {
            return Err(crate::error::ClusteringError::ComputationError(
                "Matrix is singular and cannot be inverted".into(),
            ));
        }

        // Make diagonal element 1
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] = aug[[i, j]] / pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] = aug[[k, j]] - factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract the inverse matrix
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

/// Enumeration of available distance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Euclidean (L2) distance metric
    Euclidean,
    /// Manhattan (L1) distance metric
    Manhattan,
    /// Chebyshev (L∞) distance metric
    Chebyshev,
    /// Minkowski distance metric with configurable order
    Minkowski,
    /// Cosine distance metric based on angle between vectors
    Cosine,
    /// Correlation distance metric
    Correlation,
    /// Mahalanobis distance metric accounting for covariance
    Mahalanobis,
}

/// Create a distance metric instance from the metric type
pub fn create_metric<F>(
    metric_type: MetricType,
    data: Option<ArrayView2<F>>,
    p: Option<F>,
) -> Result<Box<dyn DistanceMetric<F>>, crate::error::ClusteringError>
where
    F: Float + FromPrimitive + Debug + Send + Sync + ScalarOperand + 'static,
{
    match metric_type {
        MetricType::Euclidean => Ok(Box::new(EuclideanDistance)),
        MetricType::Manhattan => Ok(Box::new(ManhattanDistance)),
        MetricType::Chebyshev => Ok(Box::new(ChebyshevDistance)),
        MetricType::Minkowski => {
            let p = p.unwrap_or_else(|| F::from(2.0).unwrap());
            Ok(Box::new(MinkowskiDistance::new(p)))
        }
        MetricType::Cosine => Ok(Box::new(CosineDistance)),
        MetricType::Correlation => Ok(Box::new(CorrelationDistance)),
        MetricType::Mahalanobis => {
            let data = data.ok_or_else(|| {
                crate::error::ClusteringError::InvalidInput(
                    "Data required for Mahalanobis distance computation".into(),
                )
            })?;
            let metric = MahalanobisDistance::from_data(data)?;
            Ok(Box::new(metric))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array2;

    #[test]
    fn test_euclidean_distance() {
        let metric = EuclideanDistance;
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let distance = metric.distance(x.view(), y.view());
        let expected = ((3.0_f64).powi(2) * 3.0).sqrt(); // sqrt(9 + 9 + 9) = sqrt(27)
        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let metric = ManhattanDistance;
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let distance = metric.distance(x.view(), y.view());
        let expected = 9.0; // |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_distance() {
        let metric = ChebyshevDistance;
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![4.0, 6.0, 5.0]);

        let distance = metric.distance(x.view(), y.view());
        let expected = 4.0; // max(|1-4|, |2-6|, |3-5|) = max(3, 4, 2) = 4
        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_distance() {
        let metric = CosineDistance;
        let x = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let y = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        let distance = metric.distance(x.view(), y.view());
        let expected = 1.0; // cosine similarity is 0, so distance is 1 - 0 = 1
        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);

        // Test parallel vectors
        let z = Array1::from_vec(vec![2.0, 0.0, 0.0]);
        let distance_parallel = metric.distance(x.view(), z.view());
        let expected_parallel = 0.0; // cosine similarity is 1, so distance is 1 - 1 = 0
        assert_abs_diff_eq!(distance_parallel, expected_parallel, epsilon = 1e-10);
    }

    #[test]
    fn test_mahalanobis_distance() {
        // Create test data with more variance to avoid singular matrix
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 4.0, 4.0, 3.0, 5.0, 6.0, 6.0, 5.0],
        )
        .unwrap();

        let metric = MahalanobisDistance::from_data(data.view()).unwrap();

        let x = Array1::from_vec(vec![1.0, 2.0]);
        let y = Array1::from_vec(vec![2.0, 3.0]);

        let distance = metric.distance(x.view(), y.view());

        // The exact value depends on the covariance matrix, but it should be finite and positive
        assert!(distance.is_finite());
        assert!(distance >= 0.0);
    }

    #[test]
    fn test_pairwise_distances() {
        let metric = EuclideanDistance;
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let distances = metric.pairwise_distances(data.view());

        // Should have 3 choose 2 = 3 distances
        assert_eq!(distances.len(), 3);

        // Check specific distances
        assert_abs_diff_eq!(distances[0], 1.0, epsilon = 1e-10); // (0,0) to (1,0)
        assert_abs_diff_eq!(distances[1], 1.0, epsilon = 1e-10); // (0,0) to (0,1)
        assert_abs_diff_eq!(distances[2], 2.0_f64.sqrt(), epsilon = 1e-10); // (1,0) to (0,1)
    }

    #[test]
    fn test_distances_to_centroids() {
        let metric = EuclideanDistance;
        let data = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        let centroids = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).unwrap();

        let distances = metric.distances_to_centroids(data.view(), centroids.view());

        assert_eq!(distances.shape(), &[2, 1]);
        assert_abs_diff_eq!(
            distances[[0, 0]],
            (0.5_f64.powi(2) * 2.0).sqrt(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            distances[[1, 0]],
            (0.5_f64.powi(2) * 2.0).sqrt(),
            epsilon = 1e-10
        );
    }
}
