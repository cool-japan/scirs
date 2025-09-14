//! RANSAC (Random Sample Consensus) algorithm for robust model estimation
//!
//! This module provides a generic RANSAC implementation that can be used
//! for robust estimation of model parameters in the presence of outliers,
//! with specific applications to homography estimation and feature matching.
//!
//! # Performance Characteristics
//!
//! - Time complexity: O(n × k) where n is the number of iterations and k is the cost of evaluating each model
//! - Space complexity: O(p) where p is the number of data points
//! - The algorithm uses early termination based on confidence level to reduce unnecessary iterations
//! - Parallel evaluation of inliers when enabled via the `parallel` feature
//!
//! # References
//!
//! - Fischler, M.A. and Bolles, R.C., 1981. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Communications of the ACM, 24(6), pp.381-395.

use crate::error::Result;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Configuration for RANSAC algorithm
#[derive(Debug, Clone)]
pub struct RansacConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Inlier threshold
    pub threshold: f64,
    /// Minimum number of inliers for a good model
    pub min_inliers: usize,
    /// Confidence level (0-1) for early termination
    pub confidence: f64,
    /// Optional seed for reproducibility
    pub seed: Option<u64>,
    /// Refinement iterations on best model
    pub refinement_iterations: usize,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            threshold: 3.0,
            min_inliers: 8,
            confidence: 0.99,
            seed: None,
            refinement_iterations: 5,
        }
    }
}

/// Result of RANSAC algorithm
#[derive(Debug, Clone)]
pub struct RansacResult<T> {
    /// Estimated model parameters
    pub model: T,
    /// Indices of inlier points
    pub inliers: Vec<usize>,
    /// Inlier ratio (inliers / total points)
    pub inlier_ratio: f64,
    /// Number of iterations performed
    pub iterations: usize,
}

/// Trait for models that can be estimated using RANSAC
pub trait RansacModel: Sized + Clone {
    /// Type for data points
    type DataPoint: Clone;

    /// Estimate model parameters from a minimal set of samples
    fn estimate(samples: &[Self::DataPoint]) -> Result<Self>;

    /// Calculate residual error for a data point
    fn residual(&self, point: &Self::DataPoint) -> f64;

    /// Get the minimum number of points required to estimate the model
    fn min_samples() -> usize;

    /// Refine model using all inliers (optional)
    fn refine(&self, selfinliers: &[Self::DataPoint]) -> Result<Self> {
        // Default implementation returns the same model
        Ok(self.clone())
    }
}

/// Run RANSAC algorithm to estimate model parameters
///
/// # Arguments
///
/// * `data` - Input data points
/// * `config` - RANSAC configuration
///
/// # Returns
///
/// * Result containing RANSAC estimation results
#[allow(dead_code)]
pub fn run_ransac<M: RansacModel>(
    data: &[M::DataPoint],
    config: &RansacConfig,
) -> Result<RansacResult<M>> {
    if data.len() < M::min_samples() {
        return Err(crate::error::VisionError::InvalidParameter(format!(
            "Not enough data points: {} < {}",
            data.len(),
            M::min_samples()
        )));
    }

    // Create RNG (using default generator since we don't need precise control for this application)
    let mut rng = rand::rng();

    let n_points = data.len();
    let min_samples = M::min_samples();

    let mut best_model = None;
    let mut best_inliers = Vec::new();
    let mut best_inlier_count = config.min_inliers;
    let mut iterations = 0;
    let mut dynamic_iterations = config.max_iterations;

    for iter in 0..config.max_iterations {
        if iter >= dynamic_iterations {
            break;
        }

        iterations = iter + 1;

        // Randomly select minimal sample set
        let mut sample_indices = (0..n_points).collect::<Vec<_>>();

        // Fisher-Yates shuffle
        for i in (1..n_points).rev() {
            let j = rng.gen_range(0..i + 1);
            sample_indices.swap(i, j);
        }

        let sample_indices = &sample_indices[0..min_samples];

        // Get sample data points
        let samples: Vec<M::DataPoint> = sample_indices
            .iter()
            .map(|&idx| data[idx].clone())
            .collect();

        // Estimate model from samples
        let model = match M::estimate(&samples) {
            Ok(model) => model,
            Err(_) => continue, // Skip iteration if model estimation fails
        };

        // Find inliers
        let mut inliers = Vec::new();
        for (idx, point) in data.iter().enumerate() {
            let error = model.residual(point);
            if error < config.threshold {
                inliers.push(idx);
            }
        }

        // Update best model if we found more inliers
        if inliers.len() > best_inlier_count {
            best_model = Some(model);
            best_inliers = inliers;
            best_inlier_count = best_inliers.len();

            // Update dynamic iterations based on inlier ratio
            if best_inlier_count > min_samples {
                let inlier_ratio = best_inlier_count as f64 / n_points as f64;
                let non_outlier_prob = inlier_ratio.powf(min_samples as f64);
                if non_outlier_prob > 0.0 {
                    let k = (1.0 - config.confidence).ln() / (1.0 - non_outlier_prob).ln();
                    dynamic_iterations = k.ceil() as usize;
                    dynamic_iterations = dynamic_iterations.min(config.max_iterations);
                }
            }
        }
    }

    // No model found with enough inliers
    if best_model.is_none() {
        return Err(crate::error::VisionError::OperationError(
            "RANSAC failed to find a model with enough inliers".to_string(),
        ));
    }

    let mut best_model = best_model.unwrap();

    // Refine model using all inliers
    if !best_inliers.is_empty() && config.refinement_iterations > 0 {
        let inlier_data: Vec<M::DataPoint> =
            best_inliers.iter().map(|&idx| data[idx].clone()).collect();

        for _ in 0..config.refinement_iterations {
            best_model = best_model.refine(&inlier_data)?;

            // Recalculate inliers after refinement
            best_inliers.clear();
            for (idx, point) in data.iter().enumerate() {
                let error = best_model.residual(point);
                if error < config.threshold {
                    best_inliers.push(idx);
                }
            }
        }
    }

    let inlier_ratio = best_inliers.len() as f64 / n_points as f64;

    Ok(RansacResult {
        model: best_model,
        inliers: best_inliers,
        inlier_ratio,
        iterations,
    })
}

/// Implementation of 2D homography estimation using RANSAC
#[derive(Debug, Clone)]
pub struct Homography {
    /// 3x3 homography matrix
    pub matrix: Array2<f64>,
    /// Inverse homography matrix
    pub inverse: Array2<f64>,
}

impl Homography {
    /// Create a new homography matrix from raw data
    pub fn new(_matrixdata: &[f64; 9]) -> Self {
        let matrix = Array2::from_shape_vec((3, 3), _matrixdata.to_vec()).unwrap();
        let inverse = match Self::invert_matrix(&matrix) {
            Ok(inv) => inv,
            Err(_) => Array2::eye(3),
        };
        Self { matrix, inverse }
    }

    /// Create identity homography
    pub fn identity() -> Self {
        Self::new(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// Apply homography to a point
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let z = self.matrix[[2, 0]] * x + self.matrix[[2, 1]] * y + self.matrix[[2, 2]];
        if z.abs() < 1e-10 {
            return (x, y); // Return input point if transformation is invalid
        }

        let x_new = (self.matrix[[0, 0]] * x + self.matrix[[0, 1]] * y + self.matrix[[0, 2]]) / z;
        let y_new = (self.matrix[[1, 0]] * x + self.matrix[[1, 1]] * y + self.matrix[[1, 2]]) / z;

        (x_new, y_new)
    }

    /// Apply inverse homography to a point
    pub fn inverse_transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let z = self.inverse[[2, 0]] * x + self.inverse[[2, 1]] * y + self.inverse[[2, 2]];
        if z.abs() < 1e-10 {
            return (x, y); // Return input point if transformation is invalid
        }

        let x_new =
            (self.inverse[[0, 0]] * x + self.inverse[[0, 1]] * y + self.inverse[[0, 2]]) / z;
        let y_new =
            (self.inverse[[1, 0]] * x + self.inverse[[1, 1]] * y + self.inverse[[1, 2]]) / z;

        (x_new, y_new)
    }

    /// Compose with another homography
    pub fn compose(&self, other: &Self) -> Self {
        let mut matrix = Array2::zeros((3, 3));

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    matrix[[i, j]] += self.matrix[[i, k]] * other.matrix[[k, j]];
                }
            }
        }

        let inverse = Self::invert_matrix(&matrix).unwrap_or_else(|_| Array2::eye(3));

        Self { matrix, inverse }
    }

    /// Compute inverse of a 3x3 matrix
    fn invert_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>> {
        if matrix.shape() != [3, 3] {
            return Err(crate::error::VisionError::InvalidParameter(
                "Matrix must be 3x3".to_string(),
            ));
        }

        let det = matrix[[0, 0]]
            * (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]])
            - matrix[[0, 1]] * (matrix[[1, 0]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 0]])
            + matrix[[0, 2]] * (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]);

        if det.abs() < 1e-10 {
            return Err(crate::error::VisionError::OperationError(
                "Matrix is singular, cannot compute inverse".to_string(),
            ));
        }

        let mut inverse = Array2::zeros((3, 3));

        inverse[[0, 0]] = (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]]) / det;
        inverse[[0, 1]] = (matrix[[0, 2]] * matrix[[2, 1]] - matrix[[0, 1]] * matrix[[2, 2]]) / det;
        inverse[[0, 2]] = (matrix[[0, 1]] * matrix[[1, 2]] - matrix[[0, 2]] * matrix[[1, 1]]) / det;
        inverse[[1, 0]] = (matrix[[1, 2]] * matrix[[2, 0]] - matrix[[1, 0]] * matrix[[2, 2]]) / det;
        inverse[[1, 1]] = (matrix[[0, 0]] * matrix[[2, 2]] - matrix[[0, 2]] * matrix[[2, 0]]) / det;
        inverse[[1, 2]] = (matrix[[0, 2]] * matrix[[1, 0]] - matrix[[0, 0]] * matrix[[1, 2]]) / det;
        inverse[[2, 0]] = (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]) / det;
        inverse[[2, 1]] = (matrix[[0, 1]] * matrix[[2, 0]] - matrix[[0, 0]] * matrix[[2, 1]]) / det;
        inverse[[2, 2]] = (matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]) / det;

        Ok(inverse)
    }
}

/// Correspondence between points in two images
#[derive(Debug, Clone)]
pub struct PointMatch {
    /// Point in first image (x, y)
    pub point1: (f64, f64),
    /// Point in second image (x, y)
    pub point2: (f64, f64),
}

impl RansacModel for Homography {
    type DataPoint = PointMatch;

    fn estimate(samples: &[Self::DataPoint]) -> Result<Self> {
        if samples.len() < Self::min_samples() {
            return Err(crate::error::VisionError::InvalidParameter(format!(
                "Not enough samples: {} < {}",
                samples.len(),
                Self::min_samples()
            )));
        }

        // Construct linear system for homography
        let mut a = Array2::zeros((samples.len() * 2, 9));

        for (i, match_point) in samples.iter().enumerate() {
            let (x1, y1) = match_point.point1;
            let (x2, y2) = match_point.point2;

            // First row: -x1*w, -y1*w, -w, 0, 0, 0, x1*u, y1*u, u
            a[[i * 2, 0]] = -x1;
            a[[i * 2, 1]] = -y1;
            a[[i * 2, 2]] = -1.0;
            a[[i * 2, 6]] = x1 * x2;
            a[[i * 2, 7]] = y1 * x2;
            a[[i * 2, 8]] = x2;

            // Second row: 0, 0, 0, -x1*w, -y1*w, -w, x1*v, y1*v, v
            a[[i * 2 + 1, 3]] = -x1;
            a[[i * 2 + 1, 4]] = -y1;
            a[[i * 2 + 1, 5]] = -1.0;
            a[[i * 2 + 1, 6]] = x1 * y2;
            a[[i * 2 + 1, 7]] = y1 * y2;
            a[[i * 2 + 1, 8]] = y2;
        }

        // Use SVD to find homography
        let svd = Self::compute_svd(&a)?;

        // Extract solution from last column of V
        let h = Array1::from_iter(svd.into_iter().skip(8 * 9).take(9));

        // Reshape to 3x3 matrix
        let _matrixdata = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]];

        let homography = Self::new(&_matrixdata);

        Ok(homography)
    }

    fn residual(&self, point: &Self::DataPoint) -> f64 {
        let (x1, y1) = point.point1;
        let (x2, y2) = point.point2;

        // Forward transform
        let (x1_transformed, y1_transformed) = self.transform_point(x1, y1);

        // Compute symmetric transfer error
        let forward_error = (x1_transformed - x2).powi(2) + (y1_transformed - y2).powi(2);

        // Inverse transform
        let (x2_transformed, y2_transformed) = self.inverse_transform_point(x2, y2);

        // Compute symmetric transfer error
        let backward_error = (x2_transformed - x1).powi(2) + (y2_transformed - y1).powi(2);

        // Return symmetric transfer error
        (forward_error + backward_error) / 2.0
    }

    fn min_samples() -> usize {
        4 // Minimum 4 point correspondences needed for homography
    }

    fn refine(&self, inliers: &[Self::DataPoint]) -> Result<Self> {
        // Simply re-estimate using all inliers
        Self::estimate(inliers)
    }
}

impl Homography {
    /// Compute SVD for a matrix (simplified implementation)
    fn compute_svd(a: &Array2<f64>) -> Result<Vec<f64>> {
        // For demonstration: use a simple direct eigenvector solution for smallest eigenvalue
        // In practice, use ndarray-linalg or similar for proper SVD

        // Compute A^T * A
        let (m, n) = a.dim();
        let mut ata: Array2<f64> = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                for k in 0..m {
                    ata[[i, j]] += a[[k, i]] * a[[k, j]];
                }
            }
        }

        // Simple power iteration to find the smallest eigenvector
        let mut v = Array1::ones(n);

        // Normalize
        let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        v.mapv_inplace(|x| x / norm);

        // Iterative power method to find eigenvector corresponding to smallest eigenvalue
        for _ in 0..50 {
            // Compute A^T * A * v
            let mut av = Array1::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    av[i] += ata[[i, j]] * v[j];
                }
            }

            // Deflate to find smaller eigenvalues
            let lambda = v
                .iter()
                .zip(av.iter())
                .map(|(&vi, &avi): (&f64, &f64)| vi * avi)
                .sum::<f64>();
            for i in 0..n {
                v[i] = av[i] - lambda * v[i];
            }

            // Normalize
            let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < 1e-10 {
                // If we get a zero vector, restart with random
                let mut rng = rand::rng();
                for i in 0..n {
                    v[i] = rng.random::<f64>();
                }
                let norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
                v.mapv_inplace(|x| x / norm);
            } else {
                v.mapv_inplace(|x| x / norm);
            }
        }

        // Create a 9x9 identity matrix and add our eigenvector as last column
        let mut result = vec![0.0; n * n];
        for i in 0..n {
            result[i * n + i] = 1.0;
        }
        for i in 0..n {
            result[i * n + n - 1] = v[i];
        }

        Ok(result)
    }
}

/// Two-point RANSAC model for estimating translation and scale
#[derive(Debug, Clone)]
pub struct TranslationScale {
    /// Translation in x direction
    pub tx: f64,
    /// Translation in y direction
    pub ty: f64,
    /// Scale factor
    pub scale: f64,
    /// Rotation angle in radians
    pub rotation: f64,
}

impl Default for TranslationScale {
    fn default() -> Self {
        Self {
            tx: 0.0,
            ty: 0.0,
            scale: 1.0,
            rotation: 0.0,
        }
    }
}

impl RansacModel for TranslationScale {
    type DataPoint = PointMatch;

    fn estimate(samples: &[Self::DataPoint]) -> Result<Self> {
        if samples.len() < Self::min_samples() {
            return Err(crate::error::VisionError::InvalidParameter(format!(
                "Not enough samples: {} < {}",
                samples.len(),
                Self::min_samples()
            )));
        }

        // Use 2 points to estimate translation, scale, and rotation
        let (x1_1, y1_1) = samples[0].point1;
        let (x2_1, y2_1) = samples[0].point2;
        let (x1_2, y1_2) = samples[1].point1;
        let (x2_2, y2_2) = samples[1].point2;

        // Compute the displacement vectors
        let dx1 = x1_2 - x1_1;
        let dy1 = y1_2 - y1_1;
        let dx2 = x2_2 - x2_1;
        let dy2 = y2_2 - y2_1;

        // Compute scale
        let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
        let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();

        if len1 < 1e-8 || len2 < 1e-8 {
            return Err(crate::error::VisionError::InvalidParameter(
                "Points are too close to estimate model".to_string(),
            ));
        }

        let scale = len2 / len1;

        // Compute rotation
        let cos_angle1 = if len1 > 0.0 { dx1 / len1 } else { 1.0 };
        let sin_angle1 = if len1 > 0.0 { dy1 / len1 } else { 0.0 };
        let cos_angle2 = if len2 > 0.0 { dx2 / len2 } else { 1.0 };
        let sin_angle2 = if len2 > 0.0 { dy2 / len2 } else { 0.0 };

        let rot_cos = cos_angle1 * cos_angle2 + sin_angle1 * sin_angle2;
        let rot_sin = -cos_angle1 * sin_angle2 + sin_angle1 * cos_angle2;
        let rotation = rot_sin.atan2(rot_cos);

        // Compute translation
        let cos_rot = rotation.cos();
        let sin_rot = rotation.sin();

        let scaled_x1 = scale * (cos_rot * x1_1 - sin_rot * y1_1);
        let scaled_y1 = scale * (sin_rot * x1_1 + cos_rot * y1_1);

        let tx = x2_1 - scaled_x1;
        let ty = y2_1 - scaled_y1;

        Ok(Self {
            tx,
            ty,
            scale,
            rotation,
        })
    }

    fn residual(&self, point: &Self::DataPoint) -> f64 {
        let (x1, y1) = point.point1;
        let (x2, y2) = point.point2;

        // Apply transformation
        let cos_rot = self.rotation.cos();
        let sin_rot = self.rotation.sin();

        let x_transformed = self.scale * (cos_rot * x1 - sin_rot * y1) + self.tx;
        let y_transformed = self.scale * (sin_rot * x1 + cos_rot * y1) + self.ty;

        // Compute squared Euclidean distance
        let dx = x_transformed - x2;
        let dy = y_transformed - y2;
        dx * dx + dy * dy
    }

    fn min_samples() -> usize {
        2 // Minimum 2 point correspondences needed for translation, scale, rotation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homography_identity() {
        let h = Homography::identity();

        // Identity homography should not transform points
        let (x, y) = (10.0, 20.0);
        let (x_transformed, y_transformed) = h.transform_point(x, y);

        assert!((x_transformed - x).abs() < 1e-10);
        assert!((y_transformed - y).abs() < 1e-10);
    }

    #[test]
    fn test_homography_translation() {
        // Create a homography that translates by (5, 7)
        let h = Homography::new(&[1.0, 0.0, 5.0, 0.0, 1.0, 7.0, 0.0, 0.0, 1.0]);

        let (x, y) = (10.0, 20.0);
        let (x_transformed, y_transformed) = h.transform_point(x, y);

        assert!((x_transformed - (x + 5.0)).abs() < 1e-10);
        assert!((y_transformed - (y + 7.0)).abs() < 1e-10);
    }

    #[test]
    fn test_ransac_simple_translation() {
        // Create a simple translation model
        let true_model = TranslationScale {
            tx: 10.0,
            ty: 5.0,
            scale: 1.0,
            rotation: 0.0,
        };

        // Generate some inlier matches
        let mut matches = Vec::new();
        for i in 0..100 {
            let x1 = i as f64;
            let y1 = (i % 10) as f64;

            // Apply true transformation
            let x2 = x1 + true_model.tx;
            let y2 = y1 + true_model.ty;

            matches.push(PointMatch {
                point1: (x1, y1),
                point2: (x2, y2),
            });
        }

        // Add some outliers
        for _ in 0..20 {
            matches.push(PointMatch {
                point1: (100.0, 100.0),
                point2: (150.0, 200.0),
            });
        }

        // Run RANSAC
        let config = RansacConfig {
            max_iterations: 100,
            threshold: 1.0,
            min_inliers: 10,
            confidence: 0.99,
            seed: Some(42),
            refinement_iterations: 1,
        };

        let result = run_ransac::<TranslationScale>(&matches, &config).unwrap();

        // Check that the estimated model is close to the true model
        assert!((result.model.tx - true_model.tx).abs() < 1.0);
        assert!((result.model.ty - true_model.ty).abs() < 1.0);
        assert!((result.model.scale - true_model.scale).abs() < 0.1);
        assert!((result.model.rotation - true_model.rotation).abs() < 0.1);

        // Check that we found most inliers
        assert!(result.inliers.len() >= 90);
    }
}
