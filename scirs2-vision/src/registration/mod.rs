//! Image registration algorithms
//!
//! This module provides various image registration techniques for aligning images
//! based on features, intensity, or geometric constraints.

pub mod affine;
pub mod feature_based;
pub mod homography;
pub mod intensity;
pub mod metrics;
pub mod non_rigid;
pub mod optimization;
pub mod rigid;
pub mod warping;

pub use affine::*;
pub use feature_based::*;
pub use homography::*;
pub use intensity::*;
pub use metrics::*;
pub use non_rigid::*;
pub use optimization::*;
pub use rigid::*;
pub use warping::*;

use crate::error::{Result, VisionError};
use ndarray::{Array1, Array2};
use scirs2_linalg::{lstsq, solve};
use std::fmt::Debug;

/// 2D transformation matrix (3x3 homogeneous coordinates)
pub type TransformMatrix = Array2<f64>;

/// Point in 2D space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl Point2D {
    /// Create a new 2D point
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// Match between two points
#[derive(Debug, Clone)]
pub struct PointMatch {
    /// Source point
    pub source: Point2D,
    /// Target point
    pub target: Point2D,
    /// Match confidence score
    pub confidence: f64,
}

/// Registration parameters
#[derive(Debug, Clone)]
pub struct RegistrationParams {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Use multi-resolution pyramid
    pub use_pyramid: bool,
    /// Number of pyramid levels
    pub pyramid_levels: usize,
    /// RANSAC parameters
    pub ransac_threshold: f64,
    /// Number of RANSAC iterations
    pub ransac_iterations: usize,
    /// RANSAC confidence level
    pub ransac_confidence: f64,
}

impl Default for RegistrationParams {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            use_pyramid: true,
            pyramid_levels: 3,
            ransac_threshold: 3.0,
            ransac_iterations: 1000,
            ransac_confidence: 0.99,
        }
    }
}

/// Registration result
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// Final transformation matrix
    pub transform: TransformMatrix,
    /// Final cost/error value
    pub final_cost: f64,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Inlier matches (for RANSAC-based methods)
    pub inliers: Vec<usize>,
}

/// Type of transformation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformType {
    /// Rigid transformation (rotation + translation)
    Rigid,
    /// Similarity transformation (rotation + translation + uniform scaling)
    Similarity,
    /// Affine transformation (rotation + translation + scaling + shearing)
    Affine,
    /// Homography transformation (perspective transformation)
    Homography,
}

/// Create identity transformation matrix
pub fn identity_transform() -> TransformMatrix {
    Array2::eye(3)
}

/// Apply transformation to a point
pub fn transform_point(point: Point2D, transform: &TransformMatrix) -> Point2D {
    let homogeneous = Array1::from(vec![point.x, point.y, 1.0]);
    let transformed = transform.dot(&homogeneous);

    if transformed[2].abs() < 1e-10 {
        Point2D::new(transformed[0], transformed[1])
    } else {
        Point2D::new(
            transformed[0] / transformed[2],
            transformed[1] / transformed[2],
        )
    }
}

/// Apply transformation to multiple points
pub fn transform_points(points: &[Point2D], transform: &TransformMatrix) -> Vec<Point2D> {
    points
        .iter()
        .map(|&p| transform_point(p, transform))
        .collect()
}

/// Invert a transformation matrix
pub fn invert_transform(transform: &TransformMatrix) -> Result<TransformMatrix> {
    // TODO: Replace with proper matrix inversion once ndarray-linalg alternative is available
    // For now, use a simple 3x3 matrix inversion
    invert_3x3_matrix(transform)
        .map_err(|e| VisionError::OperationError(format!("Failed to invert transformation: {}", e)))
}

/// Compose two transformations (T2 * T1)
pub fn compose_transforms(t1: &TransformMatrix, t2: &TransformMatrix) -> TransformMatrix {
    t2.dot(t1)
}

/// Decompose affine transformation into components
pub fn decompose_affine(transform: &TransformMatrix) -> Result<AffineComponents> {
    if transform.shape() != [3, 3] {
        return Err(VisionError::InvalidParameter(
            "Transform must be 3x3 matrix".to_string(),
        ));
    }

    let tx = transform[[0, 2]];
    let ty = transform[[1, 2]];

    let a = transform[[0, 0]];
    let b = transform[[0, 1]];
    let c = transform[[1, 0]];
    let d = transform[[1, 1]];

    let scale_x = (a * a + c * c).sqrt();
    let scale_y = (b * b + d * d).sqrt();

    let rotation = (c / scale_x).atan2(a / scale_x);
    let shear = (a * b + c * d) / (scale_x * scale_y);

    Ok(AffineComponents {
        translation: Point2D::new(tx, ty),
        rotation,
        scale: Point2D::new(scale_x, scale_y),
        shear,
    })
}

/// Components of an affine transformation
#[derive(Debug, Clone)]
pub struct AffineComponents {
    /// Translation vector (dx, dy)
    pub translation: Point2D,
    /// Rotation angle in radians
    pub rotation: f64,
    /// Scale factors (sx, sy)
    pub scale: Point2D,
    /// Shear angle in radians
    pub shear: f64,
}

/// Estimate transformation robustly using RANSAC
pub fn ransac_estimate_transform(
    matches: &[PointMatch],
    transform_type: TransformType,
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    let min_samples = match transform_type {
        TransformType::Rigid => 2,
        TransformType::Similarity => 2,
        TransformType::Affine => 3,
        TransformType::Homography => 4,
    };

    if matches.len() < min_samples {
        return Err(VisionError::InvalidParameter(format!(
            "Need at least {} matches for {:?} transformation",
            min_samples, transform_type
        )));
    }

    let mut _best_transform = identity_transform();
    let mut best_inliers = Vec::new();
    let mut best_cost = f64::INFINITY;

    use rand::prelude::*;
    let mut rng = rand::rng();

    for _iteration in 0..params.ransac_iterations {
        // Sample minimum required points
        let mut sample_indices: Vec<usize> = (0..matches.len()).collect();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(min_samples);

        let sample_matches: Vec<_> = sample_indices.iter().map(|&i| matches[i].clone()).collect();

        // Estimate transformation from sample
        let transform = match transform_type {
            TransformType::Rigid => estimate_rigid_transform(&sample_matches)?,
            TransformType::Similarity => estimate_similarity_transform(&sample_matches)?,
            TransformType::Affine => estimate_affine_transform(&sample_matches)?,
            TransformType::Homography => estimate_homography_transform(&sample_matches)?,
        };

        // Find inliers
        let mut inliers = Vec::new();
        let mut total_error = 0.0;

        for (i, m) in matches.iter().enumerate() {
            let transformed = transform_point(m.source, &transform);
            let error = ((transformed.x - m.target.x).powi(2)
                + (transformed.y - m.target.y).powi(2))
            .sqrt();

            if error < params.ransac_threshold {
                inliers.push(i);
                total_error += error;
            }
        }

        if inliers.len() >= min_samples {
            let cost = total_error / inliers.len() as f64;
            if cost < best_cost {
                best_cost = cost;
                _best_transform = transform;
                best_inliers = inliers;
            }
        }
    }

    if best_inliers.is_empty() {
        return Err(VisionError::OperationError(
            "RANSAC failed to find valid transformation".to_string(),
        ));
    }

    // Refine using all inliers
    let inlier_matches: Vec<_> = best_inliers.iter().map(|&i| matches[i].clone()).collect();

    let refined_transform = match transform_type {
        TransformType::Rigid => estimate_rigid_transform(&inlier_matches)?,
        TransformType::Similarity => estimate_similarity_transform(&inlier_matches)?,
        TransformType::Affine => estimate_affine_transform(&inlier_matches)?,
        TransformType::Homography => estimate_homography_transform(&inlier_matches)?,
    };

    Ok(RegistrationResult {
        transform: refined_transform,
        final_cost: best_cost,
        iterations: params.ransac_iterations,
        converged: true,
        inliers: best_inliers,
    })
}

/// Estimate rigid transformation (translation + rotation)
fn estimate_rigid_transform(matches: &[PointMatch]) -> Result<TransformMatrix> {
    if matches.len() < 2 {
        return Err(VisionError::InvalidParameter(
            "Need at least 2 matches for rigid transformation".to_string(),
        ));
    }

    // Calculate centroids
    let n = matches.len() as f64;
    let source_centroid = Point2D::new(
        matches.iter().map(|m| m.source.x).sum::<f64>() / n,
        matches.iter().map(|m| m.source.y).sum::<f64>() / n,
    );
    let target_centroid = Point2D::new(
        matches.iter().map(|m| m.target.x).sum::<f64>() / n,
        matches.iter().map(|m| m.target.y).sum::<f64>() / n,
    );

    // Calculate rotation using cross-correlation
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syx = 0.0;
    let mut syy = 0.0;

    for m in matches {
        let sx = m.source.x - source_centroid.x;
        let sy = m.source.y - source_centroid.y;
        let tx = m.target.x - target_centroid.x;
        let ty = m.target.y - target_centroid.y;

        sxx += sx * tx;
        sxy += sx * ty;
        syx += sy * tx;
        syy += sy * ty;
    }

    let angle = (sxy - syx).atan2(sxx + syy);
    let cos_a = angle.cos();
    let sin_a = angle.sin();

    // Calculate translation
    let tx = target_centroid.x - (cos_a * source_centroid.x - sin_a * source_centroid.y);
    let ty = target_centroid.y - (sin_a * source_centroid.x + cos_a * source_centroid.y);

    // Construct transformation matrix
    let mut transform = Array2::zeros((3, 3));
    transform[[0, 0]] = cos_a;
    transform[[0, 1]] = -sin_a;
    transform[[0, 2]] = tx;
    transform[[1, 0]] = sin_a;
    transform[[1, 1]] = cos_a;
    transform[[1, 2]] = ty;
    transform[[2, 2]] = 1.0;

    Ok(transform)
}

/// Estimate similarity transformation (translation + rotation + uniform scale)
fn estimate_similarity_transform(matches: &[PointMatch]) -> Result<TransformMatrix> {
    if matches.len() < 2 {
        return Err(VisionError::InvalidParameter(
            "Need at least 2 matches for similarity transformation".to_string(),
        ));
    }

    // Calculate centroids
    let n = matches.len() as f64;
    let source_centroid = Point2D::new(
        matches.iter().map(|m| m.source.x).sum::<f64>() / n,
        matches.iter().map(|m| m.source.y).sum::<f64>() / n,
    );
    let target_centroid = Point2D::new(
        matches.iter().map(|m| m.target.x).sum::<f64>() / n,
        matches.iter().map(|m| m.target.y).sum::<f64>() / n,
    );

    // Calculate scale, rotation using least squares
    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syx = 0.0;
    let mut syy = 0.0;
    let mut source_var = 0.0;

    for m in matches {
        let sx = m.source.x - source_centroid.x;
        let sy = m.source.y - source_centroid.y;
        let tx = m.target.x - target_centroid.x;
        let ty = m.target.y - target_centroid.y;

        sxx += sx * tx;
        sxy += sx * ty;
        syx += sy * tx;
        syy += sy * ty;
        source_var += sx * sx + sy * sy;
    }

    if source_var < 1e-10 {
        return Err(VisionError::OperationError(
            "Source points are collinear".to_string(),
        ));
    }

    let scale = (sxx + syy) / source_var;
    let angle = (sxy - syx).atan2(sxx + syy);

    let cos_a = scale * angle.cos();
    let sin_a = scale * angle.sin();

    // Calculate translation
    let tx = target_centroid.x - (cos_a * source_centroid.x - sin_a * source_centroid.y);
    let ty = target_centroid.y - (sin_a * source_centroid.x + cos_a * source_centroid.y);

    // Construct transformation matrix
    let mut transform = Array2::zeros((3, 3));
    transform[[0, 0]] = cos_a;
    transform[[0, 1]] = -sin_a;
    transform[[0, 2]] = tx;
    transform[[1, 0]] = sin_a;
    transform[[1, 1]] = cos_a;
    transform[[1, 2]] = ty;
    transform[[2, 2]] = 1.0;

    Ok(transform)
}

/// Estimate affine transformation
fn estimate_affine_transform(matches: &[PointMatch]) -> Result<TransformMatrix> {
    if matches.len() < 3 {
        return Err(VisionError::InvalidParameter(
            "Need at least 3 matches for affine transformation".to_string(),
        ));
    }

    // use ndarray_linalg::LeastSquaresSvd; // TODO: Replace with alternative

    let n = matches.len();
    let mut a = Array2::zeros((2 * n, 6));
    let mut b = Array1::zeros(2 * n);

    for (i, m) in matches.iter().enumerate() {
        let row1 = 2 * i;
        let row2 = 2 * i + 1;

        // First equation: target.x = a*source.x + b*source.y + c
        a[[row1, 0]] = m.source.x;
        a[[row1, 1]] = m.source.y;
        a[[row1, 2]] = 1.0;
        b[row1] = m.target.x;

        // Second equation: target.y = d*source.x + e*source.y + f
        a[[row2, 3]] = m.source.x;
        a[[row2, 4]] = m.source.y;
        a[[row2, 5]] = 1.0;
        b[row2] = m.target.y;
    }

    // Use scirs2-linalg's least squares solver
    let result = lstsq(&a.view(), &b.view(), None).map_err(|e| {
        VisionError::OperationError(format!("Failed to solve affine system: {}", e))
    })?;

    let params = result.x;

    let mut transform = Array2::zeros((3, 3));
    transform[[0, 0]] = params[0];
    transform[[0, 1]] = params[1];
    transform[[0, 2]] = params[2];
    transform[[1, 0]] = params[3];
    transform[[1, 1]] = params[4];
    transform[[1, 2]] = params[5];
    transform[[2, 2]] = 1.0;

    Ok(transform)
}

/// Normalize points for homography estimation
fn normalize_points_homography(points: Vec<Point2D>) -> (Vec<Point2D>, TransformMatrix) {
    let n = points.len() as f64;

    // Calculate centroid
    let mut cx = 0.0;
    let mut cy = 0.0;
    for p in &points {
        cx += p.x;
        cy += p.y;
    }
    cx /= n;
    cy /= n;

    // Calculate average distance from centroid
    let mut avg_dist = 0.0;
    for p in &points {
        let dx = p.x - cx;
        let dy = p.y - cy;
        avg_dist += (dx * dx + dy * dy).sqrt();
    }
    avg_dist /= n;

    // Scale factor to make average distance sqrt(2)
    let scale = if avg_dist > 1e-10 {
        2.0_f64.sqrt() / avg_dist
    } else {
        1.0
    };

    // Create normalization matrix
    let mut t = Array2::eye(3);
    t[[0, 0]] = scale;
    t[[1, 1]] = scale;
    t[[0, 2]] = -scale * cx;
    t[[1, 2]] = -scale * cy;

    // Normalize points
    let mut norm_points = Vec::new();
    for p in points {
        norm_points.push(Point2D::new(scale * (p.x - cx), scale * (p.y - cy)));
    }

    (norm_points, t)
}

/// Estimate homography transformation
fn estimate_homography_transform(matches: &[PointMatch]) -> Result<TransformMatrix> {
    if matches.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "Need at least 4 matches for homography transformation".to_string(),
        ));
    }

    // Check if all points are very close to their targets (identity transformation)
    let mut is_identity = true;
    for m in matches {
        let dx = m.source.x - m.target.x;
        let dy = m.source.y - m.target.y;
        if dx.abs() > 1e-10 || dy.abs() > 1e-10 {
            is_identity = false;
            break;
        }
    }

    if is_identity {
        return Ok(identity_transform());
    }

    // Use Direct Linear Transform (DLT) algorithm for full homography estimation
    // This avoids SVD issues while still providing full 8-parameter homography

    // First normalize the points for numerical stability
    let (norm_source, t1) = normalize_points_homography(matches.iter().map(|m| m.source).collect());
    let (norm_target, t2) = normalize_points_homography(matches.iter().map(|m| m.target).collect());

    // Build the constraint matrix for DLT
    // For each correspondence, we get 2 equations
    let n = matches.len();
    let mut a_mat = Array2::zeros((2 * n, 9));

    for (i, (src, tgt)) in norm_source.iter().zip(norm_target.iter()).enumerate() {
        let sx = src.x;
        let sy = src.y;
        let tx = tgt.x;
        let ty = tgt.y;

        // First equation: -sx*h11 - sy*h12 - h13 + tx*sx*h31 + tx*sy*h32 + tx*h33 = 0
        a_mat[[2 * i, 0]] = -sx;
        a_mat[[2 * i, 1]] = -sy;
        a_mat[[2 * i, 2]] = -1.0;
        a_mat[[2 * i, 6]] = tx * sx;
        a_mat[[2 * i, 7]] = tx * sy;
        a_mat[[2 * i, 8]] = tx;

        // Second equation: -sx*h21 - sy*h22 - h23 + ty*sx*h31 + ty*sy*h32 + ty*h33 = 0
        a_mat[[2 * i + 1, 3]] = -sx;
        a_mat[[2 * i + 1, 4]] = -sy;
        a_mat[[2 * i + 1, 5]] = -1.0;
        a_mat[[2 * i + 1, 6]] = ty * sx;
        a_mat[[2 * i + 1, 7]] = ty * sy;
        a_mat[[2 * i + 1, 8]] = ty;
    }

    // Find the null space of A using least squares with regularization
    // We want to minimize ||Ah|| subject to ||h|| = 1
    // Add regularization to avoid h33 = 0
    let mut ata = a_mat.t().dot(&a_mat);

    // Add small regularization to ensure numerical stability
    for i in 0..9 {
        ata[[i, i]] += 1e-10;
    }

    // Find eigenvector corresponding to smallest eigenvalue
    // Since we can't use full eigendecomposition, use power iteration on the inverse
    let mut h_vec = Array1::from_elem(9, 1.0 / 3.0); // Initial guess
    h_vec[8] = 1.0; // Bias towards h33 = 1

    // Use iterative refinement to find approximate solution
    for _ in 0..20 {
        // Solve (A^T A + λI) h_new = h_old to get direction
        let b = h_vec.clone();
        match solve(&ata.view(), &b.view(), None) {
            Ok(h_new) => {
                // Normalize
                let norm = h_new.dot(&h_new).sqrt();
                if norm > 1e-10 {
                    h_vec = h_new / norm;
                }
            }
            Err(_) => {
                // If solve fails, fall back to simpler approach
                break;
            }
        }
    }

    // Ensure h33 is positive
    if h_vec[8] < 0.0 {
        h_vec = -h_vec;
    }

    // Reshape to 3x3 matrix
    let mut h_matrix = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            h_matrix[[i, j]] = h_vec[i * 3 + j] / h_vec[8]; // Normalize by h33
        }
    }

    // Denormalize
    let t2_inv = invert_3x3_matrix(&t2)?;
    let h_denorm = t2_inv.dot(&h_matrix.dot(&t1));

    Ok(h_denorm)
}

/// Simple 3x3 matrix inversion for TransformMatrix
/// TODO: Replace with proper implementation from linear algebra library
fn invert_3x3_matrix(matrix: &TransformMatrix) -> Result<TransformMatrix> {
    if matrix.shape() != [3, 3] {
        return Err(VisionError::InvalidParameter(
            "Matrix must be 3x3".to_string(),
        ));
    }

    // Compute determinant
    let det = matrix[[0, 0]] * (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]])
        - matrix[[0, 1]] * (matrix[[1, 0]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 0]])
        + matrix[[0, 2]] * (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]);

    if det.abs() < 1e-10 {
        return Err(VisionError::OperationError(
            "Matrix is singular, cannot invert".to_string(),
        ));
    }

    let mut inv = Array2::zeros((3, 3));

    // Compute adjugate matrix
    inv[[0, 0]] = (matrix[[1, 1]] * matrix[[2, 2]] - matrix[[1, 2]] * matrix[[2, 1]]) / det;
    inv[[0, 1]] = (matrix[[0, 2]] * matrix[[2, 1]] - matrix[[0, 1]] * matrix[[2, 2]]) / det;
    inv[[0, 2]] = (matrix[[0, 1]] * matrix[[1, 2]] - matrix[[0, 2]] * matrix[[1, 1]]) / det;
    inv[[1, 0]] = (matrix[[1, 2]] * matrix[[2, 0]] - matrix[[1, 0]] * matrix[[2, 2]]) / det;
    inv[[1, 1]] = (matrix[[0, 0]] * matrix[[2, 2]] - matrix[[0, 2]] * matrix[[2, 0]]) / det;
    inv[[1, 2]] = (matrix[[0, 2]] * matrix[[1, 0]] - matrix[[0, 0]] * matrix[[1, 2]]) / det;
    inv[[2, 0]] = (matrix[[1, 0]] * matrix[[2, 1]] - matrix[[1, 1]] * matrix[[2, 0]]) / det;
    inv[[2, 1]] = (matrix[[0, 1]] * matrix[[2, 0]] - matrix[[0, 0]] * matrix[[2, 1]]) / det;
    inv[[2, 2]] = (matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]) / det;

    Ok(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_transformation() {
        let transform = identity_transform();
        let point = Point2D::new(1.0, 2.0);
        let transformed = transform_point(point, &transform);

        assert!((transformed.x - point.x).abs() < 1e-10);
        assert!((transformed.y - point.y).abs() < 1e-10);
    }

    #[test]
    fn test_rigid_transform_estimation() {
        let matches = vec![
            PointMatch {
                source: Point2D::new(0.0, 0.0),
                target: Point2D::new(1.0, 1.0),
                confidence: 1.0,
            },
            PointMatch {
                source: Point2D::new(1.0, 0.0),
                target: Point2D::new(1.0, 2.0),
                confidence: 1.0,
            },
        ];

        let transform = estimate_rigid_transform(&matches).unwrap();

        // Verify transformation
        let transformed1 = transform_point(matches[0].source, &transform);
        let transformed2 = transform_point(matches[1].source, &transform);

        assert!((transformed1.x - matches[0].target.x).abs() < 1e-10);
        assert!((transformed1.y - matches[0].target.y).abs() < 1e-10);
        assert!((transformed2.x - matches[1].target.x).abs() < 1e-10);
        assert!((transformed2.y - matches[1].target.y).abs() < 1e-10);
    }

    #[test]
    fn test_affine_transform_estimation() {
        let matches = vec![
            PointMatch {
                source: Point2D::new(0.0, 0.0),
                target: Point2D::new(1.0, 2.0),
                confidence: 1.0,
            },
            PointMatch {
                source: Point2D::new(1.0, 0.0),
                target: Point2D::new(3.0, 3.0),
                confidence: 1.0,
            },
            PointMatch {
                source: Point2D::new(0.0, 1.0),
                target: Point2D::new(2.0, 4.0),
                confidence: 1.0,
            },
        ];

        let transform = estimate_affine_transform(&matches).unwrap();

        // Verify transformation
        for m in &matches {
            let transformed = transform_point(m.source, &transform);
            assert!((transformed.x - m.target.x).abs() < 1e-10);
            assert!((transformed.y - m.target.y).abs() < 1e-10);
        }
    }

    #[test]
    fn test_transform_composition() {
        let t1 = identity_transform();
        let mut t2 = identity_transform();
        t2[[0, 2]] = 1.0; // Translation

        let composed = compose_transforms(&t1, &t2);
        assert_eq!(composed[[0, 2]], 1.0);
    }

    #[test]
    fn test_transform_inversion() {
        let mut transform = identity_transform();
        transform[[0, 2]] = 1.0; // Translation
        transform[[1, 2]] = 2.0;

        let inverse = invert_transform(&transform).unwrap();
        let composed = compose_transforms(&transform, &inverse);

        // Should be close to identity
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((composed[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }
}
