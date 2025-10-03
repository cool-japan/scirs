//! Non-rigid (deformable) registration algorithms
//!
//! This module provides non-rigid registration using thin plate splines (TPS) and other
//! deformable transformation models.

use crate::error::{Result, VisionError};
use crate::registration::{identity_transform, Point2D, RegistrationParams, RegistrationResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Simple least squares solver result
#[derive(Debug)]
pub struct LstsqResult {
    /// Solution vector
    pub x: Array1<f64>,
}

/// Simple least squares solver (A * x = b)
/// Returns the solution x that minimizes ||A * x - b||^2
#[allow(dead_code)]
fn lstsq(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
    _rcond: Option<f64>,
) -> std::result::Result<LstsqResult, String> {
    let (m, n) = a.dim();

    if m != b.len() {
        return Err("Matrix dimensions don't match".to_string());
    }

    // For overdetermined systems (m >= n), use normal equations: A^T * A * x = A^T * b
    if m >= n {
        // Compute A^T
        let at = a.t();

        // Compute A^T * A
        let ata = at.dot(a);

        // Compute A^T * b
        let atb = at.dot(b);

        // Solve the system using simple Gaussian elimination
        let x = solve_linear_system(&ata.view(), &atb.view())?;

        Ok(LstsqResult { x })
    } else {
        // Underdetermined system - use minimum norm solution
        // For now, just return a zero solution
        Ok(LstsqResult {
            x: Array1::zeros(n),
        })
    }
}

/// Simple linear system solver using Gaussian elimination
#[allow(dead_code)]
fn solve_linear_system(
    a: &ArrayView2<f64>,
    b: &ArrayView1<f64>,
) -> std::result::Result<Array1<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n {
        return Err("Matrix must be square and match vector dimension".to_string());
    }

    // Create augmented matrix [A | b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
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
            for j in 0..=n {
                let tmp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Check for singular matrix
        if aug[[i, i]].abs() < 1e-14 {
            return Err("Matrix is singular".to_string());
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug[[k, i]] / aug[[i, i]];
            for j in i..=n {
                aug[[k, j]] -= factor * aug[[i, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }

    Ok(x)
}

/// Thin Plate Spline transformation for non-rigid registration
#[derive(Debug, Clone)]
pub struct ThinPlateSpline {
    /// Control points (landmarks)
    control_points: Vec<Point2D>,
    /// Weights for radial basis functions
    weights: Array2<f64>,
    /// Affine transformation parameters
    affine_params: Array2<f64>,
}

impl ThinPlateSpline {
    /// Create a new TPS transformation from control points and targets
    pub fn new(source_points: &[Point2D], target_points: &[Point2D]) -> Result<Self> {
        if source_points.len() != target_points.len() {
            return Err(VisionError::InvalidParameter(
                "Source and target _points must have same length".to_string(),
            ));
        }

        if source_points.len() < 3 {
            return Err(VisionError::InvalidParameter(
                "Need at least 3 control _points for TPS".to_string(),
            ));
        }

        let n = source_points.len();

        // Build the TPS system matrix
        let mut k_matrix = Array2::zeros((n + 3, n + 3));

        // Fill K matrix (radial basis function values)
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist_sq = (source_points[i].x - source_points[j].x).powi(2)
                        + (source_points[i].y - source_points[j].y).powi(2);
                    if dist_sq > 0.0 {
                        k_matrix[[i, j]] = dist_sq * (dist_sq.ln());
                    }
                }
            }
        }

        // Fill P matrix (affine part)
        for i in 0..n {
            k_matrix[[i, n]] = 1.0;
            k_matrix[[i, n + 1]] = source_points[i].x;
            k_matrix[[i, n + 2]] = source_points[i].y;

            k_matrix[[n, i]] = 1.0;
            k_matrix[[n + 1, i]] = source_points[i].x;
            k_matrix[[n + 2, i]] = source_points[i].y;
        }

        // Create target vectors for x and y coordinates
        let mut target_x = Array1::zeros(n + 3);
        let mut target_y = Array1::zeros(n + 3);

        for i in 0..n {
            target_x[i] = target_points[i].x;
            target_y[i] = target_points[i].y;
        }

        // Use scirs2-linalg's least squares solver
        let result_x = lstsq(&k_matrix.view(), &target_x.view(), None)
            .map_err(|e| VisionError::OperationError(format!("TPS solve failed for x: {e}")))?;
        let weights_x = result_x.x;

        let result_y = lstsq(&k_matrix.view(), &target_y.view(), None)
            .map_err(|e| VisionError::OperationError(format!("TPS solve failed for y: {e}")))?;
        let weights_y = result_y.x;

        // Extract weights and affine parameters
        let mut weights = Array2::zeros((n, 2));
        let mut affine_params = Array2::zeros((3, 2));

        for i in 0..n {
            weights[[i, 0]] = weights_x[i];
            weights[[i, 1]] = weights_y[i];
        }

        for i in 0..3 {
            affine_params[[i, 0]] = weights_x[n + i];
            affine_params[[i, 1]] = weights_y[n + i];
        }

        Ok(ThinPlateSpline {
            control_points: source_points.to_vec(),
            weights,
            affine_params,
        })
    }

    /// Transform a point using the TPS transformation
    pub fn transform_point(&self, point: Point2D) -> Point2D {
        let mut result_x = self.affine_params[[0, 0]]
            + self.affine_params[[1, 0]] * point.x
            + self.affine_params[[2, 0]] * point.y;

        let mut result_y = self.affine_params[[0, 1]]
            + self.affine_params[[1, 1]] * point.x
            + self.affine_params[[2, 1]] * point.y;

        // Add radial basis function contributions
        for (i, &control_point) in self.control_points.iter().enumerate() {
            let dist_sq = (point.x - control_point.x).powi(2) + (point.y - control_point.y).powi(2);

            if dist_sq > 0.0 {
                let rbf_value = dist_sq * (dist_sq.ln());
                result_x += self.weights[[i, 0]] * rbf_value;
                result_y += self.weights[[i, 1]] * rbf_value;
            }
        }

        Point2D::new(result_x, result_y)
    }

    /// Transform multiple points
    pub fn transform_points(&self, points: &[Point2D]) -> Vec<Point2D> {
        points.iter().map(|&p| self.transform_point(p)).collect()
    }
}

/// Non-rigid registration using Thin Plate Splines
#[allow(dead_code)]
pub fn register_non_rigid_points(
    source_points: &[(f64, f64)],
    target_points: &[(f64, f64)],
    _params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source_points.len() != target_points.len() {
        return Err(VisionError::InvalidParameter(
            "Source and target point sets must have the same length".to_string(),
        ));
    }

    if source_points.len() < 3 {
        return Err(VisionError::InvalidParameter(
            "Need at least 3 point correspondences for non-rigid registration".to_string(),
        ));
    }

    // Convert to Point2D
    let source_pts: Vec<Point2D> = source_points
        .iter()
        .map(|&(x, y)| Point2D::new(x, y))
        .collect();

    let target_pts: Vec<Point2D> = target_points
        .iter()
        .map(|&(x, y)| Point2D::new(x, y))
        .collect();

    // Create TPS transformation
    let tps = ThinPlateSpline::new(&source_pts, &target_pts)?;

    // Calculate registration error
    let mut total_error = 0.0;
    for (i, &source_pt) in source_pts.iter().enumerate() {
        let transformed = tps.transform_point(source_pt);
        let target_pt = target_pts[i];
        let error =
            ((transformed.x - target_pt.x).powi(2) + (transformed.y - target_pt.y).powi(2)).sqrt();
        total_error += error;
    }
    let final_cost = total_error / source_pts.len() as f64;

    // For non-rigid transformations, we return an identity matrix as the transform
    // since the actual transformation is encoded in the TPS parameters
    // In a real implementation, you might want to store the TPS parameters
    let transform = identity_transform();

    Ok(RegistrationResult {
        transform,
        final_cost,
        iterations: 1,
        converged: true,
        inliers: (0..source_points.len()).collect(),
    })
}

/// Non-rigid registration with regularization
#[allow(dead_code)]
pub fn register_non_rigid_regularized(
    source_points: &[(f64, f64)],
    target_points: &[(f64, f64)],
    regularization_weight: f64,
    _params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source_points.len() != target_points.len() {
        return Err(VisionError::InvalidParameter(
            "Source and target point sets must have the same length".to_string(),
        ));
    }

    if source_points.len() < 3 {
        return Err(VisionError::InvalidParameter(
            "Need at least 3 point correspondences for non-rigid registration".to_string(),
        ));
    }

    let source_pts: Vec<Point2D> = source_points
        .iter()
        .map(|&(x, y)| Point2D::new(x, y))
        .collect();

    let target_pts: Vec<Point2D> = target_points
        .iter()
        .map(|&(x, y)| Point2D::new(x, y))
        .collect();

    let n = source_pts.len();

    // Build regularized TPS system
    let mut k_matrix = Array2::zeros((n + 3, n + 3));

    // Fill K matrix with regularization
    for i in 0..n {
        for j in 0..n {
            if i == j {
                k_matrix[[i, j]] = regularization_weight;
            } else {
                let dist_sq = (source_pts[i].x - source_pts[j].x).powi(2)
                    + (source_pts[i].y - source_pts[j].y).powi(2);
                if dist_sq > 0.0 {
                    k_matrix[[i, j]] = dist_sq * (dist_sq.ln());
                }
            }
        }
    }

    // Fill P matrix (affine part)
    for i in 0..n {
        k_matrix[[i, n]] = 1.0;
        k_matrix[[i, n + 1]] = source_pts[i].x;
        k_matrix[[i, n + 2]] = source_pts[i].y;

        k_matrix[[n, i]] = 1.0;
        k_matrix[[n + 1, i]] = source_pts[i].x;
        k_matrix[[n + 2, i]] = source_pts[i].y;
    }

    // Create target vectors
    let mut target_x = Array1::zeros(n + 3);
    let mut target_y = Array1::zeros(n + 3);

    for i in 0..n {
        target_x[i] = target_pts[i].x;
        target_y[i] = target_pts[i].y;
    }

    // Use scirs2-linalg's least squares solver
    let result_x = lstsq(&k_matrix.view(), &target_x.view(), None).map_err(|e| {
        VisionError::OperationError(format!("Regularized TPS solve failed for x: {e}"))
    })?;
    let weights_x = result_x.x;

    let result_y = lstsq(&k_matrix.view(), &target_y.view(), None).map_err(|e| {
        VisionError::OperationError(format!("Regularized TPS solve failed for y: {e}"))
    })?;
    let weights_y = result_y.x;

    // Extract weights and affine parameters
    let mut weights = Array2::zeros((n, 2));
    let mut affine_params = Array2::zeros((3, 2));

    for i in 0..n {
        weights[[i, 0]] = weights_x[i];
        weights[[i, 1]] = weights_y[i];
    }

    affine_params[[0, 0]] = weights_x[n];
    affine_params[[1, 0]] = weights_x[n + 1];
    affine_params[[2, 0]] = weights_x[n + 2];
    affine_params[[0, 1]] = weights_y[n];
    affine_params[[1, 1]] = weights_y[n + 1];
    affine_params[[2, 1]] = weights_y[n + 2];

    // Create TPS
    let tps = ThinPlateSpline {
        control_points: source_pts.clone(),
        weights,
        affine_params,
    };

    // Calculate error
    let mut total_error = 0.0;
    for (i, &source_pt) in source_pts.iter().enumerate() {
        let transformed = tps.transform_point(source_pt);
        let target_pt = target_pts[i];
        let error =
            ((transformed.x - target_pt.x).powi(2) + (transformed.y - target_pt.y).powi(2)).sqrt();
        total_error += error;
    }
    let final_cost = total_error / source_pts.len() as f64;

    Ok(RegistrationResult {
        transform: identity_transform(),
        final_cost,
        iterations: 1,
        converged: true,
        inliers: (0..source_points.len()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_tps_identity_transformation() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
        ];

        let tps = ThinPlateSpline::new(&points, &points).unwrap();

        // Test that identity transformation works
        for &point in &points {
            let transformed = tps.transform_point(point);
            assert_relative_eq!(transformed.x, point.x, epsilon = 1e-10);
            assert_relative_eq!(transformed.y, point.y, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tps_simple_deformation() {
        let source = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
        ];

        let target = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.1, 0.0), // Slight stretch
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.1), // Slight stretch
        ];

        let tps = ThinPlateSpline::new(&source, &target).unwrap();

        // Test that control points map correctly
        for (i, &source_pt) in source.iter().enumerate() {
            let transformed = tps.transform_point(source_pt);
            assert_relative_eq!(transformed.x, target[i].x, epsilon = 1e-8);
            assert_relative_eq!(transformed.y, target[i].y, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_non_rigid_registration_identical_points() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_non_rigid_points(&points, &points, &params).unwrap();

        // Should have zero error for identical points
        assert!(result.final_cost < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_non_rigid_registration_deformation() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let target = vec![(0.0, 0.0), (1.1, 0.1), (0.1, 1.1), (1.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_non_rigid_points(&source, &target, &params).unwrap();

        // Should find a valid transformation
        assert!(result.final_cost < 1.0);
        assert!(!result.inliers.is_empty());
    }

    #[test]
    fn test_non_rigid_registration_insufficient_points() {
        let source = vec![(0.0, 0.0), (1.0, 0.0)];
        let target = vec![(1.0, 1.0), (2.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_non_rigid_points(&source, &target, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_regularized_non_rigid_registration() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let target = vec![(0.0, 0.0), (1.1, 0.1), (0.1, 1.1), (1.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_non_rigid_regularized(&source, &target, 0.01, &params).unwrap();

        // Should find a valid transformation
        assert!(result.final_cost < 2.0);
        assert!(result.converged);
    }
}
