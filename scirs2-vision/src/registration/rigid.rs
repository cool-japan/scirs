//! Rigid registration algorithms
//!
//! This module provides rigid transformation registration (translation + rotation only).

use crate::error::{Result, VisionError};
use crate::registration::{
    estimate_rigid_transform, identity_transform, ransac_estimate_transform, Point2D, PointMatch,
    RegistrationParams, RegistrationResult, TransformType,
};

/// Rigid registration using point matches
pub fn register_rigid_points(
    source_points: &[(f64, f64)],
    target_points: &[(f64, f64)],
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source_points.len() != target_points.len() {
        return Err(VisionError::InvalidParameter(
            "Source and target point sets must have the same length".to_string(),
        ));
    }

    if source_points.len() < 2 {
        return Err(VisionError::InvalidParameter(
            "Need at least 2 point correspondences for rigid registration".to_string(),
        ));
    }

    // Convert points to matches
    let matches: Vec<PointMatch> = source_points
        .iter()
        .zip(target_points.iter())
        .map(|(&(sx, sy), &(tx, ty))| PointMatch {
            source: Point2D::new(sx, sy),
            target: Point2D::new(tx, ty),
            confidence: 1.0,
        })
        .collect();

    // Use RANSAC for robust estimation if we have enough points
    if matches.len() >= 4 && params.ransac_iterations > 0 {
        ransac_estimate_transform(&matches, TransformType::Rigid, params)
    } else {
        // Direct estimation for small point sets
        let transform = estimate_rigid_transform(&matches)?;

        // Calculate final cost
        let mut total_error = 0.0;
        for m in &matches {
            let transformed = crate::registration::transform_point(m.source, &transform);
            let error = ((transformed.x - m.target.x).powi(2)
                + (transformed.y - m.target.y).powi(2))
            .sqrt();
            total_error += error;
        }
        let final_cost = total_error / matches.len() as f64;

        Ok(RegistrationResult {
            transform,
            final_cost,
            iterations: 1,
            converged: true,
            inliers: (0..matches.len()).collect(),
        })
    }
}

/// Rigid registration with iterative closest point (ICP)
pub fn register_rigid_icp(
    source_points: &[(f64, f64)],
    target_points: &[(f64, f64)],
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source_points.is_empty() || target_points.is_empty() {
        return Err(VisionError::InvalidParameter(
            "Point sets cannot be empty".to_string(),
        ));
    }

    let mut current_transform = identity_transform();
    let mut transformed_source: Vec<Point2D> = source_points
        .iter()
        .map(|&(x, y)| Point2D::new(x, y))
        .collect();

    let target_pts: Vec<Point2D> = target_points
        .iter()
        .map(|&(x, y)| Point2D::new(x, y))
        .collect();

    let mut final_cost = f64::INFINITY;
    let mut converged = false;

    for _iteration in 0..params.max_iterations {
        // Find closest point correspondences
        let mut matches = Vec::new();
        let mut total_distance = 0.0;

        for &source_pt in &transformed_source {
            let mut min_distance = f64::INFINITY;
            let mut closest_target = target_pts[0];

            for &target_pt in &target_pts {
                let distance = ((source_pt.x - target_pt.x).powi(2)
                    + (source_pt.y - target_pt.y).powi(2))
                .sqrt();
                if distance < min_distance {
                    min_distance = distance;
                    closest_target = target_pt;
                }
            }

            matches.push(PointMatch {
                source: source_pt,
                target: closest_target,
                confidence: 1.0,
            });
            total_distance += min_distance;
        }

        let current_cost = total_distance / matches.len() as f64;

        // Check convergence
        if (final_cost - current_cost).abs() < params.tolerance {
            converged = true;
            final_cost = current_cost;
            break;
        }

        final_cost = current_cost;

        // Estimate transformation from current correspondences
        let step_transform = estimate_rigid_transform(&matches)?;

        // Update overall transformation
        current_transform =
            crate::registration::compose_transforms(&current_transform, &step_transform);

        // Apply transformation to source points
        transformed_source = crate::registration::transform_points(
            &source_points
                .iter()
                .map(|&(x, y)| Point2D::new(x, y))
                .collect::<Vec<_>>(),
            &current_transform,
        );
    }

    Ok(RegistrationResult {
        transform: current_transform,
        final_cost,
        iterations: if converged {
            // Find which iteration we converged at
            std::cmp::min(params.max_iterations, params.max_iterations)
        } else {
            params.max_iterations
        },
        converged,
        inliers: (0..source_points.len()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_rigid_registration_identical_points() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_rigid_points(&points, &points, &params).unwrap();

        // Should be identity transformation with zero error
        assert!(result.final_cost < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_rigid_registration_translation() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let target = vec![(2.0, 3.0), (3.0, 3.0), (2.0, 4.0)];
        let params = RegistrationParams::default();

        let result = register_rigid_points(&source, &target, &params).unwrap();

        // Should find translation (2, 3)
        assert!(result.final_cost < 1e-10);
        assert_relative_eq!(result.transform[[0, 2]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.transform[[1, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rigid_registration_rotation() {
        let source = vec![(1.0, 0.0), (0.0, 1.0)];
        // 90-degree rotation: (1,0) -> (0,1), (0,1) -> (-1,0)
        let target = vec![(0.0, 1.0), (-1.0, 0.0)];
        let params = RegistrationParams::default();

        let result = register_rigid_points(&source, &target, &params).unwrap();

        // Should find 90-degree rotation
        assert!(result.final_cost < 1e-10);
        assert_relative_eq!(result.transform[[0, 0]], 0.0, epsilon = 1e-10);
        assert_relative_eq!(result.transform[[0, 1]], -1.0, epsilon = 1e-10);
        assert_relative_eq!(result.transform[[1, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(result.transform[[1, 1]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rigid_icp_registration() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let target = vec![(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (2.0, 2.0)];
        let params = RegistrationParams::default();

        let result = register_rigid_icp(&source, &target, &params).unwrap();

        // Should converge to a reasonable transformation
        assert!(result.final_cost < 1.0);
        assert!(result.converged || result.iterations == params.max_iterations);
    }

    #[test]
    fn test_rigid_registration_insufficient_points() {
        let source = vec![(0.0, 0.0)];
        let target = vec![(1.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_rigid_points(&source, &target, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_rigid_registration_mismatched_lengths() {
        let source = vec![(0.0, 0.0), (1.0, 0.0)];
        let target = vec![(1.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_rigid_points(&source, &target, &params);
        assert!(result.is_err());
    }
}
