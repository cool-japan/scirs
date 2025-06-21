//! Homography registration algorithms
//!
//! This module provides homography (perspective) transformation registration.

use crate::error::{Result, VisionError};
use crate::registration::{
    estimate_homography_transform, identity_transform, ransac_estimate_transform, Point2D,
    PointMatch, RegistrationParams, RegistrationResult, TransformType,
};

/// Homography registration using point matches
pub fn register_homography_points(
    source_points: &[(f64, f64)],
    target_points: &[(f64, f64)],
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source_points.len() != target_points.len() {
        return Err(VisionError::InvalidParameter(
            "Source and target point sets must have the same length".to_string(),
        ));
    }

    if source_points.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "Need at least 4 point correspondences for homography registration".to_string(),
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
    if matches.len() >= 8 && params.ransac_iterations > 0 {
        ransac_estimate_transform(&matches, TransformType::Homography, params)
    } else {
        // Direct estimation for small point sets
        let transform = estimate_homography_transform(&matches)?;

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

/// Homography registration with iterative refinement
pub fn register_homography_iterative(
    source_points: &[(f64, f64)],
    target_points: &[(f64, f64)],
    params: &RegistrationParams,
) -> Result<RegistrationResult> {
    if source_points.len() != target_points.len() {
        return Err(VisionError::InvalidParameter(
            "Source and target point sets must have the same length".to_string(),
        ));
    }

    if source_points.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "Need at least 4 point correspondences for homography registration".to_string(),
        ));
    }

    // Convert points to matches
    let mut matches: Vec<PointMatch> = source_points
        .iter()
        .zip(target_points.iter())
        .map(|(&(sx, sy), &(tx, ty))| PointMatch {
            source: Point2D::new(sx, sy),
            target: Point2D::new(tx, ty),
            confidence: 1.0,
        })
        .collect();

    let mut current_transform = identity_transform();
    let mut prev_cost = f64::INFINITY;
    let mut converged = false;

    for _iteration in 0..params.max_iterations {
        // Estimate transformation from current matches
        let transform = estimate_homography_transform(&matches)?;

        // Calculate cost
        let mut total_error = 0.0;
        for m in &matches {
            let transformed = crate::registration::transform_point(m.source, &transform);
            let error = ((transformed.x - m.target.x).powi(2)
                + (transformed.y - m.target.y).powi(2))
            .sqrt();
            total_error += error;
        }
        let current_cost = total_error / matches.len() as f64;

        // Check convergence
        if (prev_cost - current_cost).abs() < params.tolerance {
            converged = true;
            current_transform = transform;
            prev_cost = current_cost;
            break;
        }

        prev_cost = current_cost;
        current_transform = transform;

        // Update point weights based on residuals (simple weighting scheme)
        for m in &mut matches {
            let transformed = crate::registration::transform_point(m.source, &current_transform);
            let error = ((transformed.x - m.target.x).powi(2)
                + (transformed.y - m.target.y).powi(2))
            .sqrt();

            // Decrease confidence for outliers
            if error > params.ransac_threshold {
                m.confidence *= 0.5;
            } else {
                m.confidence = (m.confidence + 1.0) / 2.0; // Increase confidence for inliers
            }
        }
    }

    // Find inliers based on final residuals
    let mut inliers = Vec::new();
    for (i, m) in matches.iter().enumerate() {
        let transformed = crate::registration::transform_point(m.source, &current_transform);
        let error =
            ((transformed.x - m.target.x).powi(2) + (transformed.y - m.target.y).powi(2)).sqrt();
        if error < params.ransac_threshold {
            inliers.push(i);
        }
    }

    Ok(RegistrationResult {
        transform: current_transform,
        final_cost: prev_cost,
        iterations: if converged {
            params.max_iterations - (params.max_iterations - 1)
        } else {
            params.max_iterations
        },
        converged,
        inliers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_homography_registration_identical_points() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let params = RegistrationParams::default();

        let result = register_homography_points(&points, &points, &params).unwrap();

        // Should be identity transformation with zero error
        assert!(result.final_cost < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_homography_registration_translation() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let target = vec![(2.0, 3.0), (3.0, 3.0), (2.0, 4.0), (3.0, 4.0)];
        let params = RegistrationParams::default();

        let result = register_homography_points(&source, &target, &params).unwrap();

        // Should find translation (2, 3)
        assert!(result.final_cost < 1e-10);
        assert_relative_eq!(result.transform[[0, 2]], 2.0, epsilon = 1e-10);
        assert_relative_eq!(result.transform[[1, 2]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_homography_registration_perspective() {
        // Test a perspective transformation (trapezoid)
        let source = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let target = vec![(0.0, 0.0), (2.0, 0.0), (1.8, 1.0), (0.2, 1.0)]; // Perspective distortion
        let params = RegistrationParams::default();

        let result = register_homography_points(&source, &target, &params).unwrap();

        // Should find a valid transformation with low error
        assert!(result.final_cost < 1e-8);

        // Verify transformation by applying it to source points
        for (i, &(sx, sy)) in source.iter().enumerate() {
            let source_pt = Point2D::new(sx, sy);
            let transformed = crate::registration::transform_point(source_pt, &result.transform);
            let (tx, ty) = target[i];

            assert_relative_eq!(transformed.x, tx, epsilon = 1e-8);
            assert_relative_eq!(transformed.y, ty, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_homography_registration_insufficient_points() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let target = vec![(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)];
        let params = RegistrationParams::default();

        let result = register_homography_points(&source, &target, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_homography_registration_mismatched_lengths() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        let target = vec![(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)];
        let params = RegistrationParams::default();

        let result = register_homography_points(&source, &target, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_homography_iterative_registration() {
        let source = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let target = vec![(0.1, 0.1), (2.1, 0.1), (1.9, 1.1), (0.1, 1.1)]; // Slight perspective distortion
        let params = RegistrationParams::default();

        let result = register_homography_iterative(&source, &target, &params).unwrap();

        // Should find a valid transformation
        assert!(result.final_cost < 1e-6);
        assert!(!result.inliers.is_empty());
    }
}
