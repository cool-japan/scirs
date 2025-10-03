//! RANSAC-based homography estimation
//!
//! This module provides robust homography estimation using RANSAC (Random Sample Consensus)
//! to handle noisy correspondence data with outliers, commonly found in real-world
//! computer vision applications.

use super::core::{PerspectiveTransform, RansacParams, RansacResult};
use crate::error::{Result, VisionError};
use scirs2_core::random::Random;
use scirs2_core::random::{rngs::StdRng, seq::SliceRandom, RngCore, SeedableRng};

/// Find a homography transformation using RANSAC
///
/// # Arguments
///
/// * `srcpoints` - Source points [(x, y), ...]
/// * `dst_points` - Destination points [(x', y'), ...]
/// * `params` - RANSAC parameters
///
/// # Returns
///
/// * Result containing the best homography and inlier information
///
/// # Robustness
///
/// RANSAC provides robustness against outliers by iteratively sampling
/// minimal sets of correspondences and evaluating consensus.
pub fn find_homography_ransac(
    srcpoints: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    params: &RansacParams,
) -> Result<RansacResult> {
    if srcpoints.len() != dst_points.len() {
        return Err(VisionError::InvalidParameter(
            "Source and destination point arrays must have the same length".to_string(),
        ));
    }

    if srcpoints.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "At least 4 point correspondences are required".to_string(),
        ));
    }

    let n_points = srcpoints.len();
    let threshold_sq = params.threshold * params.threshold;

    // Initialize random number generator
    let mut rng: Random<StdRng> = if let Some(seed) = params.seed {
        Random::seed(seed)
    } else {
        Random::seed(scirs2_core::random::random::<u64>())
    };

    // Track the best model found so far
    let mut best_transform: Option<PerspectiveTransform> = None;
    let mut best_inliers = Vec::new();
    let mut best_score = 0;

    // Create indices for sampling
    let indices: Vec<usize> = (0..n_points).collect();

    for iteration in 0..params.max_iterations {
        // Sample 4 random correspondences
        let mut sample_indices = indices.clone();
        sample_indices.shuffle(&mut rng);
        let sample = &sample_indices[0..4];

        // Extract sample points
        let samplesrc: Vec<(f64, f64)> = sample.iter().map(|&i_| srcpoints[i_]).collect();
        let sample_dst: Vec<(f64, f64)> = sample.iter().map(|&i_| dst_points[i_]).collect();

        // Estimate homography from sample
        let transform = match PerspectiveTransform::from_points(&samplesrc, &sample_dst) {
            Ok(t) => t,
            Err(_) => continue, // Skip degenerate cases
        };

        // Count inliers
        let mut inliers = Vec::new();
        for (i_, (&src_pt, &dst_pt)) in srcpoints.iter().zip(dst_points.iter()).enumerate() {
            let transformed = transform.transform_point(src_pt);
            let error_sq = (transformed.0 - dst_pt.0).powi(2) + (transformed.1 - dst_pt.1).powi(2);

            if error_sq <= threshold_sq {
                inliers.push(i_);
            }
        }

        // Update best model if this one is better
        if inliers.len() > best_score {
            best_transform = Some(transform);
            best_inliers = inliers;
            best_score = best_inliers.len();

            // Early termination if we have enough inliers
            if best_score >= params.min_inliers {
                let inlier_ratio = best_score as f64 / n_points as f64;
                if inlier_ratio >= 0.5 {
                    // Good enough to try early termination
                    // Estimate number of iterations needed
                    let outlier_ratio = 1.0 - inlier_ratio;
                    let prob_all_outliers = outlier_ratio.powi(4);
                    if prob_all_outliers > 0.0 {
                        let needed_iterations =
                            (1.0_f64 - params.confidence).ln() / (prob_all_outliers).ln();
                        if iteration as f64 >= needed_iterations {
                            break;
                        }
                    }
                }
            }
        }
    }

    if best_score < params.min_inliers {
        return Err(VisionError::OperationError(format!(
            "RANSAC failed: only {} inliers found (minimum {})",
            best_score, params.min_inliers
        )));
    }

    let best_transform = best_transform.ok_or_else(|| {
        VisionError::OperationError("RANSAC failed to find a valid transformation".to_string())
    })?;

    // Refine the transformation using all inliers
    let inliersrc: Vec<(f64, f64)> = best_inliers.iter().map(|&i_| srcpoints[i_]).collect();
    let inlier_dst: Vec<(f64, f64)> = best_inliers.iter().map(|&i_| dst_points[i_]).collect();

    let refined_transform =
        PerspectiveTransform::from_points(&inliersrc, &inlier_dst).unwrap_or(best_transform); // Fall back to unrefined if refinement fails

    Ok(RansacResult {
        transform: refined_transform,
        inliers: best_inliers,
        iterations: params.max_iterations.min(best_score + 1),
        inlier_ratio: best_score as f64 / n_points as f64,
    })
}

/// Evaluate the quality of a homography transformation
///
/// # Arguments
///
/// * `transform` - The perspective transformation to evaluate
/// * `srcpoints` - Source points for evaluation
/// * `dst_points` - Destination points for evaluation
/// * `threshold` - Distance threshold for inlier classification
///
/// # Returns
///
/// * Tuple of (inlier_count, inlier_ratio, mean_error, median_error)
pub fn evaluate_homography_quality(
    transform: &PerspectiveTransform,
    srcpoints: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    threshold: f64,
) -> (usize, f64, f64, f64) {
    let errors = transform.reprojection_errors(srcpoints, dst_points);
    let sqrt_errors: Vec<f64> = errors.iter().map(|&e| e.sqrt()).collect();

    let inlier_count = sqrt_errors.iter().filter(|&&e| e <= threshold).count();
    let inlier_ratio = inlier_count as f64 / srcpoints.len() as f64;

    let mean_error = sqrt_errors.iter().sum::<f64>() / sqrt_errors.len() as f64;

    // Compute median error
    let mut sorted_errors = sqrt_errors.clone();
    sorted_errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_error = if sorted_errors.len().is_multiple_of(2) {
        let mid = sorted_errors.len() / 2;
        (sorted_errors[mid - 1] + sorted_errors[mid]) / 2.0
    } else {
        sorted_errors[sorted_errors.len() / 2]
    };

    (inlier_count, inlier_ratio, mean_error, median_error)
}

/// Adaptive RANSAC that adjusts the number of iterations based on inlier ratio
///
/// # Arguments
///
/// * `srcpoints` - Source points [(x, y), ...]
/// * `dst_points` - Destination points [(x', y'), ...]
/// * `base_params` - Base RANSAC parameters
///
/// # Returns
///
/// * Result containing the best homography and inlier information
///
/// # Efficiency
///
/// Dynamically adjusts iteration count based on observed inlier ratio,
/// potentially reducing computation time while maintaining robustness.
pub fn find_homography_adaptive_ransac(
    srcpoints: &[(f64, f64)],
    dst_points: &[(f64, f64)],
    base_params: &RansacParams,
) -> Result<RansacResult> {
    if srcpoints.len() < 4 {
        return Err(VisionError::InvalidParameter(
            "At least 4 point correspondences are required".to_string(),
        ));
    }

    let n_points = srcpoints.len();
    let threshold_sq = base_params.threshold * base_params.threshold;

    // Initialize random number generator
    let mut rng: Random<StdRng> = if let Some(seed) = base_params.seed {
        Random::seed(seed)
    } else {
        Random::seed(scirs2_core::random::random::<u64>())
    };

    // Track the best model found so far
    let mut best_transform: Option<PerspectiveTransform> = None;
    let mut best_inliers = Vec::new();
    let mut best_score = 0;

    // Adaptive iteration control
    let mut max_iterations = base_params.max_iterations;
    let mut current_iterations = 0;

    // Create indices for sampling
    let indices: Vec<usize> = (0..n_points).collect();

    while current_iterations < max_iterations {
        // Sample 4 random correspondences
        let mut sample_indices = indices.clone();
        sample_indices.shuffle(&mut rng);
        let sample = &sample_indices[0..4];

        // Extract sample points
        let samplesrc: Vec<(f64, f64)> = sample.iter().map(|&i_| srcpoints[i_]).collect();
        let sample_dst: Vec<(f64, f64)> = sample.iter().map(|&i_| dst_points[i_]).collect();

        // Estimate homography from sample
        let transform = match PerspectiveTransform::from_points(&samplesrc, &sample_dst) {
            Ok(t) => t,
            Err(_) => {
                current_iterations += 1;
                continue; // Skip degenerate cases
            }
        };

        // Count inliers
        let mut inliers = Vec::new();
        for (i_, (&src_pt, &dst_pt)) in srcpoints.iter().zip(dst_points.iter()).enumerate() {
            let transformed = transform.transform_point(src_pt);
            let error_sq = (transformed.0 - dst_pt.0).powi(2) + (transformed.1 - dst_pt.1).powi(2);

            if error_sq <= threshold_sq {
                inliers.push(i_);
            }
        }

        // Update best model if this one is better
        if inliers.len() > best_score {
            best_transform = Some(transform);
            best_inliers = inliers;
            best_score = best_inliers.len();

            // Adaptive iteration adjustment
            if best_score >= base_params.min_inliers {
                let inlier_ratio = best_score as f64 / n_points as f64;
                let outlier_ratio = 1.0 - inlier_ratio;

                if outlier_ratio > 0.0 && outlier_ratio < 1.0 {
                    let prob_all_outliers = outlier_ratio.powi(4);
                    if prob_all_outliers > 0.0 {
                        let needed_iterations = ((1.0_f64 - base_params.confidence).ln()
                            / prob_all_outliers.ln())
                        .ceil() as usize;
                        max_iterations = needed_iterations.min(base_params.max_iterations);
                    }
                }
            }
        }

        current_iterations += 1;
    }

    if best_score < base_params.min_inliers {
        return Err(VisionError::OperationError(format!(
            "Adaptive RANSAC failed: only {} inliers found (minimum {})",
            best_score, base_params.min_inliers
        )));
    }

    let best_transform = best_transform.ok_or_else(|| {
        VisionError::OperationError(
            "Adaptive RANSAC failed to find a valid transformation".to_string(),
        )
    })?;

    // Refine the transformation using all inliers
    let inliersrc: Vec<(f64, f64)> = best_inliers.iter().map(|&i_| srcpoints[i_]).collect();
    let inlier_dst: Vec<(f64, f64)> = best_inliers.iter().map(|&i_| dst_points[i_]).collect();

    let refined_transform =
        PerspectiveTransform::from_points(&inliersrc, &inlier_dst).unwrap_or(best_transform);

    Ok(RansacResult {
        transform: refined_transform,
        inliers: best_inliers,
        iterations: current_iterations,
        inlier_ratio: best_score as f64 / n_points as f64,
    })
}

/// Validate homography transformation using geometric constraints
///
/// # Arguments
///
/// * `transform` - The perspective transformation to validate
///
/// # Returns
///
/// * True if the transformation passes basic geometric validity checks
pub fn validate_homography_geometry(transform: &PerspectiveTransform) -> bool {
    // Check if the transformation is well-conditioned
    let det = transform.compute_determinant();
    if det.abs() < 1e-12 {
        return false; // Nearly singular
    }

    // Check for reasonable scale factors
    let h = &transform.matrix;
    let scale_x = (h[[0, 0]].powi(2) + h[[1, 0]].powi(2)).sqrt();
    let scale_y = (h[[0, 1]].powi(2) + h[[1, 1]].powi(2)).sqrt();

    // Reasonable scale bounds (between 0.1x and 10x)
    if !(0.1..=10.0).contains(&scale_x) || !(0.1..=10.0).contains(&scale_y) {
        return false;
    }

    // Check perspective terms are not too large
    let perspective_strength = (h[[2, 0]].powi(2) + h[[2, 1]].powi(2)).sqrt();
    if perspective_strength > 0.01 {
        // Strong perspective might indicate degenerate case
        return false;
    }

    true
}
