//! Stereo rectification and epipolar geometry functions
//!
//! This module provides functionality for rectifying stereo image pairs to align
//! epipolar lines horizontally, which simplifies stereo matching and depth computation.

use super::core_warping::{warp_image, BoundaryMethod, InterpolationMethod};
use crate::error::{Result, VisionError};
use crate::registration::{identity_transform, transform_point, Point2D, TransformMatrix};
use image::{DynamicImage, GenericImageView};
use scirs2_core::ndarray::Array2;

/// Rectify a stereo pair to align epipolar lines horizontally
///
/// # Arguments
///
/// * `left_image` - Left stereo image
/// * `right_image` - Right stereo image
/// * `fundamental_matrix` - Fundamental matrix relating the two images
///
/// # Returns
///
/// * Result containing the rectified left and right images
pub fn rectify_stereo_pair(
    left_image: &DynamicImage,
    right_image: &DynamicImage,
    fundamental_matrix: &TransformMatrix,
) -> Result<(DynamicImage, DynamicImage)> {
    // Ensure both images have the same dimensions
    let (left_width, left_height) = left_image.dimensions();
    let (right_width, right_height) = right_image.dimensions();

    if left_width != right_width || left_height != right_height {
        return Err(VisionError::InvalidParameter(
            "Stereo images must have the same dimensions".to_string(),
        ));
    }

    // Compute epipoles from fundamental matrix
    let (left_epipole, right_epipole) = compute_epipoles(fundamental_matrix)?;

    // Compute rectification transforms
    let (left_transform, right_transform) = compute_rectification_transforms(
        left_epipole,
        right_epipole,
        (left_width, left_height),
        fundamental_matrix,
    )?;

    // Apply rectification transforms
    let left_rectified = warp_image(
        &left_image.to_luma8(),
        &left_transform,
        (left_width, left_height),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    let right_rectified = warp_image(
        &right_image.to_luma8(),
        &right_transform,
        (right_width, right_height),
        InterpolationMethod::Bilinear,
        BoundaryMethod::Zero,
    )?;

    Ok((
        DynamicImage::ImageLuma8(left_rectified),
        DynamicImage::ImageLuma8(right_rectified),
    ))
}

/// Compute epipoles from fundamental matrix
///
/// The epipoles are the null spaces of F and F^T respectively.
/// For left epipole: F * e_left = 0
/// For right epipole: F^T * e_right = 0
#[allow(dead_code)]
fn compute_epipoles(fundamental_matrix: &TransformMatrix) -> Result<(Point2D, Point2D)> {
    // Find left epipole (null space of F^T)
    let left_epipole = find_null_space(&transpose_matrix(fundamental_matrix))?;

    // Find right epipole (null space of F)
    let right_epipole = find_null_space(fundamental_matrix)?;

    Ok((left_epipole, right_epipole))
}

/// Transpose a 3x3 matrix
#[allow(dead_code)]
fn transpose_matrix(matrix: &TransformMatrix) -> TransformMatrix {
    let mut transposed = Array2::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            transposed[[i, j]] = matrix[[j, i]];
        }
    }
    transposed
}

/// Find the null space of a 3x3 matrix (the eigenvector corresponding to the smallest eigenvalue)
#[allow(dead_code)]
fn find_null_space(matrix: &TransformMatrix) -> Result<Point2D> {
    // Use power iteration to find the smallest eigenvalue and corresponding eigenvector
    // We solve (A^T * A) * v = lambda * v where lambda is the smallest eigenvalue

    let mut ata: Array2<f64> = Array2::zeros((3, 3));

    // Compute A^T * A
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                ata[[i, j]] += matrix[[k, i]] * matrix[[k, j]];
            }
        }
    }

    // Use inverse power iteration to find the smallest eigenvalue
    let mut v = vec![1.0, 1.0, 1.0]; // Initial guess

    for _ in 0..50 {
        // Iteration limit
        // Solve (A^T * A) * v_new = v_old using Gauss-Seidel iteration
        let mut v_new = vec![0.0; 3];

        for _ in 0..10 {
            // Inner iterations for solving linear system
            for i in 0..3 {
                let mut sum = 0.0;
                for j in 0..3 {
                    if i != j {
                        sum += ata[[i, j]] * v_new[j];
                    }
                }

                if ata[[i, i]].abs() > 1e-10 {
                    v_new[i] = (v[i] - sum) / ata[[i, i]];
                } else {
                    v_new[i] = v[i]; // Avoid division by zero
                }
            }
        }

        // Normalize
        let norm = (v_new[0] * v_new[0] + v_new[1] * v_new[1] + v_new[2] * v_new[2]).sqrt();
        if norm > 1e-10 {
            for v_new_item in v_new.iter_mut().take(3) {
                *v_new_item /= norm;
            }
        }

        v = v_new;
    }

    // Convert homogeneous coordinates to 2D point
    if v[2].abs() > 1e-10_f64 {
        Ok(Point2D::new(v[0] / v[2], v[1] / v[2]))
    } else {
        // Point at infinity - use large coordinates
        Ok(Point2D::new(v[0] * 1e6, v[1] * 1e6))
    }
}

/// Compute rectification transforms using Hartley's method
#[allow(dead_code)]
fn compute_rectification_transforms(
    left_epipole: Point2D,
    right_epipole: Point2D,
    image_size: (u32, u32),
    fundamental_matrix: &TransformMatrix,
) -> Result<(TransformMatrix, TransformMatrix)> {
    let (width, height) = image_size;
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;

    // Compute left rectification transform
    let left_transform =
        compute_single_rectification_transform(left_epipole, (center_x, center_y), image_size)?;

    // For the right transform, we need to ensure epipolar lines are horizontal
    // and corresponding to the left transform
    let right_transform = compute_right_rectification_transform(
        right_epipole,
        (center_x, center_y),
        image_size,
        &left_transform,
        fundamental_matrix,
    )?;

    Ok((left_transform, right_transform))
}

/// Compute rectification transform for a single image
#[allow(dead_code)]
fn compute_single_rectification_transform(
    epipole: Point2D,
    center: (f64, f64),
    image_size: (u32, u32),
) -> Result<TransformMatrix> {
    let _width_height = image_size;
    let (center_x, center_y) = center;

    // If epipole is at infinity (parallel cameras), use identity transform
    if epipole.x.abs() > 1e5 || epipole.y.abs() > 1e5 {
        return Ok(identity_transform());
    }

    // Translate epipole to origin
    let mut t1 = identity_transform();
    t1[[0, 2]] = -center_x;
    t1[[1, 2]] = -center_y;

    // Rotate so that epipole is on positive x-axis
    let ex = epipole.x - center_x;
    let ey = epipole.y - center_y;
    let e_dist = (ex * ex + ey * ey).sqrt();

    let mut rotation = identity_transform();
    if e_dist > 1e-10 {
        let cos_theta = ex / e_dist;
        let sin_theta = ey / e_dist;

        rotation[[0, 0]] = cos_theta;
        rotation[[0, 1]] = sin_theta;
        rotation[[1, 0]] = -sin_theta;
        rotation[[1, 1]] = cos_theta;
    }

    // Apply shearing to make epipolar lines horizontal
    let mut shear = identity_transform();

    // Use a simple shearing that maps the epipole to infinity
    let shear_factor = if e_dist > 1e-10 { -ey / ex } else { 0.0 };
    shear[[0, 1]] = shear_factor;

    // Translate back to center
    let mut t2 = identity_transform();
    t2[[0, 2]] = center_x;
    t2[[1, 2]] = center_y;

    // Combine transforms: T2 * Shear * Rotation * T1
    let temp1 = matrix_multiply(&rotation, &t1)?;
    let temp2 = matrix_multiply(&shear, &temp1)?;
    let final_transform = matrix_multiply(&t2, &temp2)?;

    Ok(final_transform)
}

/// Compute right rectification transform that aligns with the left transform
#[allow(dead_code)]
fn compute_right_rectification_transform(
    right_epipole: Point2D,
    center: (f64, f64),
    image_size: (u32, u32),
    left_transform: &TransformMatrix,
    fundamental_matrix: &TransformMatrix,
) -> Result<TransformMatrix> {
    // Start with single-image rectification for right image
    let mut right_transform =
        compute_single_rectification_transform(right_epipole, center, image_size)?;

    // Adjust the right transform to ensure epipolar lines match with left image
    // This involves computing a corrective transform based on the fundamental matrix

    // For simplicity, we use the same approach as left image but with different parameters
    // In a full implementation, this would involve more sophisticated epipolar geometry

    // Apply a vertical adjustment to align epipolar lines
    let vertical_adjustment = compute_vertical_alignment(
        left_transform,
        &right_transform,
        fundamental_matrix,
        image_size,
    )?;

    right_transform[[1, 2]] += vertical_adjustment;

    Ok(right_transform)
}

/// Compute vertical adjustment to align epipolar lines between left and right images
#[allow(dead_code)]
fn compute_vertical_alignment(
    left_transform: &TransformMatrix,
    _transform: &TransformMatrix,
    fundamental_matrix: &TransformMatrix,
    image_size: (u32, u32),
) -> Result<f64> {
    let (width, height) = image_size;

    // Sample points from the left image and compute their epipolar lines in the right image
    let test_points = vec![
        Point2D::new(width as f64 * 0.25, height as f64 * 0.25),
        Point2D::new(width as f64 * 0.75, height as f64 * 0.25),
        Point2D::new(width as f64 * 0.25, height as f64 * 0.75),
        Point2D::new(width as f64 * 0.75, height as f64 * 0.75),
    ];

    let mut total_adjustment = 0.0;
    let mut count = 0;

    for point in test_points {
        // Transform point through left rectification
        let left_rectified = transform_point(point, left_transform);

        // Compute corresponding epipolar line in right image using fundamental matrix
        let epipolar_line = compute_epipolar_line(left_rectified, fundamental_matrix);

        // The y-coordinate of this line should be the same as the rectified left point
        // Compute the adjustment needed
        let expected_y = left_rectified.y;
        let actual_y = compute_epipolar_line_y_intercept(&epipolar_line, left_rectified.x);

        total_adjustment += expected_y - actual_y;
        count += 1;
    }

    if count > 0 {
        Ok(total_adjustment / count as f64)
    } else {
        Ok(0.0)
    }
}

/// Compute epipolar line in the right image corresponding to a point in the left image
#[allow(dead_code)]
fn compute_epipolar_line(point: Point2D, fundamental_matrix: &TransformMatrix) -> (f64, f64, f64) {
    // Epipolar line l = F * p where p is in homogeneous coordinates
    let p = [point.x, point.y, 1.0];
    let mut line = [0.0; 3];

    for i in 0..3 {
        for j in 0..3 {
            line[i] += fundamental_matrix[[i, j]] * p[j];
        }
    }

    (line[0], line[1], line[2])
}

/// Compute y-intercept of an epipolar line at a given x coordinate
#[allow(dead_code)]
fn compute_epipolar_line_y_intercept(line: &(f64, f64, f64), x: f64) -> f64 {
    let (a, b, c) = *line;

    if b.abs() > 1e-10 {
        -(a * x + c) / b
    } else {
        0.0 // Vertical line, return y=0
    }
}

/// Multiply two 3x3 matrices
#[allow(dead_code)]
pub fn matrix_multiply(a: &TransformMatrix, b: &TransformMatrix) -> Result<TransformMatrix> {
    let mut result = Array2::zeros((3, 3));

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[[i, j]] += a[[i, k]] * b[[k, j]];
            }
        }
    }

    Ok(result)
}
