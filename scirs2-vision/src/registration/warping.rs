//! Image warping and resampling functions
//!
//! This module provides functionality for transforming images using various
//! interpolation methods and geometric transformations.
//!
//! **Note**: This module has been refactored into smaller, focused sub-modules
//! for better maintainability. The original API is preserved for backward compatibility.

use crate::error::{Result, VisionError};
use crate::registration::{identity_transform, transform_point, Point2D, TransformMatrix};
use image::{DynamicImage, GenericImageView, GrayImage, Luma, Rgb, RgbImage};
use scirs2_core::ndarray::{Array1, Array2, Array3};
use std::time::{Duration, Instant};

// Import the modular implementation
#[path = "warping_modules/core_warping.rs"]
pub mod core_warping;

#[path = "warping_modules/stereo_rectification.rs"]
pub mod stereo_rectification;

#[path = "warping_modules/image_stitching.rs"]
pub mod image_stitching;

#[path = "warping_modules/depth_mapping.rs"]
pub mod depth_mapping;

// Re-export types from core_warping for backward compatibility
pub use core_warping::{BoundaryMethod, InterpolationMethod};

/// Warp a grayscale image using a transformation matrix
pub fn warp_image(
    image: &GrayImage,
    transform: &TransformMatrix,
    output_size: (u32, u32),
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
) -> Result<GrayImage> {
    core_warping::warp_image(image, transform, output_size, interpolation, boundary)
}

/// GPU-accelerated image warping
pub fn warp_image_gpu(
    image: &GrayImage,
    transform: &TransformMatrix,
    output_size: (u32, u32),
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
) -> Result<GrayImage> {
    core_warping::warp_image_gpu(image, transform, output_size, interpolation, boundary)
}

/// Warp an RGB image using a transformation matrix
pub fn warp_rgb_image(
    image: &RgbImage,
    transform: &TransformMatrix,
    output_size: (u32, u32),
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
) -> Result<RgbImage> {
    core_warping::warp_rgb_image(image, transform, output_size, interpolation, boundary)
}

/// Rectify a stereo pair to align epipolar lines horizontally
pub fn rectify_stereo_pair(
    left_image: &DynamicImage,
    right_image: &DynamicImage,
    fundamental_matrix: &TransformMatrix,
) -> Result<(DynamicImage, DynamicImage)> {
    stereo_rectification::rectify_stereo_pair(left_image, right_image, fundamental_matrix)
}

/// Create a panorama by stitching multiple images
pub fn stitch_images(
    images: &[DynamicImage],
    transforms: &[TransformMatrix],
    output_size: (u32, u32),
) -> Result<DynamicImage> {
    image_stitching::stitch_images(images, transforms, output_size)
}

/// Memory-efficient panorama stitching using streaming tile-based processing
pub fn stitch_images_streaming(
    images: &[DynamicImage],
    transforms: &[TransformMatrix],
    output_size: (u32, u32),
) -> Result<DynamicImage> {
    image_stitching::stitch_images_streaming(images, transforms, output_size)
}

/// Create a mesh grid for perspective correction
pub fn create_mesh_grid(width: u32, height: u32) -> (Array2<f64>, Array2<f64>) {
    let mut x_grid = Array2::zeros((height as usize, width as usize));
    let mut y_grid = Array2::zeros((height as usize, width as usize));

    for y in 0..height as usize {
        for x in 0..width as usize {
            x_grid[[y, x]] = x as f64;
            y_grid[[y, x]] = y as f64;
        }
    }

    (x_grid, y_grid)
}

/// Perspective correct coordinates
pub fn perspective_correct(
    x_grid: &Array2<f64>,
    y_grid: &Array2<f64>,
    transform: &TransformMatrix,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let (height, width) = x_grid.dim();
    let mut corrected_x = Array2::zeros((height, width));
    let mut corrected_y = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let src_x = x_grid[[y, x]];
            let src_y = y_grid[[y, x]];

            // Apply homogeneous transformation
            let dst_x = transform[[0, 0]] * src_x + transform[[0, 1]] * src_y + transform[[0, 2]];
            let dst_y = transform[[1, 0]] * src_x + transform[[1, 1]] * src_y + transform[[1, 2]];
            let w = transform[[2, 0]] * src_x + transform[[2, 1]] * src_y + transform[[2, 2]];

            if w.abs() > 1e-10 {
                corrected_x[[y, x]] = dst_x / w;
                corrected_y[[y, x]] = dst_y / w;
            } else {
                return Err(VisionError::OperationError(
                    "Perspective transformation resulted in invalid coordinates".to_string(),
                ));
            }
        }
    }

    Ok((corrected_x, corrected_y))
}

// Re-export types for backward compatibility
pub use depth_mapping::{
    compute_depth_map, disparity_to_depth, DepthMapResult, DepthMapStats, MatchingCostFunction,
    ProcessingTimes, SgmParams, StereoMatchingParams,
};
pub use image_stitching::{BlendingMode, StreamingPanoramaProcessor, TileConfig};

/// Multiply two 3x3 matrices
pub fn matrix_multiply(a: &TransformMatrix, b: &TransformMatrix) -> Result<TransformMatrix> {
    stereo_rectification::matrix_multiply(a, b)
}

/// Simple 3x3 matrix inversion for TransformMatrix
fn invert_3x3_matrix(matrix: &TransformMatrix) -> Result<TransformMatrix> {
    core_warping::invert_3x3_matrix(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_interpolation_method() {
        let method = InterpolationMethod::Bilinear;
        assert_eq!(method, InterpolationMethod::Bilinear);
    }

    #[test]
    fn test_boundary_method() {
        let boundary = BoundaryMethod::Zero;
        match boundary {
            BoundaryMethod::Zero => {} // Expected case
            _ => unreachable!(),
        }
    }

    #[test]
    fn test_create_mesh_grid() {
        let (x_grid, y_grid) = create_mesh_grid(3, 2);
        assert_eq!(x_grid.dim(), (2, 3));
        assert_eq!(y_grid.dim(), (2, 3));
        assert_eq!(x_grid[[0, 0]], 0.0);
        assert_eq!(x_grid[[0, 2]], 2.0);
        assert_eq!(y_grid[[0, 0]], 0.0);
        assert_eq!(y_grid[[1, 0]], 1.0);
    }

    #[test]
    fn test_matrix_multiply() {
        let a: Array2<f64> = Array2::eye(3);
        let b: Array2<f64> = Array2::eye(3);
        let result = matrix_multiply(&a, &b).unwrap();
        let expected: Array2<f64> = Array2::eye(3);
        assert_eq!(result, expected);
    }
}
