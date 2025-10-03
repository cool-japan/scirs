//! Image warping and transformation module
//!
//! This module has been refactored into focused sub-modules for better organization:
//! - `core_warping`: Basic image warping and interpolation
//! - `stereo_rectification`: Stereo pair rectification and epipolar geometry
//! - `image_stitching`: Panorama creation and image stitching
//! - `depth_mapping`: Stereo depth mapping and disparity computation

pub mod core_warping;
pub mod stereo_rectification;
pub mod image_stitching;
pub mod depth_mapping;

// Re-export main public API
pub use core_warping::{
    warp_image, warp_image_gpu, warp_image_cpu, warp_rgb_image,
    sample_image, InterpolationMethod, BoundaryMethod,
};

pub use stereo_rectification::{
    rectify_stereo_pair, matrix_multiply,
};

pub use image_stitching::{
    stitch_images, stitch_images_streaming,
    TileConfig, BlendingMode, StreamingPanoramaProcessor,
};

pub use depth_mapping::{
    compute_depth_map, disparity_to_depth,
    StereoMatchingParams, MatchingCostFunction, SgmParams,
    DepthMapResult, DepthMapStats, ProcessingTimes,
};

/// Create a mesh grid for perspective correction
///
/// # Arguments
///
/// * `width` - Grid width
/// * `height` - Grid height
///
/// # Returns
///
/// * Tuple of (X, Y) coordinate grids
pub fn create_mesh_grid(width: u32, height: u32) -> (scirs2_core::ndarray::Array2<f64>, scirs2_core::ndarray::Array2<f64>) {
    use scirs2_core::ndarray::Array2;

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
///
/// # Arguments
///
/// * `x_grid` - X coordinate grid
/// * `y_grid` - Y coordinate grid
/// * `transform` - Transformation matrix
///
/// # Returns
///
/// * Result containing corrected coordinate grids
pub fn perspective_correct(
    x_grid: &scirs2_core::ndarray::Array2<f64>,
    y_grid: &scirs2_core::ndarray::Array2<f64>,
    transform: &crate::registration::TransformMatrix,
) -> crate::error::Result<(scirs2_core::ndarray::Array2<f64>, scirs2_core::ndarray::Array2<f64>)> {
    use crate::error::{Result, VisionError};
    use scirs2_core::ndarray::Array2;

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
                    "Perspective transformation resulted in invalid coordinates".to_string()
                ));
            }
        }
    }

    Ok((corrected_x, corrected_y))
}