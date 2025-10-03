//! Feature detection and extraction module
//!
//! This module provides functionality for detecting and extracting features
//! from images.

pub mod advanced_enhancement;
pub mod advanced_tracking;
pub mod brief;
pub mod canny;
pub mod descriptor;
pub mod dog;
pub mod fast;
pub mod gabor;
pub mod glcm;
pub mod hog;
pub mod homography;
pub mod hough_circle;
pub mod hough_line;
pub mod integral_image;
pub mod laplacian;
pub mod lbp;
pub mod log_blob;
pub mod matching;
pub mod mser;
pub mod neural_features;
pub mod nms;
pub mod optical_flow;
pub mod orb;
pub mod prewitt;
pub mod ransac;
pub mod shi_tomasi;
pub mod sobel;
pub mod tamura;
pub mod template_matching;
pub mod tracker;
pub mod vision_transformer;

pub use advanced_enhancement::{
    AdvancedDenoiser, DenoisingMethod, HDRProcessor, SuperResolutionMethod,
    SuperResolutionProcessor, ToneMappingMethod,
};
pub use advanced_tracking::{
    AppearanceExtractor, BoundingBox as TrackingBoundingBox, DeepSORT, Detection, KalmanFilter,
    Track, TrackState, TrackingMetrics,
};
pub use brief::*;
pub use canny::*;
pub use descriptor::*;
pub use dog::*;
pub use fast::*;
pub use gabor::*;
pub use glcm::*;
pub use hog::*;
pub use homography::*;
pub use hough_circle::*;
pub use hough_line::*;
pub use integral_image::*;
pub use laplacian::*;
pub use lbp::*;
pub use log_blob::*;
pub use matching::*;
pub use mser::*;
pub use neural_features::*;
pub use nms::*;
pub use optical_flow::*;
pub use orb::*;
pub use prewitt::*;
pub use ransac::*;
pub use shi_tomasi::*;
pub use sobel::*;
pub use tamura::*;
pub use template_matching::*;
pub use tracker::*;
pub use vision_transformer::{
    MatcherConfig, SwinConfig, SwinTransformer, TransformerFeatureMatcher, ViTConfig,
    VisionTransformer,
};

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage};
use scirs2_core::ndarray::Array2;

/// Convert an image to a 2D grayscale array
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// * Result containing a 2D array of pixel intensities (grayscale)
#[allow(dead_code)]
pub fn image_to_array(img: &DynamicImage) -> Result<Array2<f32>> {
    // Convert to grayscale
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Create 2D array
    let mut array = Array2::zeros((height as usize, width as usize));

    // Fill array with grayscale values
    for y in 0..height {
        for x in 0..width {
            let pixel = gray.get_pixel(x, y)[0] as f32 / 255.0;
            array[[y as usize, x as usize]] = pixel;
        }
    }

    Ok(array)
}

/// Convert a 2D array to an image
///
/// # Arguments
///
/// * `array` - Input array
///
/// # Returns
///
/// * Result containing a grayscale image
#[allow(dead_code)]
pub fn array_to_image(array: &Array2<f32>) -> Result<GrayImage> {
    let height = array.shape()[0];
    let width = array.shape()[1];

    let mut img = GrayImage::new(width as u32, height as u32);

    for y in 0..height {
        for x in 0..width {
            let value = (array[[y, x]] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Luma([value]));
        }
    }

    Ok(img)
}

/// Extract features using a Sobel edge detector
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Threshold value for edge detection
///
/// # Returns
///
/// * Result containing an edge image
#[allow(dead_code)]
pub fn sobel_edges(img: &DynamicImage, threshold: f32) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Create output array
    let mut edges = Array2::zeros(array.dim());

    // Apply Sobel operator
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Horizontal Sobel kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            let gx = -array[[y - 1, x - 1]]
                + 1.0 * array[[y - 1, x + 1]]
                + -2.0 * array[[y, x - 1]]
                + 2.0 * array[[y, x + 1]]
                + -array[[y + 1, x - 1]]
                + 1.0 * array[[y + 1, x + 1]];

            // Vertical Sobel kernel: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            let gy = -array[[y - 1, x - 1]]
                + -2.0 * array[[y - 1, x]]
                + -array[[y - 1, x + 1]]
                + 1.0 * array[[y + 1, x - 1]]
                + 2.0 * array[[y + 1, x]]
                + 1.0 * array[[y + 1, x + 1]];

            // Calculate magnitude
            let magnitude = (gx * gx + gy * gy).sqrt();

            // Apply threshold
            if magnitude > threshold {
                edges[[y, x]] = 1.0;
            }
        }
    }

    // Convert to image
    array_to_image(&edges)
}

/// Detect corners using a Harris corner detector
///
/// The Harris corner detector is a corner detection operator that is commonly
/// used in computer vision algorithms to extract corners and infer features
/// of an image.
///
/// # Arguments
///
/// * `img` - Input image
/// * `block_size` - Size of the window for corner detection
/// * `k` - Harris detector free parameter (typically 0.04-0.06)
/// * `threshold` - Threshold for corner detection
///
/// # Returns
///
/// * Result containing an image with corners marked
///
/// # References
///
/// - Harris, C. and Stephens, M., 1988, August. A combined corner and edge detector. In Alvey vision conference (Vol. 15, No. 50, pp. 10-5244).
#[allow(dead_code)]
pub fn harris_corners(
    img: &DynamicImage,
    block_size: usize,
    k: f32,
    threshold: f32,
) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Check if block_size is valid
    if block_size.is_multiple_of(2) || block_size < 3 {
        return Err(VisionError::InvalidParameter(
            "block_size must be odd and at least 3".to_string(),
        ));
    }

    // Create arrays for gradients
    let mut ix2 = Array2::zeros((height, width));
    let mut iy2 = Array2::zeros((height, width));
    let mut ixy = Array2::zeros((height, width));

    // Step 1: Calculate gradients using simple finite differences
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let dx = (array[[y, x + 1]] - array[[y, x - 1]]) / 2.0;
            let dy = (array[[y + 1, x]] - array[[y - 1, x]]) / 2.0;

            ix2[[y, x]] = dx * dx;
            iy2[[y, x]] = dy * dy;
            ixy[[y, x]] = dx * dy;
        }
    }

    // Step 2: Apply box filter (simplified Gaussian)
    let radius = block_size / 2;
    let mut smoothed_ix2 = Array2::zeros((height, width));
    let mut smoothed_iy2 = Array2::zeros((height, width));
    let mut smoothed_ixy = Array2::zeros((height, width));

    for y in radius..(height - radius) {
        for x in radius..(width - radius) {
            let mut sum_ix2 = 0.0;
            let mut sum_iy2 = 0.0;
            let mut sum_ixy = 0.0;
            let mut count = 0;

            for dy in (y - radius)..=(y + radius) {
                for dx in (x - radius)..=(x + radius) {
                    sum_ix2 += ix2[[dy, dx]];
                    sum_iy2 += iy2[[dy, dx]];
                    sum_ixy += ixy[[dy, dx]];
                    count += 1;
                }
            }

            smoothed_ix2[[y, x]] = sum_ix2 / count as f32;
            smoothed_iy2[[y, x]] = sum_iy2 / count as f32;
            smoothed_ixy[[y, x]] = sum_ixy / count as f32;
        }
    }

    // Step 3: Calculate Harris response
    let mut corners = Array2::zeros((height, width));

    for y in radius..(height - radius) {
        for x in radius..(width - radius) {
            let det = smoothed_ix2[[y, x]] * smoothed_iy2[[y, x]]
                - smoothed_ixy[[y, x]] * smoothed_ixy[[y, x]];
            let trace = smoothed_ix2[[y, x]] + smoothed_iy2[[y, x]];
            let r = det - k * trace * trace;

            if r > threshold {
                // Non-maximum suppression
                let mut is_local_max = true;

                'window: for dy in (y - radius)..=(y + radius) {
                    for dx in (x - radius)..=(x + radius) {
                        if dy == y && dx == x {
                            continue;
                        }

                        if corners[[dy, dx]] >= r {
                            is_local_max = false;
                            break 'window;
                        }
                    }
                }

                if is_local_max {
                    corners[[y, x]] = r;
                }
            }
        }
    }

    // Normalize and threshold
    let max_response = corners.iter().fold(0.0f32, |a, &b| a.max(b));
    if max_response > 0.0 {
        corners.mapv_inplace(|x| if x > 0.0 { 1.0 } else { 0.0 });
    }

    // Convert to image
    array_to_image(&corners)
}

/// Extract feature coordinates
///
/// # Arguments
///
/// * `img` - Image with detected features
///
/// # Returns
///
/// * Vector of (x, y) coordinates of features
#[allow(dead_code)]
pub fn extract_feature_coordinates(img: &GrayImage) -> Vec<(u32, u32)> {
    let mut coords = Vec::new();
    let (width, height) = img.dimensions();

    for y in 0..height {
        for x in 0..width {
            if img.get_pixel(x, y)[0] > 0 {
                coords.push((x, y));
            }
        }
    }

    coords
}
