// Image feature extraction
//
// This module provides functions for extracting features from images,
// including color histograms, texture features, edge features, and more.
// These features can be used for image analysis, classification, and retrieval.

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::fmt::Debug;

// Import internal modules
#[allow(unused_imports)]
mod color;
mod edge;
mod haralick;
mod histogram;
mod lbp;
mod moments;
mod statistical;
mod texture;
mod types;
mod utils;

// Re-export public components
pub use color::extract_color_features;
pub use edge::extract_edge_features;
pub use haralick::extract_haralick_features;
pub use histogram::extract_histogram_features;
pub use lbp::extract_lbp_features;
pub use moments::extract_moment_features;
pub use statistical::extract_intensity_features;
pub use texture::extracttexture_features;
pub use types::ImageFeatureOptions;

// Common imports for internal use
/// Extract features from a grayscale image
///
/// # Arguments
///
/// * `image` - Input grayscale image as a 2D array
/// * `options` - Feature extraction options
///
/// # Returns
///
/// * A HashMap containing the extracted features
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::image_features::{extract_image_features, ImageFeatureOptions};
///
/// // Create a simple test image (8x8)
/// let mut image = Array2::zeros((8, 8));
/// for i in 0..8 {
///     for j in 0..8 {
///         image[[i, j]] = (i * j) as f64 % 256.0;
///     }
/// }
///
/// // Extract features with default options
/// let features = extract_image_features(&image, &ImageFeatureOptions::default()).unwrap();
///
/// // Access individual features
/// let contrast = *features.get("haralick_contrast").unwrap();
/// let mean = *features.get("intensity_mean").unwrap();
/// ```
#[allow(dead_code)]
pub fn extract_image_features<T>(
    image: &Array2<T>,
    options: &ImageFeatureOptions,
) -> SignalResult<HashMap<String, f64>>
where
    T: Clone + Copy + Into<f64> + Debug,
{
    // Validate input
    if image.is_empty() {
        return Err(SignalError::ValueError("Input image is empty".to_string()));
    }

    // Convert image to f64
    let image_f64 = image.mapv(|x| x.into());

    let mut features = HashMap::new();

    // Extract histogram features
    if options.histogram {
        extract_histogram_features(&image_f64, options, &mut features)?;
    }

    // Extract intensity features
    extract_intensity_features(&image_f64, &mut features)?;

    // Extract edge features
    if options.edges {
        extract_edge_features(&image_f64, &mut features)?;
    }

    // Extract moment-based features
    if options.moments {
        extract_moment_features(&image_f64, &mut features)?;
    }

    // Extract texture features
    if options.texture {
        extracttexture_features(&image_f64, options, &mut features)?;
    }

    // Extract Haralick features
    if options.haralick {
        extract_haralick_features(&image_f64, options, &mut features)?;
    }

    // Extract Local Binary Pattern features
    if options.lbp {
        extract_lbp_features(&image_f64, &mut features)?;
    }

    Ok(features)
}

/// Extract features from a color image (3-channel RGB)
///
/// # Arguments
///
/// * `image` - Input color image as an array with shape (height, width, 3)
/// * `options` - Feature extraction options
///
/// # Returns
///
/// * A HashMap containing the extracted features
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array3, Dim};
/// use scirs2_signal::image_features::{extract_color_image_features, ImageFeatureOptions};
///
/// // Create a simple test RGB image (8x8x3)
/// let mut image = Array3::zeros((8, 8, 3));
/// for i in 0..8 {
///     for j in 0..8 {
///         // Red channel
///         image[[i, j, 0]] = (i * j) as f64 % 256.0;
///         // Green channel
///         image[[i, j, 1]] = (i + j) as f64 % 256.0;
///         // Blue channel
///         image[[i, j, 2]] = ((i as i32 - j as i32).abs() as usize) as f64 % 256.0;
///     }
/// }
///
/// // Extract features with default options
/// let features = extract_color_image_features(&image, &ImageFeatureOptions::default()).unwrap();
///
/// // Access individual features
/// let r_mean = *features.get("r_intensity_mean").unwrap();
/// let g_mean = *features.get("g_intensity_mean").unwrap();
/// let b_mean = *features.get("b_intensity_mean").unwrap();
/// ```
#[allow(dead_code)]
pub fn extract_color_image_features<T>(
    image: &scirs2_core::ndarray::Array3<T>,
    options: &ImageFeatureOptions,
) -> SignalResult<HashMap<String, f64>>
where
    T: Clone + Copy + Into<f64> + Debug,
{
    // Validate input
    if image.is_empty() {
        return Err(SignalError::ValueError("Input image is empty".to_string()));
    }

    // Check that image has 3 channels
    let shape = image.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(SignalError::ValueError(
            "Input image should have 3 channels (RGB) with shape (height, width, 3)".to_string(),
        ));
    }

    let mut features = HashMap::new();

    // Extract RGB channels
    let height = shape[0];
    let width = shape[1];

    // Create views for each channel
    let r_channel = Array2::from_shape_fn((height, width), |(i, j)| image[[i, j, 0]].into());
    let g_channel = Array2::from_shape_fn((height, width), |(i, j)| image[[i, j, 1]].into());
    let b_channel = Array2::from_shape_fn((height, width), |(i, j)| image[[i, j, 2]].into());

    // Convert to grayscale for some features
    let gray_image = Array2::from_shape_fn((height, width), |(i, j)| {
        // Standard RGB to grayscale conversion
        0.299 * image[[i, j, 0]].into()
            + 0.587 * image[[i, j, 1]].into()
            + 0.114 * image[[i, j, 2]].into()
    });

    // Extract features for each channel
    let r_features = extract_image_features(&r_channel, options)?;
    let g_features = extract_image_features(&g_channel, options)?;
    let b_features = extract_image_features(&b_channel, options)?;

    // Extract grayscale features
    let gray_features = extract_image_features(&gray_image, options)?;

    // Add channel prefixes to feature names
    for (key, value) in r_features {
        features.insert(format!("r_{}", key), value);
    }

    for (key, value) in g_features {
        features.insert(format!("g_{}", key), value);
    }

    for (key, value) in b_features {
        features.insert(format!("b_{}", key), value);
    }

    for (key, value) in gray_features {
        if !key.starts_with("histogram_bin") {
            features.insert(format!("gray_{}", key), value);
        }
    }

    // Extract color-specific features
    extract_color_features(image, &mut features)?;

    Ok(features)
}
