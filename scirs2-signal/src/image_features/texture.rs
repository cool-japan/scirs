//! Texture feature extraction for images

use super::types::ImageFeatureOptions;
use crate::error::SignalResult;
use ndarray::Array2;
use std::collections::HashMap;

/// Extract texture features from an image
pub fn extract_texture_features(
    image: &Array2<f64>,
    _options: &ImageFeatureOptions, // Unused, but kept for consistency with other extraction functions
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    // Calculate Global features based on gradient statistics

    // Calculate gradient
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];

    if height < 3 || width < 3 {
        // Not enough pixels for gradient calculation
        return Ok(());
    }

    let mut gradient_x = Array2::zeros((height - 2, width - 2));
    let mut gradient_y = Array2::zeros((height - 2, width - 2));

    // Simple 3x3 Sobel filters
    for i in 1..height - 1 {
        for j in 1..width - 1 {
            // Horizontal gradient
            gradient_x[[i - 1, j - 1]] =
                (image[[i - 1, j + 1]] + 2.0 * image[[i, j + 1]] + image[[i + 1, j + 1]]
                    - image[[i - 1, j - 1]]
                    - 2.0 * image[[i, j - 1]]
                    - image[[i + 1, j - 1]])
                    / 8.0;

            // Vertical gradient
            gradient_y[[i - 1, j - 1]] =
                (image[[i + 1, j - 1]] + 2.0 * image[[i + 1, j]] + image[[i + 1, j + 1]]
                    - image[[i - 1, j - 1]]
                    - 2.0 * image[[i - 1, j]]
                    - image[[i - 1, j + 1]])
                    / 8.0;
        }
    }

    // Calculate gradient magnitude and direction
    let mut gradient_mag = Array2::zeros((height - 2, width - 2));
    for i in 0..height - 2 {
        for j in 0..width - 2 {
            let gx = gradient_x[[i, j]];
            let gy = gradient_y[[i, j]];
            gradient_mag[[i, j]] = (gx * gx + gy * gy).sqrt();
        }
    }

    // Calculate texture features based on gradient magnitude
    let flat_gradient = gradient_mag.iter().cloned().collect::<Vec<f64>>();

    // Calculate statistics of gradient magnitude
    let n = flat_gradient.len() as f64;
    let mean = flat_gradient.iter().sum::<f64>() / n;
    let variance = flat_gradient
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();

    // Store texture features
    features.insert("texture_mean_gradient".to_string(), mean);
    features.insert("texture_std_gradient".to_string(), std_dev);

    // Calculate texture coarseness
    let coarseness = 1.0 / mean;
    features.insert("texture_coarseness".to_string(), coarseness);

    // Calculate texture contrast
    let min_val = *flat_gradient
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let max_val = *flat_gradient
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let contrast = (max_val - min_val) / (max_val + min_val + 1e-10);
    features.insert("texture_contrast".to_string(), contrast);

    // Calculate texture energy
    let energy = flat_gradient.iter().map(|&x| x * x).sum::<f64>() / n;
    features.insert("texture_energy".to_string(), energy);

    // Calculate texture directionality
    let directionality = variance / (mean * mean + 1e-10);
    features.insert("texture_directionality".to_string(), directionality);

    Ok(())
}
