// Edge-based feature extraction for images

use crate::error::SignalResult;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

#[allow(unused_imports)]
/// Extract edge-based features from an image
#[allow(dead_code)]
pub fn extract_edge_features(
    image: &Array2<f64>,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];

    if height < 3 || width < 3 {
        // Not enough pixels for edge detection
        return Ok(());
    }

    // Detect edges using simple Sobel filters
    let mut gradient_magnitude = Array2::zeros((height - 2, width - 2));
    let mut gradient_direction = Array2::zeros((height - 2, width - 2));

    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    // Apply Sobel operators
    let mut edge_sum = 0.0;
    let mut edge_count = 0;

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            let mut gx = 0.0;
            let mut gy = 0.0;

            for ki in 0..3 {
                for kj in 0..3 {
                    let pixel = image[[i + ki - 1, j + kj - 1]];
                    gx += pixel * sobel_x[ki][kj];
                    gy += pixel * sobel_y[ki][kj];
                }
            }

            let magnitude = (gx * gx + gy * gy).sqrt();
            let direction = gy.atan2(gx);

            gradient_magnitude[[i - 1, j - 1]] = magnitude;
            gradient_direction[[i - 1, j - 1]] = direction;

            edge_sum += magnitude;
            if magnitude > (image[[i, j]] * 0.1) {
                edge_count += 1;
            }
        }
    }

    // Calculate edge statistics
    let flat_magnitude = gradient_magnitude.iter().cloned().collect::<Vec<f64>>();
    let n = flat_magnitude.len() as f64;

    // Mean gradient magnitude
    let mean_gradient = edge_sum / n;
    features.insert("edge_mean_gradient".to_string(), mean_gradient);

    // Percentage of edge pixels
    let edge_percentage = edge_count as f64 / n;
    features.insert("edge_percentage".to_string(), edge_percentage);

    // Standard deviation of gradient magnitude
    let variance = flat_magnitude
        .iter()
        .map(|&x| (x - mean_gradient).powi(2))
        .sum::<f64>()
        / n;
    let std_dev = variance.sqrt();
    features.insert("edge_std_gradient".to_string(), std_dev);

    // Histogram of gradient directions (binned into 8 directions)
    let mut direction_hist = [0; 8];
    for &dir in gradient_direction.iter() {
        // Convert from [-π, π] to [0, 8)
        let bin = (((dir + std::f64::consts::PI) / (2.0 * std::f64::consts::PI)) * 8.0).floor()
            as usize
            % 8;
        direction_hist[bin] += 1;
    }

    // Normalize direction histogram
    let direction_hist_norm: Vec<f64> = direction_hist
        .iter()
        .map(|&count| count as f64 / n)
        .collect();

    // Direction histogram features
    for (i, &count) in direction_hist_norm.iter().enumerate() {
        features.insert(format!("edge_direction_{}", i), count);
    }

    // Calculate direction entropy and uniformity
    let dir_entropy = direction_hist_norm
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    features.insert("edge_direction_entropy".to_string(), dir_entropy);

    let dir_uniformity = direction_hist_norm.iter().map(|&p| p * p).sum::<f64>();
    features.insert("edge_direction_uniformity".to_string(), dir_uniformity);

    Ok(())
}
