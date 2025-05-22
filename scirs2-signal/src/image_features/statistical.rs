//! Statistical image feature extraction functions

use super::utils::{calculate_kurtosis, calculate_skewness};
use crate::error::SignalResult;
use ndarray::Array2;
use std::collections::HashMap;

/// Extract basic intensity features from an image
pub fn extract_intensity_features(
    image: &Array2<f64>,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    // Get flattened image data
    let flat_image = image.iter().cloned().collect::<Vec<f64>>();
    let n = flat_image.len() as f64;

    // Calculate basic statistics
    let sum: f64 = flat_image.iter().sum();
    let mean = sum / n;
    features.insert("intensity_mean".to_string(), mean);

    // Calculate variance and std
    let sum_squared_diff: f64 = flat_image.iter().map(|&x| (x - mean).powi(2)).sum();
    let variance = sum_squared_diff / n;
    let std_dev = variance.sqrt();
    features.insert("intensity_variance".to_string(), variance);
    features.insert("intensity_std".to_string(), std_dev);

    // Calculate median
    let mut sorted = flat_image.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };
    features.insert("intensity_median".to_string(), median);

    // Calculate min, max, range
    let min = *flat_image
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    let max = *flat_image
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    features.insert("intensity_min".to_string(), min);
    features.insert("intensity_max".to_string(), max);
    features.insert("intensity_range".to_string(), max - min);

    // Calculate skewness
    let skewness = calculate_skewness(&flat_image, mean, std_dev);
    features.insert("intensity_skewness".to_string(), skewness);

    // Calculate kurtosis
    let kurtosis = calculate_kurtosis(&flat_image, mean, std_dev);
    features.insert("intensity_kurtosis".to_string(), kurtosis);

    // Calculate energy and root mean square
    let energy: f64 = flat_image.iter().map(|&x| x * x).sum();
    let rms = (energy / n).sqrt();
    features.insert("intensity_energy".to_string(), energy);
    features.insert("intensity_rms".to_string(), rms);

    // Calculate coefficient of variation
    if mean.abs() > 1e-10 {
        features.insert("intensity_cv".to_string(), std_dev / mean.abs());
    } else {
        features.insert("intensity_cv".to_string(), f64::MAX);
    }

    Ok(())
}
