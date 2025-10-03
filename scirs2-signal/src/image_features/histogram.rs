// Histogram-based feature extraction for images

use super::types::ImageFeatureOptions;
use crate::error::SignalResult;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

#[allow(unused_imports)]
/// Extract histogram features from an image
#[allow(dead_code)]
pub fn extract_histogram_features(
    image: &Array2<f64>,
    options: &ImageFeatureOptions,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    // Get flattened image data for histogram calculation
    let flat_image = image.iter().cloned().collect::<Vec<f64>>();

    // Find min and max values
    let min_val = flat_image.iter().fold(f64::MAX, |a, &b| a.min(b));
    let max_val = flat_image.iter().fold(f64::MIN, |a, &b| a.max(b));

    // Compute histogram
    let bin_count = options.histogram_bins;
    let mut histogram = vec![0; bin_count];

    // Edge case: if min == max, all values go in the same bin
    if (max_val - min_val).abs() < 1e-10 {
        histogram[0] = flat_image.len();
    } else {
        let bin_width = (max_val - min_val) / bin_count as f64;

        for &val in &flat_image {
            let bin = ((val - min_val) / bin_width)
                .floor()
                .min((bin_count - 1) as f64) as usize;
            histogram[bin] += 1;
        }
    }

    // Normalize histogram if requested
    let hist_norm: Vec<f64> = if options.normalize_histogram {
        let total = flat_image.len() as f64;
        histogram
            .iter()
            .map(|&count| count as f64 / total)
            .collect()
    } else {
        histogram.iter().map(|&count| count as f64).collect()
    };

    // Store all histogram bins as features
    if !options.fast_mode {
        for (i, &count) in hist_norm.iter().enumerate() {
            features.insert(format!("histogram_bin_{}", i), count);
        }
    }

    // Calculate histogram statistics

    // Mode (bin with highest frequency)
    let mode_bin = hist_norm
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let mode_value = min_val + (mode_bin as f64 + 0.5) * (max_val - min_val) / bin_count as f64;
    features.insert("histogram_mode".to_string(), mode_value);

    // Entropy of histogram
    let entropy = hist_norm
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    features.insert("histogram_entropy".to_string(), entropy);

    // Uniformity
    let uniformity = hist_norm.iter().map(|&p| p * p).sum::<f64>();
    features.insert("histogram_uniformity".to_string(), uniformity);

    // Number of used bins
    let used_bins = hist_norm.iter().filter(|&&count| count > 0.0).count();
    features.insert("histogram_used_bins".to_string(), used_bins as f64);
    features.insert(
        "histogram_bin_utilization".to_string(),
        used_bins as f64 / bin_count as f64,
    );

    Ok(())
}
