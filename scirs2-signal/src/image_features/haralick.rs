// Haralick texture feature extraction for images

use super::types::ImageFeatureOptions;
use super::utils::compute_glcm;
use crate::error::SignalResult;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;

#[allow(unused_imports)]
/// Extract Haralick texture features from an image
#[allow(dead_code)]
pub fn extract_haralick_features(
    image: &Array2<f64>,
    options: &ImageFeatureOptions,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    // Compute the gray-level co-occurrence matrix (GLCM)
    let glcm = compute_glcm(image, options.cooccurrence_distance, 8);

    // Normalized GLCM
    let sum = glcm.sum();
    let norm_glcm = if sum > 0.0 {
        glcm.mapv(|x| x / sum)
    } else {
        glcm
    };

    // Calculate Haralick texture features

    // 1. Angular Second Moment (Energy)
    let asm = norm_glcm.iter().map(|&x| x * x).sum::<f64>();
    features.insert("haralick_energy".to_string(), asm);

    // 2. Contrast
    let mut contrast = 0.0;
    let n = norm_glcm.shape()[0]; // Matrix size
    for i in 0..n {
        for j in 0..n {
            contrast += (i as isize - j as isize).pow(2) as f64 * norm_glcm[[i, j]];
        }
    }
    features.insert("haralick_contrast".to_string(), contrast);

    // 3. Correlation
    let mut mean_i = 0.0;
    let mut mean_j = 0.0;
    let mut std_i = 0.0;
    let mut std_j = 0.0;

    // Calculate marginal means
    for i in 0..n {
        for j in 0..n {
            mean_i += i as f64 * norm_glcm[[i, j]];
            mean_j += j as f64 * norm_glcm[[i, j]];
        }
    }

    // Calculate marginal standard deviations
    for i in 0..n {
        for j in 0..n {
            std_i += (i as f64 - mean_i).powi(2) * norm_glcm[[i, j]];
            std_j += (j as f64 - mean_j).powi(2) * norm_glcm[[i, j]];
        }
    }
    std_i = std_i.sqrt();
    std_j = std_j.sqrt();

    // Calculate correlation
    let mut correlation = 0.0;
    if std_i > 1e-10 && std_j > 1e-10 {
        for i in 0..n {
            for j in 0..n {
                correlation +=
                    (i as f64 - mean_i) * (j as f64 - mean_j) * norm_glcm[[i, j]] / (std_i * std_j);
            }
        }
    }
    features.insert("haralick_correlation".to_string(), correlation);

    // 4. Sum of Squares (Variance)
    let mut variance = 0.0;
    for i in 0..n {
        for j in 0..n {
            variance += (i as f64 - mean_i).powi(2) * norm_glcm[[i, j]];
        }
    }
    features.insert("haralick_variance".to_string(), variance);

    // 5. Inverse Difference Moment (Homogeneity)
    let mut idm = 0.0;
    for i in 0..n {
        for j in 0..n {
            idm += norm_glcm[[i, j]] / (1.0 + (i as isize - j as isize).pow(2) as f64);
        }
    }
    features.insert("haralick_homogeneity".to_string(), idm);

    // 6. Sum Average
    let mut p_x_plus_y = vec![0.0; 2 * n - 1];
    for i in 0..n {
        for j in 0..n {
            p_x_plus_y[i + j] += norm_glcm[[i, j]];
        }
    }

    let sum_avg = p_x_plus_y
        .iter()
        .enumerate()
        .map(|(k, &p)| (k + 2) as f64 * p)
        .sum::<f64>();
    features.insert("haralick_sum_average".to_string(), sum_avg);

    // 7. Sum Entropy
    let sum_entropy = p_x_plus_y
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    features.insert("haralick_sum_entropy".to_string(), sum_entropy);

    // 8. Entropy
    let entropy = norm_glcm
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    features.insert("haralick_entropy".to_string(), entropy);

    // 9. Difference Variance
    let mut p_x_minus_y = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            let diff = (i as isize - j as isize).unsigned_abs();
            p_x_minus_y[diff] += norm_glcm[[i, j]];
        }
    }

    let diff_mean = p_x_minus_y
        .iter()
        .enumerate()
        .map(|(k, &p)| k as f64 * p)
        .sum::<f64>();
    let diff_var = p_x_minus_y
        .iter()
        .enumerate()
        .map(|(k, &p)| (k as f64 - diff_mean).powi(2) * p)
        .sum::<f64>();
    features.insert("haralick_difference_variance".to_string(), diff_var);

    // 10. Difference Entropy
    let diff_entropy = p_x_minus_y
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    features.insert("haralick_difference_entropy".to_string(), diff_entropy);

    Ok(())
}
