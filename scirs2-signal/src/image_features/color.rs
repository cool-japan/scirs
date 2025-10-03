// Color-specific feature extraction for images

use crate::error::SignalResult;
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
use std::fmt::Debug;

#[allow(unused_imports)]
/// Extract color-specific features from a color image
#[allow(dead_code)]
pub fn extract_color_features<T>(
    image: &scirs2_core::ndarray::Array3<T>,
    features: &mut HashMap<String, f64>,
) -> SignalResult<()>
where
    T: Clone + Copy + Into<f64> + Debug,
{
    let shape = image.shape();
    let height = shape[0];
    let width = shape[1];
    let n_pixels = height * width;

    // RGB means (should already be in the channel features, but we need them here)
    let mut r_sum = 0.0;
    let mut g_sum = 0.0;
    let mut b_sum = 0.0;

    for i in 0..height {
        for j in 0..width {
            r_sum += image[[i, j, 0]].into();
            g_sum += image[[i, j, 1]].into();
            b_sum += image[[i, j, 2]].into();
        }
    }

    let r_mean = r_sum / n_pixels as f64;
    let g_mean = g_sum / n_pixels as f64;
    let b_mean = b_sum / n_pixels as f64;

    // Calculate color ratios
    features.insert(
        "color_ratio_rg".to_string(),
        if g_mean > 0.0 {
            r_mean / g_mean
        } else {
            f64::MAX
        },
    );
    features.insert(
        "color_ratio_rb".to_string(),
        if b_mean > 0.0 {
            r_mean / b_mean
        } else {
            f64::MAX
        },
    );
    features.insert(
        "color_ratio_gb".to_string(),
        if b_mean > 0.0 {
            g_mean / b_mean
        } else {
            f64::MAX
        },
    );

    // Calculate color standard deviations
    let mut r_var_sum = 0.0;
    let mut g_var_sum = 0.0;
    let mut b_var_sum = 0.0;

    for i in 0..height {
        for j in 0..width {
            r_var_sum += (image[[i, j, 0]].into() - r_mean).powi(2);
            g_var_sum += (image[[i, j, 1]].into() - g_mean).powi(2);
            b_var_sum += (image[[i, j, 2]].into() - b_mean).powi(2);
        }
    }

    let r_std = (r_var_sum / n_pixels as f64).sqrt();
    let g_std = (g_var_sum / n_pixels as f64).sqrt();
    let b_std = (b_var_sum / n_pixels as f64).sqrt();

    // Calculate color homogeneity (ratio of min std to max std)
    let min_std = r_std.min(g_std).min(b_std);
    let max_std = r_std.max(g_std).max(b_std);

    if max_std > 0.0 {
        features.insert("color_homogeneity".to_string(), min_std / max_std);
    } else {
        features.insert("color_homogeneity".to_string(), 1.0);
    }

    // Calculate color dominance (which channel has highest mean)
    if r_mean > g_mean && r_mean > b_mean {
        features.insert("color_dominant".to_string(), 0.0); // Red
    } else if g_mean > r_mean && g_mean > b_mean {
        features.insert("color_dominant".to_string(), 1.0); // Green
    } else {
        features.insert("color_dominant".to_string(), 2.0); // Blue
    }

    // Calculate "colorfulness" - a measure of color variety
    let rg_diff = Array2::from_shape_fn((height, width), |(i, j)| {
        (image[[i, j, 0]].into() - image[[i, j, 1]].into()).abs()
    });

    let yb_diff = Array2::from_shape_fn((height, width), |(i, j)| {
        ((image[[i, j, 0]].into() + image[[i, j, 1]].into()) / 2.0 - image[[i, j, 2]].into()).abs()
    });

    let rg_mean = rg_diff.iter().sum::<f64>() / n_pixels as f64;
    let rg_std =
        (rg_diff.iter().map(|&x| (x - rg_mean).powi(2)).sum::<f64>() / n_pixels as f64).sqrt();

    let yb_mean = yb_diff.iter().sum::<f64>() / n_pixels as f64;
    let yb_std =
        (yb_diff.iter().map(|&x| (x - yb_mean).powi(2)).sum::<f64>() / n_pixels as f64).sqrt();

    let colorfulness =
        (rg_std.powi(2) + yb_std.powi(2)).sqrt() + 0.3 * (rg_mean.powi(2) + yb_mean.powi(2)).sqrt();
    features.insert("colorfulness".to_string(), colorfulness);

    // Calculate hue histogram
    let mut hue_hist = [0; 18]; // 20-degree bins

    for i in 0..height {
        for j in 0..width {
            let r = image[[i, j, 0]].into();
            let g = image[[i, j, 1]].into();
            let b = image[[i, j, 2]].into();

            // Calculate hue (in degrees)
            let max = r.max(g).max(b);
            let min = r.min(g).min(b);

            if max - min < 1e-6 {
                continue; // Gray pixel, no hue
            }

            let mut hue;
            if max == r {
                hue = 60.0 * ((g - b) / (max - min)) % 360.0;
            } else if max == g {
                hue = 60.0 * ((b - r) / (max - min) + 2.0);
            } else {
                hue = 60.0 * ((r - g) / (max - min) + 4.0);
            }

            if hue < 0.0 {
                hue += 360.0;
            }

            // Bin the hue
            let bin = (hue / 20.0).floor() as usize % 18;
            hue_hist[bin] += 1;
        }
    }

    // Normalize hue histogram
    let hue_hist_norm: Vec<f64> = hue_hist
        .iter()
        .map(|&count| count as f64 / n_pixels as f64)
        .collect();

    // Calculate hue uniformity
    let hue_uniformity = hue_hist_norm.iter().map(|&p| p * p).sum::<f64>();
    features.insert("hue_uniformity".to_string(), hue_uniformity);

    // Calculate hue entropy
    let hue_entropy = hue_hist_norm
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    features.insert("hue_entropy".to_string(), hue_entropy);

    Ok(())
}
