//! Tamura texture features
//!
//! Tamura features are designed to correspond to human visual perception
//! and include coarseness, contrast, directionality, line-likeness, regularity, and roughness.

use crate::error::Result;
use image::{DynamicImage, GrayImage};
use ndarray::Array2;
use std::f32::consts::PI;

/// Tamura texture features
#[derive(Debug, Clone)]
pub struct TamuraFeatures {
    /// Coarseness - size of texture elements
    pub coarseness: f32,
    /// Contrast - intensity variations
    pub contrast: f32,
    /// Directionality - presence of oriented patterns
    pub directionality: f32,
    /// Line-likeness - presence of line-like structures (optional)
    pub line_likeness: Option<f32>,
    /// Regularity - how regular the texture pattern is (optional)
    pub regularity: Option<f32>,
    /// Roughness - combination of coarseness and contrast
    pub roughness: f32,
}

/// Compute Tamura texture features
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `compute_optional` - Whether to compute optional features (line-likeness, regularity)
///
/// # Returns
///
/// * Result containing Tamura features
pub fn compute_tamura_features(
    img: &DynamicImage,
    compute_optional: bool,
) -> Result<TamuraFeatures> {
    let gray = img.to_luma8();

    let coarseness = compute_coarseness(&gray)?;
    let contrast = compute_contrast(&gray)?;
    let directionality = compute_directionality(&gray)?;

    let (line_likeness, regularity) = if compute_optional {
        (
            Some(compute_line_likeness(&gray)?),
            Some(compute_regularity(&gray)?),
        )
    } else {
        (None, None)
    };

    // Roughness = Coarseness + Contrast
    let roughness = coarseness + contrast;

    Ok(TamuraFeatures {
        coarseness,
        contrast,
        directionality,
        line_likeness,
        regularity,
        roughness,
    })
}

/// Compute coarseness feature
///
/// Coarseness relates to the size of texture elements
fn compute_coarseness(img: &GrayImage) -> Result<f32> {
    let (width, height) = img.dimensions();
    let max_k = 5; // Maximum window size = 2^5 = 32

    // Compute average images at different scales
    let mut averages = vec![Array2::zeros((height as usize, width as usize)); max_k + 1];

    // Original image
    for y in 0..height {
        for x in 0..width {
            averages[0][[y as usize, x as usize]] = img.get_pixel(x, y)[0] as f32;
        }
    }

    // Compute averages at different scales
    for k in 1..=max_k {
        let window_size = 1 << k; // 2^k
        let half_window = window_size / 2;

        for y in 0..height as usize {
            for x in 0..width as usize {
                let mut sum = 0.0;
                let mut count = 0;

                for dy in y.saturating_sub(half_window)..=(y + half_window).min(height as usize - 1)
                {
                    for dx in
                        x.saturating_sub(half_window)..=(x + half_window).min(width as usize - 1)
                    {
                        sum += averages[0][[dy, dx]];
                        count += 1;
                    }
                }

                averages[k][[y, x]] = sum / count as f32;
            }
        }
    }

    // Compute differences between scales
    let mut s_best = Array2::zeros((height as usize, width as usize));

    for y in 0..height as usize {
        for x in 0..width as usize {
            let mut max_e = 0.0;
            let mut best_k = 1;

            for k in 1..max_k {
                let e = compute_e(&averages, x, y, k);
                if e > max_e {
                    max_e = e;
                    best_k = k;
                }
            }

            s_best[[y, x]] = (1 << best_k) as f32;
        }
    }

    // Average coarseness
    Ok(s_best.mean().unwrap_or(1.0))
}

/// Compute E value for coarseness
fn compute_e(averages: &[Array2<f32>], x: usize, y: usize, k: usize) -> f32 {
    let (height, width) = averages[0].dim();
    let d = 1 << (k - 1); // 2^(k-1)

    let mut e_h = 0.0;
    let mut e_v = 0.0;

    // Horizontal difference
    if x >= d && x + d < width {
        e_h = (averages[k][[y, x + d]] - averages[k][[y, x.saturating_sub(d)]]).abs();
    }

    // Vertical difference
    if y >= d && y + d < height {
        e_v = (averages[k][[y + d, x]] - averages[k][[y.saturating_sub(d), x]]).abs();
    }

    e_h.max(e_v)
}

/// Compute contrast feature
///
/// Contrast measures the intensity variations in the image
fn compute_contrast(img: &GrayImage) -> Result<f32> {
    let (width, height) = img.dimensions();
    let n = (width * height) as f32;

    // Compute mean
    let mut mean = 0.0;
    for pixel in img.pixels() {
        mean += pixel[0] as f32;
    }
    mean /= n;

    // Compute moments
    let mut variance = 0.0;
    let mut kurtosis = 0.0;

    for pixel in img.pixels() {
        let diff = pixel[0] as f32 - mean;
        variance += diff * diff;
        kurtosis += diff.powi(4);
    }

    variance /= n;

    // Handle uniform images (zero variance)
    if variance < 1e-10 {
        return Ok(0.0);
    }

    kurtosis = kurtosis / n / variance.powi(2);

    // Contrast formula
    let sigma = variance.sqrt();
    let alpha4 = kurtosis.max(0.01); // Avoid division by zero or negative values

    Ok(sigma / alpha4.powf(0.25))
}

/// Compute directionality feature
///
/// Directionality measures the presence of oriented patterns
fn compute_directionality(img: &GrayImage) -> Result<f32> {
    let (width, height) = img.dimensions();

    // Compute gradients
    let mut hist = vec![0.0; 16]; // 16 bins for directions
    let threshold = 10.0; // Gradient magnitude threshold

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Sobel gradients
            let gx = img.get_pixel(x + 1, y)[0] as f32 - img.get_pixel(x - 1, y)[0] as f32;
            let gy = img.get_pixel(x, y + 1)[0] as f32 - img.get_pixel(x, y - 1)[0] as f32;

            let magnitude = (gx * gx + gy * gy).sqrt();

            if magnitude > threshold {
                let angle = gy.atan2(gx);
                let bin = ((angle + PI) / (2.0 * PI) * 16.0) as usize % 16;
                hist[bin] += magnitude;
            }
        }
    }

    // Normalize histogram
    let sum: f32 = hist.iter().sum();
    if sum > 0.0 {
        for h in &mut hist {
            *h /= sum;
        }
    }

    // Find peaks in histogram
    let mut peaks = Vec::new();
    for i in 0..16 {
        let prev = hist[(i + 15) % 16];
        let curr = hist[i];
        let next = hist[(i + 1) % 16];

        if curr > prev && curr > next && curr > 0.05 {
            peaks.push(i);
        }
    }

    // Compute directionality based on histogram sharpness
    let mut entropy = 0.0;
    for &h in &hist {
        if h > 0.0 {
            entropy -= h * h.ln();
        }
    }

    // Lower entropy means higher directionality
    Ok(1.0 / (1.0 + entropy))
}

/// Compute line-likeness feature (optional)
///
/// Line-likeness measures the presence of line-like structures
fn compute_line_likeness(img: &GrayImage) -> Result<f32> {
    let (width, height) = img.dimensions();

    // Use co-occurrence matrix in different directions
    let mut line_strength = 0.0;
    let directions = [(1, 0), (0, 1), (1, 1), (1, -1)];

    for &(dx, dy) in &directions {
        let mut co_occurrence = 0.0;
        let mut count = 0;

        for y in 1..(height as i32 - 1) {
            for x in 1..(width as i32 - 1) {
                let x2 = x + dx;
                let y2 = y + dy;

                if x2 >= 0 && x2 < width as i32 && y2 >= 0 && y2 < height as i32 {
                    let p1 = img.get_pixel(x as u32, y as u32)[0] as f32;
                    let p2 = img.get_pixel(x2 as u32, y2 as u32)[0] as f32;

                    co_occurrence += (p1 - p2).abs();
                    count += 1;
                }
            }
        }

        if count > 0 {
            line_strength += co_occurrence / count as f32;
        }
    }

    Ok(line_strength / 4.0)
}

/// Compute regularity feature (optional)
///
/// Regularity measures how regular the texture pattern is
fn compute_regularity(img: &GrayImage) -> Result<f32> {
    let coarseness = compute_coarseness(img)?;
    let contrast = compute_contrast(img)?;

    // Simplified regularity based on variance of local features
    let (width, height) = img.dimensions();
    let window_size = (coarseness as usize).max(4);

    let mut local_variances = Vec::new();

    for y in (0..height as usize).step_by(window_size) {
        for x in (0..width as usize).step_by(window_size) {
            let mut values = Vec::new();

            for dy in 0..window_size {
                for dx in 0..window_size {
                    if y + dy < height as usize && x + dx < width as usize {
                        values.push(img.get_pixel((x + dx) as u32, (y + dy) as u32)[0] as f32);
                    }
                }
            }

            if !values.is_empty() {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let variance =
                    values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
                local_variances.push(variance);
            }
        }
    }

    // Regularity is inverse of variance of local variances
    if !local_variances.is_empty() {
        let mean_var = local_variances.iter().sum::<f32>() / local_variances.len() as f32;
        let var_of_var = local_variances
            .iter()
            .map(|&v| (v - mean_var).powi(2))
            .sum::<f32>()
            / local_variances.len() as f32;

        Ok(1.0 / (1.0 + var_of_var.sqrt() / contrast))
    } else {
        Ok(0.5)
    }
}

/// Quick Tamura features for real-time applications
pub fn compute_tamura_features_fast(img: &DynamicImage) -> Result<TamuraFeatures> {
    let gray = img.to_luma8();

    // Simplified coarseness using edge density
    let coarseness = compute_coarseness_fast(&gray)?;

    // Basic contrast
    let contrast = compute_contrast(&gray)?;

    // Simplified directionality
    let directionality = compute_directionality_fast(&gray)?;

    let roughness = coarseness + contrast;

    Ok(TamuraFeatures {
        coarseness,
        contrast,
        directionality,
        line_likeness: None,
        regularity: None,
        roughness,
    })
}

/// Fast coarseness computation
fn compute_coarseness_fast(img: &GrayImage) -> Result<f32> {
    let (width, height) = img.dimensions();
    let mut edge_count = 0;

    // Count edges using simple gradient
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let dx = img.get_pixel(x + 1, y)[0] as i32 - img.get_pixel(x - 1, y)[0] as i32;
            let dy = img.get_pixel(x, y + 1)[0] as i32 - img.get_pixel(x, y - 1)[0] as i32;

            if dx.abs() > 30 || dy.abs() > 30 {
                edge_count += 1;
            }
        }
    }

    // Lower edge density means higher coarseness
    let edge_density = edge_count as f32 / ((width - 2) * (height - 2)) as f32;
    Ok(1.0 / (1.0 + edge_density * 10.0))
}

/// Fast directionality computation
fn compute_directionality_fast(img: &GrayImage) -> Result<f32> {
    let (width, height) = img.dimensions();
    let mut hist = vec![0.0; 8]; // 8 bins for speed

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let gx = img.get_pixel(x + 1, y)[0] as f32 - img.get_pixel(x - 1, y)[0] as f32;
            let gy = img.get_pixel(x, y + 1)[0] as f32 - img.get_pixel(x, y - 1)[0] as f32;

            if gx.abs() > 10.0 || gy.abs() > 10.0 {
                let angle = gy.atan2(gx);
                let bin = ((angle + PI) / (2.0 * PI) * 8.0) as usize % 8;
                hist[bin] += 1.0;
            }
        }
    }

    // Normalize and compute entropy
    let sum: f32 = hist.iter().sum();
    let mut entropy = 0.0;

    if sum > 0.0 {
        for h in &mut hist {
            *h /= sum;
            if *h > 0.0 {
                entropy -= *h * h.ln();
            }
        }
    }

    Ok(1.0 / (1.0 + entropy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tamura_features() {
        let img = DynamicImage::new_luma8(50, 50);
        let result = compute_tamura_features(&img, false);
        assert!(result.is_ok());

        let features = result.unwrap();
        assert!(features.coarseness >= 0.0);
        assert!(features.contrast >= 0.0);
        assert!(features.directionality >= 0.0);
        assert!(features.roughness >= 0.0);
    }

    #[test]
    fn test_tamura_features_with_optional() {
        let img = DynamicImage::new_luma8(30, 30);
        let result = compute_tamura_features(&img, true);
        assert!(result.is_ok());

        let features = result.unwrap();
        assert!(features.line_likeness.is_some());
        assert!(features.regularity.is_some());
    }

    #[test]
    fn test_tamura_features_fast() {
        let img = DynamicImage::new_luma8(50, 50);
        let result = compute_tamura_features_fast(&img);
        assert!(result.is_ok());
    }

    #[test]
    fn test_contrast_computation() {
        // Create image with known contrast
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let value = if (x + y) % 2 == 0 { 0 } else { 255 };
                img.put_pixel(x, y, image::Luma([value]));
            }
        }

        let contrast = compute_contrast(&img).unwrap();
        assert!(contrast > 0.0);
    }
}
