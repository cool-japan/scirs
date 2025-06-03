//! Gray Level Co-occurrence Matrix (GLCM) for texture analysis
//!
//! GLCM is a statistical method of examining texture that considers
//! the spatial relationship of pixels.

use crate::error::Result;
use image::{DynamicImage, GrayImage};
use ndarray::{Array2, Axis};

/// Direction for GLCM computation
#[derive(Debug, Clone, Copy)]
pub enum GLCMDirection {
    /// Horizontal (0 degrees)
    Horizontal,
    /// Vertical (90 degrees)
    Vertical,
    /// Diagonal (45 degrees)
    Diagonal,
    /// Anti-diagonal (135 degrees)
    AntiDiagonal,
}

impl GLCMDirection {
    /// Get offset for this direction
    fn get_offset(&self, distance: i32) -> (i32, i32) {
        match self {
            GLCMDirection::Horizontal => (distance, 0),
            GLCMDirection::Vertical => (0, distance),
            GLCMDirection::Diagonal => (distance, distance),
            GLCMDirection::AntiDiagonal => (distance, -distance),
        }
    }
}

/// Parameters for GLCM computation
#[derive(Debug, Clone)]
pub struct GLCMParams {
    /// Number of gray levels to quantize to
    pub levels: usize,
    /// Distance between pixel pairs
    pub distance: i32,
    /// Direction to compute GLCM
    pub direction: GLCMDirection,
    /// Whether to make the matrix symmetric
    pub symmetric: bool,
    /// Whether to normalize the matrix
    pub normalize: bool,
}

impl Default for GLCMParams {
    fn default() -> Self {
        Self {
            levels: 8,
            distance: 1,
            direction: GLCMDirection::Horizontal,
            symmetric: true,
            normalize: true,
        }
    }
}

/// Compute Gray Level Co-occurrence Matrix
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `params` - GLCM parameters
///
/// # Returns
///
/// * Result containing the GLCM as a 2D array
pub fn compute_glcm(img: &DynamicImage, params: &GLCMParams) -> Result<Array2<f64>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Quantize the image to specified levels
    let quantized = quantize_image(&gray, params.levels);

    // Initialize GLCM
    let mut glcm = Array2::zeros((params.levels, params.levels));

    // Get direction offset
    let (dx, dy) = params.direction.get_offset(params.distance);

    // Compute co-occurrences
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let x2 = x + dx;
            let y2 = y + dy;

            // Check bounds
            if x2 >= 0 && x2 < width as i32 && y2 >= 0 && y2 < height as i32 {
                let i = quantized[[y as usize, x as usize]];
                let j = quantized[[y2 as usize, x2 as usize]];

                glcm[[i, j]] += 1.0;

                // Add symmetric pair
                if params.symmetric {
                    glcm[[j, i]] += 1.0;
                }
            }
        }
    }

    // Normalize if requested
    if params.normalize {
        let sum = glcm.sum();
        if sum > 0.0 {
            glcm /= sum;
        }
    }

    Ok(glcm)
}

/// Quantize image to specified number of levels
fn quantize_image(img: &GrayImage, levels: usize) -> Array2<usize> {
    let (width, height) = img.dimensions();
    let mut quantized = Array2::zeros((height as usize, width as usize));

    let scale = 256.0 / levels as f32;

    for y in 0..height {
        for x in 0..width {
            let value = img.get_pixel(x, y)[0] as f32;
            let level = (value / scale).floor() as usize;
            quantized[[y as usize, x as usize]] = level.min(levels - 1);
        }
    }

    quantized
}

/// Haralick texture features from GLCM
#[derive(Debug, Clone)]
pub struct HaralickFeatures {
    /// Angular Second Moment (Energy)
    pub energy: f64,
    /// Contrast
    pub contrast: f64,
    /// Correlation
    pub correlation: f64,
    /// Homogeneity (Inverse Difference Moment)
    pub homogeneity: f64,
    /// Entropy
    pub entropy: f64,
    /// Dissimilarity
    pub dissimilarity: f64,
    /// Maximum probability
    pub max_probability: f64,
}

/// Compute Haralick texture features from GLCM
///
/// # Arguments
///
/// * `glcm` - Gray Level Co-occurrence Matrix
///
/// # Returns
///
/// * Haralick features
pub fn compute_haralick_features(glcm: &Array2<f64>) -> HaralickFeatures {
    let (rows, cols) = glcm.dim();

    // Compute marginal probabilities
    let px = glcm.sum_axis(Axis(1));
    let py = glcm.sum_axis(Axis(0));

    // Compute means
    let mut mean_x = 0.0;
    let mut mean_y = 0.0;

    for i in 0..rows {
        mean_x += i as f64 * px[i];
        mean_y += i as f64 * py[i];
    }

    // Compute standard deviations
    let mut std_x = 0.0;
    let mut std_y = 0.0;

    for i in 0..rows {
        std_x += (i as f64 - mean_x).powi(2) * px[i];
        std_y += (i as f64 - mean_y).powi(2) * py[i];
    }

    std_x = std_x.sqrt();
    std_y = std_y.sqrt();

    // Compute features
    let mut energy = 0.0;
    let mut contrast = 0.0;
    let mut correlation = 0.0;
    let mut homogeneity = 0.0;
    let mut entropy = 0.0;
    let mut dissimilarity = 0.0;
    let mut max_probability = 0.0f64;

    for i in 0..rows {
        for j in 0..cols {
            let p = glcm[[i, j]];

            if p > 0.0 {
                energy += p * p;
                contrast += (i as f64 - j as f64).powi(2) * p;
                homogeneity += p / (1.0 + (i as f64 - j as f64).abs());
                entropy -= p * p.ln();
                dissimilarity += (i as f64 - j as f64).abs() * p;
                max_probability = max_probability.max(p);

                if std_x > 0.0 && std_y > 0.0 {
                    correlation +=
                        ((i as f64 - mean_x) * (j as f64 - mean_y) * p) / (std_x * std_y);
                }
            }
        }
    }

    HaralickFeatures {
        energy,
        contrast,
        correlation,
        homogeneity,
        entropy,
        dissimilarity,
        max_probability,
    }
}

/// Compute GLCM for multiple directions and aggregate features
///
/// # Arguments
///
/// * `img` - Input image
/// * `distance` - Distance parameter
/// * `levels` - Number of gray levels
///
/// # Returns
///
/// * Average Haralick features across all directions
pub fn compute_multi_direction_glcm_features(
    img: &DynamicImage,
    distance: i32,
    levels: usize,
) -> Result<HaralickFeatures> {
    let directions = [
        GLCMDirection::Horizontal,
        GLCMDirection::Vertical,
        GLCMDirection::Diagonal,
        GLCMDirection::AntiDiagonal,
    ];

    let mut all_features = Vec::new();

    for direction in &directions {
        let params = GLCMParams {
            levels,
            distance,
            direction: *direction,
            ..Default::default()
        };

        let glcm = compute_glcm(img, &params)?;
        let features = compute_haralick_features(&glcm);
        all_features.push(features);
    }

    // Average features
    let n = all_features.len() as f64;

    Ok(HaralickFeatures {
        energy: all_features.iter().map(|f| f.energy).sum::<f64>() / n,
        contrast: all_features.iter().map(|f| f.contrast).sum::<f64>() / n,
        correlation: all_features.iter().map(|f| f.correlation).sum::<f64>() / n,
        homogeneity: all_features.iter().map(|f| f.homogeneity).sum::<f64>() / n,
        entropy: all_features.iter().map(|f| f.entropy).sum::<f64>() / n,
        dissimilarity: all_features.iter().map(|f| f.dissimilarity).sum::<f64>() / n,
        max_probability: all_features.iter().map(|f| f.max_probability).sum::<f64>() / n,
    })
}

/// Extended GLCM features including higher-order statistics
#[derive(Debug, Clone)]
pub struct ExtendedGLCMFeatures {
    /// Basic Haralick features
    pub haralick: HaralickFeatures,
    /// Cluster shade
    pub cluster_shade: f64,
    /// Cluster prominence
    pub cluster_prominence: f64,
    /// Sum average
    pub sum_average: f64,
    /// Sum variance
    pub sum_variance: f64,
    /// Sum entropy
    pub sum_entropy: f64,
    /// Difference variance
    pub diff_variance: f64,
    /// Difference entropy
    pub diff_entropy: f64,
}

/// Compute extended GLCM features
pub fn compute_extended_glcm_features(glcm: &Array2<f64>) -> ExtendedGLCMFeatures {
    let haralick = compute_haralick_features(glcm);
    let (n, _) = glcm.dim();

    // Compute p_x+y and p_x-y
    let mut p_sum = vec![0.0; 2 * n - 1];
    let mut p_diff = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            let p = glcm[[i, j]];
            p_sum[i + j] += p;
            p_diff[(i as i32 - j as i32).unsigned_abs() as usize] += p;
        }
    }

    // Compute sum and difference statistics
    let mut sum_average = 0.0;
    let mut sum_entropy = 0.0;
    let mut diff_entropy = 0.0;

    for (k, &p_sum_k) in p_sum.iter().enumerate() {
        if p_sum_k > 0.0 {
            sum_average += k as f64 * p_sum_k;
            sum_entropy -= p_sum_k * p_sum_k.ln();
        }
    }

    for &p_diff_k in &p_diff {
        if p_diff_k > 0.0 {
            diff_entropy -= p_diff_k * p_diff_k.ln();
        }
    }

    // Compute sum variance
    let mut sum_variance = 0.0;
    for (k, &p_sum_k) in p_sum.iter().enumerate() {
        sum_variance += (k as f64 - sum_average).powi(2) * p_sum_k;
    }

    // Compute difference variance
    let mut diff_average = 0.0;
    for (k, &p_diff_k) in p_diff.iter().enumerate() {
        diff_average += k as f64 * p_diff_k;
    }

    let mut diff_variance = 0.0;
    for (k, &p_diff_k) in p_diff.iter().enumerate() {
        diff_variance += (k as f64 - diff_average).powi(2) * p_diff_k;
    }

    // Compute cluster shade and prominence
    let (_px, _py, mean_x, mean_y) = compute_marginals(glcm);

    let mut cluster_shade = 0.0;
    let mut cluster_prominence = 0.0;

    for i in 0..n {
        for j in 0..n {
            let term = i as f64 - mean_x + j as f64 - mean_y;
            cluster_shade += term.powi(3) * glcm[[i, j]];
            cluster_prominence += term.powi(4) * glcm[[i, j]];
        }
    }

    ExtendedGLCMFeatures {
        haralick,
        cluster_shade,
        cluster_prominence,
        sum_average,
        sum_variance,
        sum_entropy,
        diff_variance,
        diff_entropy,
    }
}

/// Compute marginal probabilities and means
fn compute_marginals(glcm: &Array2<f64>) -> (Vec<f64>, Vec<f64>, f64, f64) {
    let (n, _) = glcm.dim();

    let px = glcm.sum_axis(Axis(1)).to_vec();
    let py = glcm.sum_axis(Axis(0)).to_vec();

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;

    for i in 0..n {
        mean_x += i as f64 * px[i];
        mean_y += i as f64 * py[i];
    }

    (px, py, mean_x, mean_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glcm_basic() {
        let img = DynamicImage::new_luma8(10, 10);
        let params = GLCMParams::default();

        let result = compute_glcm(&img, &params);
        assert!(result.is_ok());

        let glcm = result.unwrap();
        assert_eq!(glcm.dim(), (8, 8));
    }

    #[test]
    fn test_haralick_features() {
        let mut glcm = Array2::zeros((4, 4));
        glcm[[0, 0]] = 0.25;
        glcm[[1, 1]] = 0.25;
        glcm[[2, 2]] = 0.25;
        glcm[[3, 3]] = 0.25;

        let features = compute_haralick_features(&glcm);

        // Perfect diagonal should have high energy and low contrast
        assert!(features.energy > 0.0);
        assert_eq!(features.contrast, 0.0);
    }

    #[test]
    fn test_multi_direction() {
        let img = DynamicImage::new_luma8(20, 20);
        let result = compute_multi_direction_glcm_features(&img, 1, 8);
        assert!(result.is_ok());
    }

    #[test]
    fn test_quantization() {
        let img = GrayImage::new(4, 4);
        let quantized = quantize_image(&img, 4);

        assert_eq!(quantized.dim(), (4, 4));
        assert!(quantized.iter().all(|&v| v < 4));
    }
}
