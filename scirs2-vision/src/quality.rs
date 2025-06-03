//! Image quality assessment metrics
//!
//! This module provides various metrics for assessing image quality,
//! including PSNR, SSIM, and other perceptual metrics.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GenericImageView, GrayImage};
use ndarray::{s, Array2};

/// Compute Peak Signal-to-Noise Ratio (PSNR) between two images
///
/// # Arguments
///
/// * `img1` - First image (reference)
/// * `img2` - Second image (distorted)
/// * `max_value` - Maximum possible pixel value (255 for 8-bit images)
///
/// # Returns
///
/// * PSNR value in dB (higher is better, typical range: 20-40 dB)
///
/// # Example
///
/// ```rust
/// use scirs2_vision::quality::psnr;
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img1 = image::open("examples/input/input.jpg").unwrap();
/// let img2 = img1.clone(); // Same image for demonstration
/// let psnr_value = psnr(&img1, &img2, 255.0)?;
/// println!("PSNR: {:.2} dB", psnr_value);
/// # Ok(())
/// # }
/// ```
pub fn psnr(img1: &DynamicImage, img2: &DynamicImage, max_value: f32) -> Result<f32> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();

    if gray1.dimensions() != gray2.dimensions() {
        return Err(VisionError::DimensionMismatch(
            "Images must have the same dimensions".to_string(),
        ));
    }

    let (width, height) = gray1.dimensions();
    let mut mse = 0.0f32;

    for y in 0..height {
        for x in 0..width {
            let p1 = gray1.get_pixel(x, y)[0] as f32;
            let p2 = gray2.get_pixel(x, y)[0] as f32;
            let diff = p1 - p2;
            mse += diff * diff;
        }
    }

    mse /= (width * height) as f32;

    if mse == 0.0 {
        // Identical images
        Ok(f32::INFINITY)
    } else {
        Ok(20.0 * (max_value / mse.sqrt()).log10())
    }
}

/// Parameters for SSIM computation
#[derive(Debug, Clone)]
pub struct SSIMParams {
    /// Gaussian window size (should be odd)
    pub window_size: usize,
    /// Standard deviation for Gaussian window
    pub sigma: f32,
    /// Constant to avoid division by zero (for luminance)
    pub k1: f32,
    /// Constant to avoid division by zero (for contrast)
    pub k2: f32,
}

impl Default for SSIMParams {
    fn default() -> Self {
        Self {
            window_size: 11,
            sigma: 1.5,
            k1: 0.01,
            k2: 0.03,
        }
    }
}

/// Compute Structural Similarity Index (SSIM) between two images
///
/// # Arguments
///
/// * `img1` - First image (reference)
/// * `img2` - Second image (distorted)
/// * `params` - SSIM parameters
///
/// # Returns
///
/// * SSIM value in range [0, 1] (1 means identical)
///
/// # Example
///
/// ```rust
/// use scirs2_vision::quality::{ssim, SSIMParams};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img1 = image::open("examples/input/input.jpg").unwrap();
/// let img2 = img1.clone();
/// let ssim_value = ssim(&img1, &img2, &SSIMParams::default())?;
/// println!("SSIM: {:.4}", ssim_value);
/// # Ok(())
/// # }
/// ```
pub fn ssim(img1: &DynamicImage, img2: &DynamicImage, params: &SSIMParams) -> Result<f32> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();

    if gray1.dimensions() != gray2.dimensions() {
        return Err(VisionError::DimensionMismatch(
            "Images must have the same dimensions".to_string(),
        ));
    }

    // Convert to arrays
    let arr1 = image_to_array(&gray1)?;
    let arr2 = image_to_array(&gray2)?;

    // Create Gaussian window
    let window = gaussian_window(params.window_size, params.sigma);

    // Compute SSIM map
    let ssim_map = ssim_map(&arr1, &arr2, &window, params)?;

    // Return mean SSIM (only for non-zero values)
    let half_window = params.window_size / 2;
    let (height, width) = ssim_map.dim();
    let mut sum = 0.0;
    let mut count = 0;

    for y in half_window..height - half_window {
        for x in half_window..width - half_window {
            sum += ssim_map[[y, x]];
            count += 1;
        }
    }

    if count > 0 {
        Ok(sum / count as f32)
    } else {
        Ok(0.0)
    }
}

/// Compute SSIM map (local SSIM values)
fn ssim_map(
    img1: &Array2<f32>,
    img2: &Array2<f32>,
    window: &Array2<f32>,
    params: &SSIMParams,
) -> Result<Array2<f32>> {
    let (height, width) = img1.dim();
    let half_window = params.window_size / 2;

    let mut ssim_map = Array2::zeros((height, width));

    let l = 255.0; // Dynamic range
    let c1 = (params.k1 * l).powi(2);
    let c2 = (params.k2 * l).powi(2);

    for y in half_window..height - half_window {
        for x in half_window..width - half_window {
            // Extract patches
            let patch1 = img1.slice(s![
                y - half_window..=y + half_window,
                x - half_window..=x + half_window
            ]);
            let patch2 = img2.slice(s![
                y - half_window..=y + half_window,
                x - half_window..=x + half_window
            ]);

            // Compute weighted statistics
            let sum_weights = window.sum();

            let mu1 = weighted_mean(&patch1, window, sum_weights);
            let mu2 = weighted_mean(&patch2, window, sum_weights);

            let sigma1_sq = weighted_variance(&patch1, window, mu1, sum_weights);
            let sigma2_sq = weighted_variance(&patch2, window, mu2, sum_weights);
            let sigma12 = weighted_covariance(&patch1, &patch2, window, mu1, mu2, sum_weights);

            // Compute SSIM
            let numerator = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2);
            let denominator = (mu1.powi(2) + mu2.powi(2) + c1) * (sigma1_sq + sigma2_sq + c2);

            ssim_map[[y, x]] = numerator / denominator;
        }
    }

    Ok(ssim_map)
}

/// Create a Gaussian window
fn gaussian_window(size: usize, sigma: f32) -> Array2<f32> {
    let half_size = size as i32 / 2;
    let mut window = Array2::zeros((size, size));
    let mut sum = 0.0;

    for y in -half_size..=half_size {
        for x in -half_size..=half_size {
            let value = (-(x * x + y * y) as f32 / (2.0 * sigma * sigma)).exp();
            window[[(y + half_size) as usize, (x + half_size) as usize]] = value;
            sum += value;
        }
    }

    // Normalize
    window /= sum;
    window
}

/// Compute weighted mean
fn weighted_mean(data: &ndarray::ArrayView2<f32>, weights: &Array2<f32>, sum_weights: f32) -> f32 {
    let mut sum = 0.0;
    for ((y, x), &value) in data.indexed_iter() {
        sum += value * weights[[y, x]];
    }
    sum / sum_weights
}

/// Compute weighted variance
fn weighted_variance(
    data: &ndarray::ArrayView2<f32>,
    weights: &Array2<f32>,
    mean: f32,
    sum_weights: f32,
) -> f32 {
    let mut sum = 0.0;
    for ((y, x), &value) in data.indexed_iter() {
        sum += weights[[y, x]] * (value - mean).powi(2);
    }
    sum / sum_weights
}

/// Compute weighted covariance
fn weighted_covariance(
    data1: &ndarray::ArrayView2<f32>,
    data2: &ndarray::ArrayView2<f32>,
    weights: &Array2<f32>,
    mean1: f32,
    mean2: f32,
    sum_weights: f32,
) -> f32 {
    let mut sum = 0.0;
    for ((y, x), &value1) in data1.indexed_iter() {
        let value2 = data2[[y, x]];
        sum += weights[[y, x]] * (value1 - mean1) * (value2 - mean2);
    }
    sum / sum_weights
}

/// Convert grayscale image to normalized array
fn image_to_array(img: &GrayImage) -> Result<Array2<f32>> {
    let (width, height) = img.dimensions();
    let mut array = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            array[[y as usize, x as usize]] = img.get_pixel(x, y)[0] as f32;
        }
    }

    Ok(array)
}

/// Multi-scale SSIM (MS-SSIM)
///
/// Computes SSIM at multiple scales and combines them
///
/// # Arguments
///
/// * `img1` - First image
/// * `img2` - Second image
/// * `params` - SSIM parameters
/// * `scales` - Number of scales to use
///
/// # Returns
///
/// * MS-SSIM value
pub fn ms_ssim(
    img1: &DynamicImage,
    img2: &DynamicImage,
    params: &SSIMParams,
    scales: usize,
) -> Result<f32> {
    // Weights for each scale (from the original paper)
    let weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333];
    let weights = &weights[..scales.min(5)];

    let mut overall_ssim = 1.0;
    let mut img1_scaled = img1.clone();
    let mut img2_scaled = img2.clone();

    for (i, &weight) in weights.iter().enumerate() {
        // Compute SSIM at current scale
        let ssim_value = ssim(&img1_scaled, &img2_scaled, params)?;

        // For the last scale, use the full SSIM
        // For other scales, use only contrast and structure
        if i == weights.len() - 1 {
            overall_ssim *= ssim_value.powf(weight);
        } else {
            // Simplified: use SSIM as proxy for contrast * structure
            overall_ssim *= ssim_value.powf(weight);

            // Downsample for next scale
            let (w, h) = img1_scaled.dimensions();
            img1_scaled = img1_scaled.resize(w / 2, h / 2, image::imageops::FilterType::Gaussian);
            img2_scaled = img2_scaled.resize(w / 2, h / 2, image::imageops::FilterType::Gaussian);
        }
    }

    Ok(overall_ssim)
}

/// Visual Information Fidelity (VIF) - simplified version
///
/// # Arguments
///
/// * `img1` - Reference image
/// * `img2` - Distorted image
///
/// # Returns
///
/// * VIF value (higher is better)
pub fn vif(img1: &DynamicImage, img2: &DynamicImage) -> Result<f32> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();

    if gray1.dimensions() != gray2.dimensions() {
        return Err(VisionError::DimensionMismatch(
            "Images must have the same dimensions".to_string(),
        ));
    }

    let arr1 = image_to_array(&gray1)?;
    let arr2 = image_to_array(&gray2)?;

    // Simplified VIF using variance ratios
    let window_size = 7;
    let half_window = window_size / 2;
    let (height, width) = arr1.dim();

    let mut num = 0.0;
    let mut den = 0.0;

    for y in half_window..height - half_window {
        for x in half_window..width - half_window {
            let patch1 = arr1.slice(s![
                y - half_window..=y + half_window,
                x - half_window..=x + half_window
            ]);
            let patch2 = arr2.slice(s![
                y - half_window..=y + half_window,
                x - half_window..=x + half_window
            ]);

            let var1 = variance(&patch1);
            let var2 = variance(&patch2);

            if var1 > 1e-7 {
                num += (var2 + 1.0).log2();
                den += (var1 + 1.0).log2();
            }
        }
    }

    Ok(if den > 0.0 { num / den } else { 0.0 })
}

/// Compute variance of an array view
fn variance(data: &ndarray::ArrayView2<f32>) -> f32 {
    let mean = data.mean().unwrap_or(0.0);
    data.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0)
}

/// Mean Absolute Error (MAE)
pub fn mae(img1: &DynamicImage, img2: &DynamicImage) -> Result<f32> {
    let gray1 = img1.to_luma8();
    let gray2 = img2.to_luma8();

    if gray1.dimensions() != gray2.dimensions() {
        return Err(VisionError::DimensionMismatch(
            "Images must have the same dimensions".to_string(),
        ));
    }

    let (width, height) = gray1.dimensions();
    let mut sum = 0.0f32;

    for y in 0..height {
        for x in 0..width {
            let p1 = gray1.get_pixel(x, y)[0] as f32;
            let p2 = gray2.get_pixel(x, y)[0] as f32;
            sum += (p1 - p2).abs();
        }
    }

    Ok(sum / (width * height) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_psnr_identical() {
        let img = DynamicImage::new_luma8(50, 50);
        let psnr_value = psnr(&img, &img, 255.0).unwrap();
        assert!(psnr_value.is_infinite());
    }

    #[test]
    fn test_psnr_different() {
        let img1 = DynamicImage::new_luma8(50, 50);
        let mut img2 = img1.to_luma8();

        // Add some noise
        for y in 0..50 {
            for x in 0..50 {
                let val = img2.get_pixel(x, y)[0];
                img2.put_pixel(x, y, image::Luma([val.saturating_add(10)]));
            }
        }

        let psnr_value = psnr(&img1, &DynamicImage::ImageLuma8(img2), 255.0).unwrap();
        assert!(psnr_value > 20.0 && psnr_value < 50.0);
    }

    #[test]
    fn test_ssim_identical() {
        // Create a non-uniform test image to avoid numerical issues with all-zero images
        let mut img_buf = image::GrayImage::new(50, 50);
        for (x, y, pixel) in img_buf.enumerate_pixels_mut() {
            *pixel = image::Luma([(x + y) as u8]);
        }
        let img = DynamicImage::ImageLuma8(img_buf);

        let ssim_value = ssim(&img, &img, &SSIMParams::default()).unwrap();
        assert!(
            (ssim_value - 1.0).abs() < 0.01,
            "SSIM of identical images should be ~1.0, got {}",
            ssim_value
        );
    }

    #[test]
    fn test_gaussian_window() {
        let window = gaussian_window(5, 1.0);
        assert_eq!(window.dim(), (5, 5));
        assert!((window.sum() - 1.0).abs() < 1e-6);

        // Center should have highest value
        assert!(window[[2, 2]] > window[[0, 0]]);
    }
}
