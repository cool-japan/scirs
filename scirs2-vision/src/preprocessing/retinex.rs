//! Retinex algorithms for image enhancement
//!
//! Retinex theory aims to explain human color perception and provides
//! algorithms for dynamic range compression and color constancy.

use crate::error::Result;
use image::{DynamicImage, GenericImageView, RgbImage};
use scirs2_core::ndarray::{Array2, Array3};

/// Single-Scale Retinex (SSR)
///
/// # Arguments
///
/// * `img` - Input image
/// * `sigma` - Standard deviation for Gaussian blur
///
/// # Returns
///
/// * Result containing enhanced image
#[allow(dead_code)]
pub fn single_scale_retinex(img: &DynamicImage, sigma: f32) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Convert to float and apply log
    let mut log_img = Array3::zeros((height as usize, width as usize, 3));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            for c in 0..3 {
                let value = pixel[c] as f32 / 255.0;
                log_img[[y as usize, x as usize, c]] = (value + 1e-6).ln();
            }
        }
    }

    // Apply Gaussian blur to get illumination
    let illumination = gaussian_blur_3d(&log_img, sigma)?;

    // Compute reflectance: R = log(I) - log(L)
    let reflectance = log_img - illumination;

    // Normalize and convert back
    let mut result = RgbImage::new(width, height);

    for c in 0..3 {
        let channel = reflectance.index_axis(scirs2_core::ndarray::Axis(2), c);
        let (min_val, max_val) = find_min_max(&channel);
        let range = max_val - min_val;

        for y in 0..height {
            for x in 0..width {
                let value = reflectance[[y as usize, x as usize, c]];
                let normalized = if range > 0.0 {
                    (value - min_val) / range
                } else {
                    0.5
                };

                let pixel = result.get_pixel_mut(x, y);
                pixel[c] = (normalized * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Multi-Scale Retinex (MSR)
///
/// # Arguments
///
/// * `img` - Input image
/// * `sigmas` - List of scales (standard deviations)
/// * `weights` - Optional weights for each scale
///
/// # Returns
///
/// * Result containing enhanced image
#[allow(dead_code)]
pub fn multi_scale_retinex(
    img: &DynamicImage,
    sigmas: &[f32],
    weights: Option<&[f32]>,
) -> Result<DynamicImage> {
    if sigmas.is_empty() {
        return single_scale_retinex(img, 15.0);
    }

    let default_weights = vec![1.0 / sigmas.len() as f32; sigmas.len()];
    let weights = weights.unwrap_or(&default_weights);

    if sigmas.len() != weights.len() {
        return Err(crate::error::VisionError::InvalidParameter(
            "Sigmas and weights must have the same length".to_string(),
        ));
    }

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Convert to float and apply log
    let mut log_img = Array3::zeros((height as usize, width as usize, 3));

    for y in 0..height {
        for x in 0..width {
            let pixel = rgb.get_pixel(x, y);
            for c in 0..3 {
                let value = pixel[c] as f32 / 255.0;
                log_img[[y as usize, x as usize, c]] = (value + 1e-6).ln();
            }
        }
    }

    // Compute weighted sum of retinex at different scales
    let mut msr = Array3::zeros((height as usize, width as usize, 3));

    for (i, &sigma) in sigmas.iter().enumerate() {
        let illumination = gaussian_blur_3d(&log_img, sigma)?;
        let retinex = &log_img - &illumination;
        msr = msr + weights[i] * retinex;
    }

    // Normalize and convert back
    let mut result = RgbImage::new(width, height);

    for c in 0..3 {
        let channel = msr.index_axis(scirs2_core::ndarray::Axis(2), c);
        let (min_val, max_val) = find_min_max(&channel);
        let range = max_val - min_val;

        for y in 0..height {
            for x in 0..width {
                let value = msr[[y as usize, x as usize, c]];
                let normalized = if range > 0.0 {
                    (value - min_val) / range
                } else {
                    0.5
                };

                let pixel = result.get_pixel_mut(x, y);
                pixel[c] = (normalized * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Multi-Scale Retinex with Color Restoration (MSRCR)
///
/// # Arguments
///
/// * `img` - Input image
/// * `sigmas` - List of scales
/// * `alpha` - Color restoration factor
/// * `beta` - Gain constant
/// * `gain` - Final gain
/// * `offset` - Final offset
///
/// # Returns
///
/// * Result containing enhanced image
#[allow(dead_code)]
pub fn msrcr(
    img: &DynamicImage,
    sigmas: &[f32],
    alpha: f32,
    beta: f32,
    gain: f32,
    offset: f32,
) -> Result<DynamicImage> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    // Get MSR result
    let msr_img = multi_scale_retinex(img, sigmas, None)?;
    let msr_rgb = msr_img.to_rgb8();

    // Compute color restoration function
    let mut result = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let orig_pixel = rgb.get_pixel(x, y);
            let msr_pixel = msr_rgb.get_pixel(x, y);

            // Compute intensity
            let intensity =
                (orig_pixel[0] as f32 + orig_pixel[1] as f32 + orig_pixel[2] as f32) / 3.0;

            let result_pixel = result.get_pixel_mut(x, y);

            for c in 0..3 {
                // Color restoration function
                let color_factor = if intensity > 0.0 {
                    (alpha * (orig_pixel[c] as f32 / intensity).ln() + beta).tanh()
                } else {
                    0.0
                };

                // Apply color restoration
                let value = msr_pixel[c] as f32 * (1.0 + color_factor);

                // Apply gain and offset
                let final_value = gain * value + offset;

                result_pixel[c] = final_value.clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(result))
}

/// Adaptive histogram equalization on Retinex output
#[allow(dead_code)]
pub fn retinex_with_clahe(
    img: &DynamicImage,
    sigmas: &[f32],
    clip_limit: f32,
    grid_size: usize,
) -> Result<DynamicImage> {
    // First apply MSR
    let msr_img = multi_scale_retinex(img, sigmas, None)?;

    // Then apply CLAHE
    crate::preprocessing::clahe(&msr_img, grid_size as u32, clip_limit)
}

/// Gaussian blur for 3D arrays (color images)
#[allow(dead_code)]
fn gaussian_blur_3d(img: &Array3<f32>, sigma: f32) -> Result<Array3<f32>> {
    let (height, width, channels) = img.dim();
    let mut blurred = Array3::zeros((height, width, channels));

    // Create Gaussian kernel
    let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd
    let kernel = create_gaussian_kernel(kernel_size, sigma);

    // Apply separable convolution to each channel
    for c in 0..channels {
        let channel = img.index_axis(scirs2_core::ndarray::Axis(2), c);
        let blurred_channel = separable_convolution(&channel, &kernel)?;

        for y in 0..height {
            for x in 0..width {
                blurred[[y, x, c]] = blurred_channel[[y, x]];
            }
        }
    }

    Ok(blurred)
}

/// Create 1D Gaussian kernel
#[allow(dead_code)]
fn create_gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
    let mut kernel = vec![0.0; size];
    let center = size / 2;
    let s = 2.0 * sigma * sigma;

    let mut sum = 0.0;

    for (i, kernel_item) in kernel.iter_mut().enumerate().take(size) {
        let x = i as f32 - center as f32;
        *kernel_item = (-x * x / s).exp();
        sum += *kernel_item;
    }

    // Normalize
    for k in &mut kernel {
        *k /= sum;
    }

    kernel
}

/// Separable convolution (horizontal then vertical)
#[allow(dead_code)]
fn separable_convolution(
    img: &scirs2_core::ndarray::ArrayView2<f32>,
    kernel: &[f32],
) -> Result<Array2<f32>> {
    let (height, width) = img.dim();
    let kernel_size = kernel.len();
    let pad = kernel_size / 2;

    // Horizontal pass
    let mut temp = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for (k, &kernel_k) in kernel.iter().enumerate().take(kernel_size) {
                let sx = x as i32 + k as i32 - pad as i32;
                if sx >= 0 && sx < width as i32 {
                    sum += img[[y, sx as usize]] * kernel_k;
                }
            }

            temp[[y, x]] = sum;
        }
    }

    // Vertical pass
    let mut result = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for (k, &kernel_k) in kernel.iter().enumerate().take(kernel_size) {
                let sy = y as i32 + k as i32 - pad as i32;
                if sy >= 0 && sy < height as i32 {
                    sum += temp[[sy as usize, x]] * kernel_k;
                }
            }

            result[[y, x]] = sum;
        }
    }

    Ok(result)
}

/// Find min and max values in 2D array
#[allow(dead_code)]
fn find_min_max(arr: &scirs2_core::ndarray::ArrayView2<f32>) -> (f32, f32) {
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;

    for &val in arr.iter() {
        min_val = min_val.min(val);
        max_val = max_val.max(val);
    }

    (min_val, max_val)
}

/// Simplified Retinex with dynamic adaptation
#[allow(dead_code)]
pub fn adaptive_retinex(img: &DynamicImage, adaptationlevel: f32) -> Result<DynamicImage> {
    // Use different scales based on image size
    let (width, height) = img.dimensions();
    let image_size = ((width * height) as f32).sqrt();

    let sigmas = vec![
        image_size * 0.01 * adaptationlevel,
        image_size * 0.05 * adaptationlevel,
        image_size * 0.1 * adaptationlevel,
    ];

    multi_scale_retinex(img, &sigmas, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_scale_retinex() {
        let img = DynamicImage::new_rgb8(20, 20);
        let result = single_scale_retinex(&img, 5.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_scale_retinex() {
        let img = DynamicImage::new_rgb8(20, 20);
        let sigmas = vec![5.0, 10.0, 15.0];
        let result = multi_scale_retinex(&img, &sigmas, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_msrcr() {
        let img = DynamicImage::new_rgb8(20, 20);
        let sigmas = vec![5.0, 10.0, 15.0];
        let result = msrcr(&img, &sigmas, 125.0, 46.0, 1.0, 0.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gaussian_kernel() {
        let kernel = create_gaussian_kernel(5, 1.0);
        assert_eq!(kernel.len(), 5);
        assert!((kernel.iter().sum::<f32>() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_retinex() {
        let img = DynamicImage::new_rgb8(50, 50);
        let result = adaptive_retinex(&img, 1.0);
        assert!(result.is_ok());
    }
}
