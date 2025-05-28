//! Bilateral filter for edge-preserving smoothing
//!
//! The bilateral filter is a non-linear, edge-preserving smoothing filter
//! that considers both spatial distance and intensity difference.

use crate::error::Result;
#[allow(unused_imports)] // GenericImageView is used for the dimensions() method
use image::{DynamicImage, GenericImageView, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use rayon::prelude::*;
use std::sync::Arc;

/// Parameters for bilateral filtering
#[derive(Debug, Clone)]
pub struct BilateralParams {
    /// Spatial sigma (standard deviation for spatial kernel)
    pub sigma_spatial: f32,
    /// Range sigma (standard deviation for intensity kernel)
    pub sigma_range: f32,
    /// Filter window radius
    pub radius: usize,
}

impl Default for BilateralParams {
    fn default() -> Self {
        Self {
            sigma_spatial: 3.0,
            sigma_range: 30.0,
            radius: 5,
        }
    }
}

/// Apply bilateral filter to a grayscale image
///
/// # Arguments
///
/// * `img` - Input image
/// * `params` - Filter parameters
///
/// # Returns
///
/// * Result containing filtered image
///
/// # Example
///
/// ```rust
/// use scirs2_vision::preprocessing::{bilateral_filter_advanced, BilateralParams};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let filtered = bilateral_filter_advanced(&img, &BilateralParams::default())?;
/// # Ok(())
/// # }
/// ```
pub fn bilateral_filter_advanced(
    img: &DynamicImage,
    params: &BilateralParams,
) -> Result<DynamicImage> {
    match img {
        DynamicImage::ImageLuma8(_) => {
            let gray = img.to_luma8();
            let filtered = bilateral_filter_gray(&gray, params)?;
            Ok(DynamicImage::ImageLuma8(filtered))
        }
        _ => {
            let rgb = img.to_rgb8();
            let filtered = bilateral_filter_rgb(&rgb, params)?;
            Ok(DynamicImage::ImageRgb8(filtered))
        }
    }
}

/// Apply bilateral filter to grayscale image
fn bilateral_filter_gray(img: &GrayImage, params: &BilateralParams) -> Result<GrayImage> {
    let (width, height) = img.dimensions();
    let radius = params.radius;

    // Precompute spatial weights
    let spatial_weights = Arc::new(compute_spatial_weights(radius, params.sigma_spatial));
    let sigma_range = params.sigma_range;

    // Process pixels in parallel
    let pixels: Vec<_> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let spatial_weights = spatial_weights.clone();
            (0..width)
                .into_par_iter()
                .map(move |x| {
                    let filtered_value = apply_bilateral_pixel_gray(
                        img,
                        x,
                        y,
                        radius,
                        &spatial_weights,
                        sigma_range,
                    );
                    (x, y, filtered_value)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Create output image
    let mut result = ImageBuffer::new(width, height);
    for (x, y, value) in pixels {
        result.put_pixel(x, y, Luma([value]));
    }

    Ok(result)
}

/// Apply bilateral filter to RGB image
fn bilateral_filter_rgb(img: &RgbImage, params: &BilateralParams) -> Result<RgbImage> {
    let (width, height) = img.dimensions();
    let radius = params.radius;

    // Precompute spatial weights
    let spatial_weights = Arc::new(compute_spatial_weights(radius, params.sigma_spatial));
    let sigma_range = params.sigma_range;

    // Process pixels in parallel
    let pixels: Vec<_> = (0..height)
        .into_par_iter()
        .flat_map(|y| {
            let spatial_weights = spatial_weights.clone();
            (0..width)
                .into_par_iter()
                .map(move |x| {
                    let filtered_rgb =
                        apply_bilateral_pixel_rgb(img, x, y, radius, &spatial_weights, sigma_range);
                    (x, y, filtered_rgb)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Create output image
    let mut result = ImageBuffer::new(width, height);
    for (x, y, rgb) in pixels {
        result.put_pixel(x, y, Rgb(rgb));
    }

    Ok(result)
}

/// Compute spatial weights for the filter kernel
fn compute_spatial_weights(radius: usize, sigma: f32) -> Vec<Vec<f32>> {
    let size = 2 * radius + 1;
    let mut weights = vec![vec![0.0; size]; size];
    let sigma2 = sigma * sigma;

    for (dy, row) in weights.iter_mut().enumerate() {
        for (dx, weight) in row.iter_mut().enumerate() {
            let y = dy as f32 - radius as f32;
            let x = dx as f32 - radius as f32;
            let dist2 = x * x + y * y;
            *weight = (-dist2 / (2.0 * sigma2)).exp();
        }
    }

    weights
}

/// Apply bilateral filter to a single grayscale pixel
fn apply_bilateral_pixel_gray(
    img: &GrayImage,
    cx: u32,
    cy: u32,
    radius: usize,
    spatial_weights: &[Vec<f32>],
    sigma_range: f32,
) -> u8 {
    let (width, height) = img.dimensions();
    let center_value = img.get_pixel(cx, cy)[0] as f32;
    let sigma_range2 = sigma_range * sigma_range;

    let mut weighted_sum = 0.0f32;
    let mut weight_sum = 0.0f32;

    for (dy, spatial_row) in spatial_weights.iter().enumerate() {
        for (dx, &spatial_weight) in spatial_row.iter().enumerate() {
            let x = cx as i32 + dx as i32 - radius as i32;
            let y = cy as i32 + dy as i32 - radius as i32;

            // Check bounds
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let pixel_value = img.get_pixel(x as u32, y as u32)[0] as f32;

                // Compute range weight
                let range_diff = pixel_value - center_value;
                let range_weight = (-(range_diff * range_diff) / (2.0 * sigma_range2)).exp();

                // Combined weight
                let weight = spatial_weight * range_weight;

                weighted_sum += pixel_value * weight;
                weight_sum += weight;
            }
        }
    }

    if weight_sum > 0.0 {
        (weighted_sum / weight_sum).round().clamp(0.0, 255.0) as u8
    } else {
        center_value as u8
    }
}

/// Apply bilateral filter to a single RGB pixel
fn apply_bilateral_pixel_rgb(
    img: &RgbImage,
    cx: u32,
    cy: u32,
    radius: usize,
    spatial_weights: &[Vec<f32>],
    sigma_range: f32,
) -> [u8; 3] {
    let (width, height) = img.dimensions();
    let center_pixel = img.get_pixel(cx, cy);
    let sigma_range2 = sigma_range * sigma_range;

    let mut weighted_sum = [0.0f32; 3];
    let mut weight_sum = 0.0f32;

    for (dy, spatial_row) in spatial_weights.iter().enumerate() {
        for (dx, &spatial_weight) in spatial_row.iter().enumerate() {
            let x = cx as i32 + dx as i32 - radius as i32;
            let y = cy as i32 + dy as i32 - radius as i32;

            // Check bounds
            if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                let pixel = img.get_pixel(x as u32, y as u32);

                // Compute range weight using Euclidean distance in RGB space
                let dr = pixel[0] as f32 - center_pixel[0] as f32;
                let dg = pixel[1] as f32 - center_pixel[1] as f32;
                let db = pixel[2] as f32 - center_pixel[2] as f32;
                let range_dist2 = dr * dr + dg * dg + db * db;
                let range_weight = (-range_dist2 / (2.0 * sigma_range2)).exp();

                // Combined weight
                let weight = spatial_weight * range_weight;

                weighted_sum[0] += pixel[0] as f32 * weight;
                weighted_sum[1] += pixel[1] as f32 * weight;
                weighted_sum[2] += pixel[2] as f32 * weight;
                weight_sum += weight;
            }
        }
    }

    if weight_sum > 0.0 {
        [
            (weighted_sum[0] / weight_sum).round().clamp(0.0, 255.0) as u8,
            (weighted_sum[1] / weight_sum).round().clamp(0.0, 255.0) as u8,
            (weighted_sum[2] / weight_sum).round().clamp(0.0, 255.0) as u8,
        ]
    } else {
        [center_pixel[0], center_pixel[1], center_pixel[2]]
    }
}

/// Fast bilateral filter using integral histograms (simplified version)
pub fn fast_bilateral_filter(img: &DynamicImage, params: &BilateralParams) -> Result<DynamicImage> {
    // For simplicity, fall back to regular bilateral filter
    // A full implementation would use integral histograms for acceleration
    bilateral_filter_advanced(img, params)
}

/// Joint bilateral filter using a guidance image
///
/// # Arguments
///
/// * `img` - Input image to filter
/// * `guide` - Guidance image for edge preservation
/// * `params` - Filter parameters
///
/// # Returns
///
/// * Result containing filtered image
pub fn joint_bilateral_filter(
    img: &DynamicImage,
    guide: &DynamicImage,
    params: &BilateralParams,
) -> Result<DynamicImage> {
    let input_gray = img.to_luma8();
    let guide_gray = guide.to_luma8();
    let (width, height) = input_gray.dimensions();

    let radius = params.radius;
    let spatial_weights = compute_spatial_weights(radius, params.sigma_spatial);
    let sigma_range2 = params.sigma_range * params.sigma_range;

    let mut result = ImageBuffer::new(width, height);

    for cy in 0..height {
        for cx in 0..width {
            let guide_center = guide_gray.get_pixel(cx, cy)[0] as f32;
            let mut weighted_sum = 0.0f32;
            let mut weight_sum = 0.0f32;

            for (dy, spatial_row) in spatial_weights.iter().enumerate() {
                for (dx, &spatial_weight) in spatial_row.iter().enumerate() {
                    let x = cx as i32 + dx as i32 - radius as i32;
                    let y = cy as i32 + dy as i32 - radius as i32;

                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        let input_value = input_gray.get_pixel(x as u32, y as u32)[0] as f32;
                        let guide_value = guide_gray.get_pixel(x as u32, y as u32)[0] as f32;

                        // Range weight based on guide image
                        let range_diff = guide_value - guide_center;
                        let range_weight =
                            (-(range_diff * range_diff) / (2.0 * sigma_range2)).exp();

                        let weight = spatial_weight * range_weight;
                        weighted_sum += input_value * weight;
                        weight_sum += weight;
                    }
                }
            }

            let filtered_value = if weight_sum > 0.0 {
                (weighted_sum / weight_sum).round().clamp(0.0, 255.0) as u8
            } else {
                input_gray.get_pixel(cx, cy)[0]
            };

            result.put_pixel(cx, cy, Luma([filtered_value]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bilateral_filter_gray() {
        let img = DynamicImage::new_luma8(20, 20);
        let params = BilateralParams {
            sigma_spatial: 2.0,
            sigma_range: 20.0,
            radius: 3,
        };

        let result = bilateral_filter_advanced(&img, &params);
        assert!(result.is_ok());

        let filtered = result.unwrap();
        assert_eq!(filtered.dimensions(), (20, 20));
    }

    #[test]
    fn test_bilateral_filter_rgb() {
        let img = DynamicImage::new_rgb8(20, 20);
        let params = BilateralParams::default();

        let result = bilateral_filter_advanced(&img, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_spatial_weights() {
        let weights = compute_spatial_weights(2, 1.0);
        assert_eq!(weights.len(), 5);
        assert_eq!(weights[0].len(), 5);

        // Center weight should be maximum
        assert!(weights[2][2] > weights[0][0]);
        assert!((weights[2][2] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_joint_bilateral() {
        let img = DynamicImage::new_luma8(10, 10);
        let guide = img.clone();
        let params = BilateralParams::default();

        let result = joint_bilateral_filter(&img, &guide, &params);
        assert!(result.is_ok());
    }
}
