//! Gamma correction for contrast enhancement

use crate::error::{Result, VisionError};
use image::{DynamicImage, GrayImage, Luma};

/// Apply gamma correction to an image
///
/// Gamma correction is a nonlinear operation used to adjust the brightness
/// and contrast of an image. It's particularly useful for correcting images
/// that appear too dark or too bright on different display devices.
///
/// # Arguments
///
/// * `img` - Input image
/// * `gamma` - Gamma value (< 1.0 brightens, > 1.0 darkens, 1.0 is identity)
///
/// # Returns
///
/// * Result containing the gamma-corrected image
///
/// # Example
///
/// ```
/// use scirs2_vision::preprocessing::gamma_correction;
/// use image::open;
///
/// let img = open("examples/input/input.jpg").unwrap();
/// let corrected = gamma_correction(&img, 2.2).unwrap();
/// ```
pub fn gamma_correction(img: &DynamicImage, gamma: f32) -> Result<DynamicImage> {
    if gamma <= 0.0 {
        return Err(VisionError::InvalidParameter(
            "Gamma must be positive".to_string(),
        ));
    }

    // Build lookup table for efficiency
    let lut: Vec<u8> = (0..256)
        .map(|i| {
            let normalized = i as f32 / 255.0;
            let corrected = normalized.powf(gamma);
            (corrected * 255.0).clamp(0.0, 255.0) as u8
        })
        .collect();

    match img {
        DynamicImage::ImageLuma8(gray) => {
            let mut result = GrayImage::new(gray.width(), gray.height());
            for (x, y, pixel) in gray.enumerate_pixels() {
                result.put_pixel(x, y, Luma([lut[pixel[0] as usize]]));
            }
            Ok(DynamicImage::ImageLuma8(result))
        }
        _ => {
            // For color images, apply gamma to each channel
            let rgb = img.to_rgb8();
            let mut result = image::RgbImage::new(rgb.width(), rgb.height());

            for (x, y, pixel) in rgb.enumerate_pixels() {
                result.put_pixel(
                    x,
                    y,
                    image::Rgb([
                        lut[pixel[0] as usize],
                        lut[pixel[1] as usize],
                        lut[pixel[2] as usize],
                    ]),
                );
            }
            Ok(DynamicImage::ImageRgb8(result))
        }
    }
}

/// Apply automatic gamma correction using image statistics
///
/// Automatically determines an appropriate gamma value based on the
/// mean brightness of the image.
///
/// # Arguments
///
/// * `img` - Input image
/// * `target_brightness` - Target mean brightness (0.0-1.0, typically 0.5)
///
/// # Returns
///
/// * Result containing the auto gamma-corrected image
pub fn auto_gamma_correction(img: &DynamicImage, target_brightness: f32) -> Result<DynamicImage> {
    if target_brightness <= 0.0 || target_brightness >= 1.0 {
        return Err(VisionError::InvalidParameter(
            "Target brightness must be between 0.0 and 1.0".to_string(),
        ));
    }

    // Calculate current mean brightness
    let gray = img.to_luma8();
    let mut sum = 0u64;
    for pixel in gray.pixels() {
        sum += pixel[0] as u64;
    }

    let pixel_count = (gray.width() * gray.height()) as f32;
    let mean_brightness = (sum as f32) / (pixel_count * 255.0);

    // Avoid division by zero or log of zero
    if mean_brightness < 0.001 {
        return gamma_correction(img, 0.5); // Apply strong brightening
    }

    // Calculate gamma to achieve target brightness
    // Using: target = current^(1/gamma)
    // Therefore: gamma = log(current) / log(target)
    let gamma = mean_brightness.ln() / target_brightness.ln();

    // Clamp gamma to reasonable range
    let gamma = gamma.clamp(0.1, 10.0);

    gamma_correction(img, gamma)
}

/// Apply adaptive gamma correction
///
/// Applies different gamma values to different regions of the image
/// based on local statistics.
///
/// # Arguments
///
/// * `img` - Input image
/// * `window_size` - Size of local window for statistics
/// * `gamma_range` - Range of gamma values to use (min, max)
///
/// # Returns
///
/// * Result containing the adaptively gamma-corrected image
pub fn adaptive_gamma_correction(
    img: &DynamicImage,
    window_size: u32,
    gamma_range: (f32, f32),
) -> Result<DynamicImage> {
    let (min_gamma, max_gamma) = gamma_range;

    if min_gamma <= 0.0 || max_gamma <= 0.0 || min_gamma > max_gamma {
        return Err(VisionError::InvalidParameter(
            "Invalid gamma range".to_string(),
        ));
    }

    if window_size == 0 {
        return Err(VisionError::InvalidParameter(
            "Window size must be positive".to_string(),
        ));
    }

    let gray = img.to_luma8();
    let (width, height) = (gray.width(), gray.height());
    let mut result = GrayImage::new(width, height);

    let half_window = window_size / 2;

    for y in 0..height {
        for x in 0..width {
            // Define local window boundaries
            let x_start = x.saturating_sub(half_window);
            let x_end = (x + half_window + 1).min(width);
            let y_start = y.saturating_sub(half_window);
            let y_end = (y + half_window + 1).min(height);

            // Calculate local statistics
            let mut sum = 0u32;
            let mut count = 0u32;

            for ly in y_start..y_end {
                for lx in x_start..x_end {
                    sum += gray.get_pixel(lx, ly)[0] as u32;
                    count += 1;
                }
            }

            let local_mean = (sum as f32) / (count as f32 * 255.0);

            // Map local mean to gamma value
            // Dark regions get lower gamma (brightening), bright regions get higher gamma
            let gamma = min_gamma + (max_gamma - min_gamma) * local_mean;

            // Apply gamma correction to current pixel
            let pixel_value = gray.get_pixel(x, y)[0];
            let normalized = pixel_value as f32 / 255.0;
            let corrected = normalized.powf(gamma);
            let output_value = (corrected * 255.0).clamp(0.0, 255.0) as u8;

            result.put_pixel(x, y, Luma([output_value]));
        }
    }

    Ok(DynamicImage::ImageLuma8(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn test_gamma_correction_identity() {
        let img = DynamicImage::new_luma8(10, 10);
        let result = gamma_correction(&img, 1.0).unwrap();

        // Gamma = 1.0 should not change the image
        match (img, result) {
            (DynamicImage::ImageLuma8(orig), DynamicImage::ImageLuma8(res)) => {
                for (o, r) in orig.pixels().zip(res.pixels()) {
                    assert_eq!(o[0], r[0]);
                }
            }
            _ => panic!("Unexpected image types"),
        }
    }

    #[test]
    fn test_gamma_correction_invalid() {
        let img = DynamicImage::new_luma8(10, 10);

        // Test invalid gamma values
        assert!(gamma_correction(&img, 0.0).is_err());
        assert!(gamma_correction(&img, -1.0).is_err());
    }

    #[test]
    fn test_gamma_correction_brightening() {
        // Create a dark image
        let mut img = GrayImage::new(10, 10);
        for pixel in img.pixels_mut() {
            *pixel = Luma([50]); // Dark gray
        }
        let dynamic_img = DynamicImage::ImageLuma8(img);

        // Apply brightening gamma
        let result = gamma_correction(&dynamic_img, 0.5).unwrap();

        // Check that image got brighter
        match result {
            DynamicImage::ImageLuma8(res) => {
                let first_pixel = res.get_pixel(0, 0)[0];
                // With gamma < 1.0, the image should get brighter
                // 50/255 ≈ 0.196, raised to power 0.5 ≈ 0.443, * 255 ≈ 113
                assert!(first_pixel > 50);
            }
            _ => panic!("Unexpected image type"),
        }
    }

    #[test]
    fn test_auto_gamma_correction() {
        // Create a dark image
        let mut img = GrayImage::new(10, 10);
        for pixel in img.pixels_mut() {
            *pixel = Luma([50]);
        }
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result = auto_gamma_correction(&dynamic_img, 0.5).unwrap();
        assert!(result.as_luma8().is_some());
    }

    #[test]
    fn test_auto_gamma_invalid_target() {
        let img = DynamicImage::new_luma8(10, 10);

        assert!(auto_gamma_correction(&img, 0.0).is_err());
        assert!(auto_gamma_correction(&img, 1.0).is_err());
        assert!(auto_gamma_correction(&img, -0.5).is_err());
        assert!(auto_gamma_correction(&img, 1.5).is_err());
    }

    #[test]
    fn test_adaptive_gamma_correction() {
        let img = DynamicImage::new_luma8(20, 20);
        let result = adaptive_gamma_correction(&img, 5, (0.5, 2.0)).unwrap();

        assert!(result.as_luma8().is_some());
        assert_eq!(result.width(), 20);
        assert_eq!(result.height(), 20);
    }

    #[test]
    fn test_adaptive_gamma_invalid_params() {
        let img = DynamicImage::new_luma8(10, 10);

        assert!(adaptive_gamma_correction(&img, 0, (0.5, 2.0)).is_err());
        assert!(adaptive_gamma_correction(&img, 5, (0.0, 2.0)).is_err());
        assert!(adaptive_gamma_correction(&img, 5, (2.0, 1.0)).is_err());
    }
}
