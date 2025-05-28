//! Color transformation module
//!
//! This module provides functionality for working with different color spaces
//! and performing color transformations.

pub mod octree_quantization;
pub mod quantization;

use crate::error::Result;
use image::{DynamicImage, ImageBuffer, Rgb};
// Note: Array2 might be needed in future implementations

pub use octree_quantization::{adaptive_octree_quantize, extract_palette, octree_quantize};
pub use quantization::{kmeans_quantize, median_cut_quantize, InitMethod, KMeansParams};

/// Represents a color space
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    /// RGB color space
    RGB,
    /// HSV (Hue, Saturation, Value) color space
    HSV,
    /// LAB color space (CIE L*a*b*)
    LAB,
    /// Grayscale
    Gray,
}

/// Convert an image from RGB to HSV
///
/// # Arguments
///
/// * `img` - Input RGB image
///
/// # Returns
///
/// * Result containing an HSV image
pub fn rgb_to_hsv(img: &DynamicImage) -> Result<DynamicImage> {
    // Ensure input is RGB
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    // Create output buffer
    let mut hsv_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);
            let r = rgb[0] as f32 / 255.0;
            let g = rgb[1] as f32 / 255.0;
            let b = rgb[2] as f32 / 255.0;

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            // Hue calculation
            let h = if delta == 0.0 {
                0.0
            } else if max == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if max == g {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            // Normalize hue to [0, 360)
            let h = if h < 0.0 { h + 360.0 } else { h };

            // Saturation calculation
            let s = if max == 0.0 { 0.0 } else { delta / max };

            // Value calculation
            let v = max;

            // Store HSV as RGB values for visualization
            // Hue [0, 360) -> [0, 255]
            // Saturation [0, 1] -> [0, 255]
            // Value [0, 1] -> [0, 255]
            hsv_img.put_pixel(
                x,
                y,
                Rgb([
                    (h / 360.0 * 255.0) as u8,
                    (s * 255.0) as u8,
                    (v * 255.0) as u8,
                ]),
            );
        }
    }

    Ok(DynamicImage::ImageRgb8(hsv_img))
}

/// Convert an image from HSV to RGB
///
/// # Arguments
///
/// * `img` - Input HSV image (represented as RGB buffer where channels are H, S, V)
///
/// # Returns
///
/// * Result containing an RGB image
pub fn hsv_to_rgb(hsv_img: &DynamicImage) -> Result<DynamicImage> {
    let hsv = hsv_img.to_rgb8();
    let (width, height) = hsv.dimensions();

    let mut rgb_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let hsv_pixel = hsv.get_pixel(x, y);

            // Convert back to HSV range
            let h = hsv_pixel[0] as f32 / 255.0 * 360.0;
            let s = hsv_pixel[1] as f32 / 255.0;
            let v = hsv_pixel[2] as f32 / 255.0;

            // HSV to RGB conversion
            let c = v * s;
            let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
            let m = v - c;

            let (r1, g1, b1) = if h < 60.0 {
                (c, x, 0.0)
            } else if h < 120.0 {
                (x, c, 0.0)
            } else if h < 180.0 {
                (0.0, c, x)
            } else if h < 240.0 {
                (0.0, x, c)
            } else if h < 300.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };

            let r = ((r1 + m) * 255.0) as u8;
            let g = ((g1 + m) * 255.0) as u8;
            let b = ((b1 + m) * 255.0) as u8;

            #[allow(clippy::unnecessary_cast)]
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(rgb_img))
}

/// Convert RGB to grayscale using weighted average
///
/// # Arguments
///
/// * `img` - Input RGB image
/// * `weights` - Optional RGB weights (default: [0.2989, 0.5870, 0.1140] - standard luminance)
///
/// # Returns
///
/// * Result containing a grayscale image
pub fn rgb_to_grayscale(img: &DynamicImage, weights: Option<[f32; 3]>) -> Result<DynamicImage> {
    // Default weights based on human perception of color
    let weights = weights.unwrap_or([0.2989, 0.5870, 0.1140]);

    // Get RGB image
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    // Create grayscale image
    let mut gray_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);

            // Apply weighted average
            let gray_value = (weights[0] * rgb[0] as f32
                + weights[1] * rgb[1] as f32
                + weights[2] * rgb[2] as f32)
                .clamp(0.0, 255.0) as u8;

            gray_img.put_pixel(x, y, image::Luma([gray_value]));
        }
    }

    Ok(DynamicImage::ImageLuma8(gray_img))
}

/// Convert RGB to LAB color space
///
/// # Arguments
///
/// * `img` - Input RGB image
///
/// # Returns
///
/// * Result containing a LAB image (represented as RGB buffer where channels are L, a, b)
pub fn rgb_to_lab(img: &DynamicImage) -> Result<DynamicImage> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let mut lab_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);

            // Convert RGB to XYZ
            let r = rgb[0] as f32 / 255.0;
            let g = rgb[1] as f32 / 255.0;
            let b = rgb[2] as f32 / 255.0;

            // Gamma correction (sRGB to linear RGB)
            let r_lin = if r > 0.04045 {
                ((r + 0.055) / 1.055).powf(2.4)
            } else {
                r / 12.92
            };
            let g_lin = if g > 0.04045 {
                ((g + 0.055) / 1.055).powf(2.4)
            } else {
                g / 12.92
            };
            let b_lin = if b > 0.04045 {
                ((b + 0.055) / 1.055).powf(2.4)
            } else {
                b / 12.92
            };

            // RGB to XYZ conversion (using sRGB D65 matrix)
            let x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375;
            let y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750;
            let z = r_lin * 0.0193339 + g_lin * 0.119192 + b_lin * 0.9503041;

            // XYZ to LAB
            // Reference white point (D65)
            let x_n = 0.95047;
            let y_n = 1.0;
            let z_n = 1.08883;

            // Scale XYZ values relative to reference white
            let x_r = x / x_n;
            let y_r = y / y_n;
            let z_r = z / z_n;

            // XYZ to LAB helper function
            let f = |t: f32| -> f32 {
                if t > 0.008856 {
                    t.powf(1.0 / 3.0)
                } else {
                    (7.787 * t) + (16.0 / 116.0)
                }
            };

            let fx = f(x_r);
            let fy = f(y_r);
            let fz = f(z_r);

            // Calculate LAB values
            let l = (116.0 * fy) - 16.0;
            let a = 500.0 * (fx - fy);
            let b_val = 200.0 * (fy - fz);

            // Scale to fit in 8-bit channels
            // L: [0, 100] -> [0, 255]
            // a: [-128, 127] -> [0, 255]
            // b: [-128, 127] -> [0, 255]
            let l_scaled = (l * 2.55).clamp(0.0, 255.0) as u8;
            let a_scaled = ((a + 128.0).clamp(0.0, 255.0)) as u8;
            let b_scaled = ((b_val + 128.0).clamp(0.0, 255.0)) as u8;

            lab_img.put_pixel(x as u32, y as u32, Rgb([l_scaled, a_scaled, b_scaled]));
        }
    }

    Ok(DynamicImage::ImageRgb8(lab_img))
}

/// Convert LAB to RGB color space
///
/// # Arguments
///
/// * `img` - Input LAB image (represented as RGB buffer where channels are L, a, b)
///
/// # Returns
///
/// * Result containing an RGB image
pub fn lab_to_rgb(lab_img: &DynamicImage) -> Result<DynamicImage> {
    let lab = lab_img.to_rgb8();
    let (width, height) = lab.dimensions();

    let mut rgb_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let lab_pixel = lab.get_pixel(x, y);

            // Scale back from 8-bit to LAB range
            let l = lab_pixel[0] as f32 / 2.55; // [0, 255] -> [0, 100]
            let a = lab_pixel[1] as f32 - 128.0; // [0, 255] -> [-128, 127]
            let b = lab_pixel[2] as f32 - 128.0; // [0, 255] -> [-128, 127]

            // LAB to XYZ
            let fy = (l + 16.0) / 116.0;
            let fx = a / 500.0 + fy;
            let fz = fy - b / 200.0;

            // Reference white point (D65)
            let x_n = 0.95047;
            let y_n = 1.0;
            let z_n = 1.08883;

            // LAB to XYZ helper function
            let f = |t: f32| -> f32 {
                if t > 0.206893 {
                    t.powi(3)
                } else {
                    (t - 16.0 / 116.0) / 7.787
                }
            };

            let x = x_n * f(fx);
            let y = y_n * f(fy);
            let z = z_n * f(fz);

            // XYZ to linear RGB (using inverse sRGB D65 matrix)
            let r_lin = x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
            let g_lin = -x * 0.969266 + y * 1.8760108 + z * 0.0415560;
            let b_lin = x * 0.0556434 - y * 0.2040259 + z * 1.0572252;

            // Linear RGB to sRGB
            let r = if r_lin > 0.0031308 {
                1.055 * r_lin.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * r_lin
            };

            let g = if g_lin > 0.0031308 {
                1.055 * g_lin.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * g_lin
            };

            let b = if b_lin > 0.0031308 {
                1.055 * b_lin.powf(1.0 / 2.4) - 0.055
            } else {
                12.92 * b_lin
            };

            // Convert to 8-bit and clamp to valid range
            let r = (r * 255.0).clamp(0.0, 255.0) as u8;
            let g = (g * 255.0).clamp(0.0, 255.0) as u8;
            let b = (b * 255.0).clamp(0.0, 255.0) as u8;

            #[allow(clippy::unnecessary_cast)]
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(rgb_img))
}

/// Split an RGB image into separate channels
///
/// # Arguments
///
/// * `img` - Input RGB image
///
/// # Returns
///
/// * Result containing a tuple of grayscale images (r, g, b)
pub fn split_channels(img: &DynamicImage) -> Result<(DynamicImage, DynamicImage, DynamicImage)> {
    let rgb_img = img.to_rgb8();
    let (width, height) = rgb_img.dimensions();

    let mut r_channel = ImageBuffer::new(width, height);
    let mut g_channel = ImageBuffer::new(width, height);
    let mut b_channel = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let rgb = rgb_img.get_pixel(x, y);

            r_channel.put_pixel(x, y, image::Luma([rgb[0]]));
            g_channel.put_pixel(x, y, image::Luma([rgb[1]]));
            b_channel.put_pixel(x, y, image::Luma([rgb[2]]));
        }
    }

    Ok((
        DynamicImage::ImageLuma8(r_channel),
        DynamicImage::ImageLuma8(g_channel),
        DynamicImage::ImageLuma8(b_channel),
    ))
}

/// Merge separate channels into an RGB image
///
/// # Arguments
///
/// * `r_channel` - Red channel image
/// * `g_channel` - Green channel image
/// * `b_channel` - Blue channel image
///
/// # Returns
///
/// * Result containing an RGB image
pub fn merge_channels(
    r_channel: &DynamicImage,
    g_channel: &DynamicImage,
    b_channel: &DynamicImage,
) -> Result<DynamicImage> {
    let r_img = r_channel.to_luma8();
    let g_img = g_channel.to_luma8();
    let b_img = b_channel.to_luma8();

    let (width, height) = r_img.dimensions();

    // Check dimensions
    if g_img.dimensions() != (width, height) || b_img.dimensions() != (width, height) {
        return Err(crate::error::VisionError::InvalidParameter(
            "Channel dimensions do not match".to_string(),
        ));
    }

    let mut rgb_img = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = r_img.get_pixel(x, y)[0];
            let g = g_img.get_pixel(x, y)[0];
            let b = b_img.get_pixel(x, y)[0];

            #[allow(clippy::unnecessary_cast)]
            rgb_img.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
        }
    }

    Ok(DynamicImage::ImageRgb8(rgb_img))
}
