//! Integral image (summed area table) implementation
//!
//! Integral images allow fast computation of rectangular region sums
//! in constant time, which is useful for many vision algorithms.

use crate::error::Result;
use image::{DynamicImage, GrayImage};
use ndarray::Array2;

/// Compute integral image from a grayscale image
///
/// The integral image at position (x, y) contains the sum of all pixels
/// in the rectangle from (0, 0) to (x, y) inclusive.
///
/// # Arguments
///
/// * `img` - Input grayscale image
///
/// # Returns
///
/// * Result containing integral image as Array2<u64>
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::compute_integral_image;
/// use image::DynamicImage;
///
/// let img = DynamicImage::new_luma8(10, 10);
/// let integral = compute_integral_image(&img).unwrap();
/// assert_eq!(integral.dim(), (10, 10));
/// ```
pub fn compute_integral_image(img: &DynamicImage) -> Result<Array2<u64>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let mut integral = Array2::zeros((height as usize, width as usize));

    // First pixel
    integral[[0, 0]] = gray.get_pixel(0, 0)[0] as u64;

    // First row
    for x in 1..width as usize {
        let pixel_val = gray.get_pixel(x as u32, 0)[0] as u64;
        integral[[0, x]] = integral[[0, x - 1]] + pixel_val;
    }

    // First column
    for y in 1..height as usize {
        let pixel_val = gray.get_pixel(0, y as u32)[0] as u64;
        integral[[y, 0]] = integral[[y - 1, 0]] + pixel_val;
    }

    // Rest of the image
    for y in 1..height as usize {
        for x in 1..width as usize {
            let pixel_val = gray.get_pixel(x as u32, y as u32)[0] as u64;
            integral[[y, x]] =
                pixel_val + integral[[y - 1, x]] + integral[[y, x - 1]] - integral[[y - 1, x - 1]];
        }
    }

    Ok(integral)
}

/// Compute squared integral image
///
/// Similar to integral image but stores sum of squared pixel values
///
/// # Arguments
///
/// * `img` - Input grayscale image
///
/// # Returns
///
/// * Result containing squared integral image as Array2<u64>
pub fn compute_squared_integral_image(img: &DynamicImage) -> Result<Array2<u64>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let mut integral = Array2::zeros((height as usize, width as usize));

    // First pixel
    let val = gray.get_pixel(0, 0)[0] as u64;
    integral[[0, 0]] = val * val;

    // First row
    for x in 1..width as usize {
        let val = gray.get_pixel(x as u32, 0)[0] as u64;
        integral[[0, x]] = integral[[0, x - 1]] + val * val;
    }

    // First column
    for y in 1..height as usize {
        let val = gray.get_pixel(0, y as u32)[0] as u64;
        integral[[y, 0]] = integral[[y - 1, 0]] + val * val;
    }

    // Rest of the image
    for y in 1..height as usize {
        for x in 1..width as usize {
            let val = gray.get_pixel(x as u32, y as u32)[0] as u64;
            integral[[y, x]] =
                val * val + integral[[y - 1, x]] + integral[[y, x - 1]] - integral[[y - 1, x - 1]];
        }
    }

    Ok(integral)
}

/// Compute tilted integral image (45-degree rotated)
///
/// # Arguments
///
/// * `img` - Input grayscale image
///
/// # Returns
///
/// * Result containing tilted integral image as Array2<u64>
pub fn compute_tilted_integral_image(img: &DynamicImage) -> Result<Array2<u64>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let mut integral = Array2::zeros((height as usize, width as usize));

    // Process diagonally
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel_val = gray.get_pixel(x as u32, y as u32)[0] as u64;

            let mut sum = pixel_val;

            // Add contributions from previous positions
            if x > 0 && y > 0 {
                sum += integral[[y - 1, x - 1]];
            }
            if x > 0 && y + 1 < height as usize {
                sum += integral[[y, x - 1]];
                if y > 0 {
                    sum -= integral[[y - 1, x - 1]];
                }
            }
            if x + 1 < width as usize && y > 0 {
                sum += integral[[y - 1, x]];
            }

            integral[[y, x]] = sum;
        }
    }

    Ok(integral)
}

/// Compute the sum of pixel values in a rectangular region using integral image
///
/// # Arguments
///
/// * `integral` - Precomputed integral image
/// * `x1`, `y1` - Top-left corner of rectangle (inclusive)
/// * `x2`, `y2` - Bottom-right corner of rectangle (inclusive)
///
/// # Returns
///
/// * Sum of pixel values in the rectangle
pub fn compute_rect_sum(integral: &Array2<u64>, x1: usize, y1: usize, x2: usize, y2: usize) -> u64 {
    let (height, width) = integral.dim();

    // Clamp coordinates
    let x1 = x1.min(width - 1);
    let y1 = y1.min(height - 1);
    let x2 = x2.min(width - 1);
    let y2 = y2.min(height - 1);

    // Handle edge cases
    if x1 > x2 || y1 > y2 {
        return 0;
    }

    let mut sum = integral[[y2, x2]];

    if x1 > 0 {
        sum -= integral[[y2, x1 - 1]];
    }

    if y1 > 0 {
        sum -= integral[[y1 - 1, x2]];
    }

    if x1 > 0 && y1 > 0 {
        sum += integral[[y1 - 1, x1 - 1]];
    }

    sum
}

/// Compute mean and variance in a rectangular region using integral images
///
/// # Arguments
///
/// * `integral` - Precomputed integral image
/// * `squared_integral` - Precomputed squared integral image
/// * `x1`, `y1` - Top-left corner of rectangle
/// * `x2`, `y2` - Bottom-right corner of rectangle
///
/// # Returns
///
/// * Tuple of (mean, variance)
pub fn compute_rect_mean_variance(
    integral: &Array2<u64>,
    squared_integral: &Array2<u64>,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
) -> (f64, f64) {
    let sum = compute_rect_sum(integral, x1, y1, x2, y2) as f64;
    let squared_sum = compute_rect_sum(squared_integral, x1, y1, x2, y2) as f64;

    let area = ((x2 - x1 + 1) * (y2 - y1 + 1)) as f64;

    if area == 0.0 {
        return (0.0, 0.0);
    }

    let mean = sum / area;
    let variance = (squared_sum / area) - (mean * mean);

    (mean, variance.max(0.0))
}

/// Compute integral image for color images (per channel)
///
/// # Arguments
///
/// * `img` - Input color image
///
/// # Returns
///
/// * Result containing tuple of (red, green, blue) integral images
pub fn compute_color_integral_image(
    img: &DynamicImage,
) -> Result<(Array2<u64>, Array2<u64>, Array2<u64>)> {
    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();

    let mut integral_r = Array2::zeros((height as usize, width as usize));
    let mut integral_g = Array2::zeros((height as usize, width as usize));
    let mut integral_b = Array2::zeros((height as usize, width as usize));

    // Process each channel
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let r = pixel[0] as u64;
            let g = pixel[1] as u64;
            let b = pixel[2] as u64;

            if x == 0 && y == 0 {
                integral_r[[y, x]] = r;
                integral_g[[y, x]] = g;
                integral_b[[y, x]] = b;
            } else if x == 0 {
                integral_r[[y, x]] = r + integral_r[[y - 1, x]];
                integral_g[[y, x]] = g + integral_g[[y - 1, x]];
                integral_b[[y, x]] = b + integral_b[[y - 1, x]];
            } else if y == 0 {
                integral_r[[y, x]] = r + integral_r[[y, x - 1]];
                integral_g[[y, x]] = g + integral_g[[y, x - 1]];
                integral_b[[y, x]] = b + integral_b[[y, x - 1]];
            } else {
                integral_r[[y, x]] = r + integral_r[[y - 1, x]] + integral_r[[y, x - 1]]
                    - integral_r[[y - 1, x - 1]];
                integral_g[[y, x]] = g + integral_g[[y - 1, x]] + integral_g[[y, x - 1]]
                    - integral_g[[y - 1, x - 1]];
                integral_b[[y, x]] = b + integral_b[[y - 1, x]] + integral_b[[y, x - 1]]
                    - integral_b[[y - 1, x - 1]];
            }
        }
    }

    Ok((integral_r, integral_g, integral_b))
}

/// Convert integral image to visualization image
///
/// Normalizes integral image values for visualization
pub fn integral_to_image(integral: &Array2<u64>) -> GrayImage {
    let (height, width) = integral.dim();
    let mut img = GrayImage::new(width as u32, height as u32);

    // Find max value for normalization
    let max_val = integral.iter().max().copied().unwrap_or(1) as f64;

    for y in 0..height {
        for x in 0..width {
            let normalized = (integral[[y, x]] as f64 / max_val * 255.0) as u8;
            img.put_pixel(x as u32, y as u32, image::Luma([normalized]));
        }
    }

    img
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integral_image_basic() {
        let img = DynamicImage::new_luma8(3, 3);
        let result = compute_integral_image(&img);
        assert!(result.is_ok());

        let integral = result.unwrap();
        assert_eq!(integral.dim(), (3, 3));
    }

    #[test]
    fn test_rect_sum() {
        let mut img = GrayImage::new(4, 4);
        // Set all pixels to 1
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, image::Luma([1]));
            }
        }

        let integral = compute_integral_image(&DynamicImage::ImageLuma8(img)).unwrap();

        // Test various rectangles
        assert_eq!(compute_rect_sum(&integral, 0, 0, 1, 1), 4); // 2x2 square
        assert_eq!(compute_rect_sum(&integral, 0, 0, 3, 3), 16); // 4x4 square
        assert_eq!(compute_rect_sum(&integral, 1, 1, 2, 2), 4); // 2x2 square offset
    }

    #[test]
    fn test_mean_variance() {
        let mut img = GrayImage::new(4, 4);
        // Create pattern with known mean/variance
        for y in 0..4 {
            for x in 0..4 {
                img.put_pixel(x, y, image::Luma([(x + y) as u8]));
            }
        }

        let integral = compute_integral_image(&DynamicImage::ImageLuma8(img.clone())).unwrap();
        let squared = compute_squared_integral_image(&DynamicImage::ImageLuma8(img)).unwrap();

        let (mean, variance) = compute_rect_mean_variance(&integral, &squared, 0, 0, 3, 3);
        assert!(mean > 0.0);
        assert!(variance > 0.0);
    }

    #[test]
    fn test_color_integral() {
        let img = DynamicImage::new_rgb8(10, 10);
        let result = compute_color_integral_image(&img);
        assert!(result.is_ok());

        let (r, g, b) = result.unwrap();
        assert_eq!(r.dim(), (10, 10));
        assert_eq!(g.dim(), (10, 10));
        assert_eq!(b.dim(), (10, 10));
    }
}
