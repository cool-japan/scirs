//! Shi-Tomasi corner detector (Good Features to Track)
//!
//! An improvement over the Harris corner detector that uses a simpler
//! corner score calculation based on the minimum eigenvalue of the
//! structure tensor matrix.

use crate::error::{Result, VisionError};
use crate::feature::image_to_array;
use image::{DynamicImage, GrayImage};
use ndarray::Array2;

/// Shi-Tomasi corner detection (Good Features to Track)
///
/// Detects corners using the Shi-Tomasi method, which improves upon Harris
/// corner detection by using the minimum eigenvalue as the corner score.
///
/// # Arguments
///
/// * `img` - Input image
/// * `block_size` - Size of the window for corner detection
/// * `threshold` - Threshold for corner detection
/// * `max_corners` - Maximum number of corners to return (0 for all)
/// * `min_distance` - Minimum distance between corners
///
/// # Returns
///
/// * Result containing corner points
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_vision::feature::shi_tomasi_corners;
/// use image::DynamicImage;
///
/// let img = image::open("input.jpg").unwrap();
/// let corners = shi_tomasi_corners(&img, 3, 0.01, 100, 10)?;
/// ```
pub fn shi_tomasi_corners(
    img: &DynamicImage,
    block_size: usize,
    threshold: f32,
    max_corners: usize,
    min_distance: usize,
) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Check if block_size is valid
    if block_size % 2 == 0 || block_size < 3 {
        return Err(VisionError::InvalidParameter(
            "block_size must be odd and at least 3".to_string(),
        ));
    }

    // Step 1: Calculate gradients
    let mut ix2 = Array2::zeros((height, width));
    let mut iy2 = Array2::zeros((height, width));
    let mut ixy = Array2::zeros((height, width));

    // Calculate gradients using Sobel operators
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Sobel X
            let gx = -1.0 * array[[y - 1, x - 1]]
                + 1.0 * array[[y - 1, x + 1]]
                + -2.0 * array[[y, x - 1]]
                + 2.0 * array[[y, x + 1]]
                + -1.0 * array[[y + 1, x - 1]]
                + 1.0 * array[[y + 1, x + 1]];

            // Sobel Y
            let gy = -1.0 * array[[y - 1, x - 1]]
                + -2.0 * array[[y - 1, x]]
                + -1.0 * array[[y - 1, x + 1]]
                + 1.0 * array[[y + 1, x - 1]]
                + 2.0 * array[[y + 1, x]]
                + 1.0 * array[[y + 1, x + 1]];

            ix2[[y, x]] = gx * gx;
            iy2[[y, x]] = gy * gy;
            ixy[[y, x]] = gx * gy;
        }
    }

    // Step 2: Apply window function (box filter)
    let radius = block_size / 2;
    let mut smoothed_ix2 = Array2::zeros((height, width));
    let mut smoothed_iy2 = Array2::zeros((height, width));
    let mut smoothed_ixy = Array2::zeros((height, width));

    for y in radius..(height - radius) {
        for x in radius..(width - radius) {
            let mut sum_ix2 = 0.0;
            let mut sum_iy2 = 0.0;
            let mut sum_ixy = 0.0;
            let mut count = 0;

            for dy in (y - radius)..=(y + radius) {
                for dx in (x - radius)..=(x + radius) {
                    sum_ix2 += ix2[[dy, dx]];
                    sum_iy2 += iy2[[dy, dx]];
                    sum_ixy += ixy[[dy, dx]];
                    count += 1;
                }
            }

            smoothed_ix2[[y, x]] = sum_ix2 / count as f32;
            smoothed_iy2[[y, x]] = sum_iy2 / count as f32;
            smoothed_ixy[[y, x]] = sum_ixy / count as f32;
        }
    }

    // Step 3: Calculate Shi-Tomasi response (minimum eigenvalue)
    let mut response = Array2::zeros((height, width));

    for y in radius..(height - radius) {
        for x in radius..(width - radius) {
            // Structure tensor matrix:
            // [a b]
            // [b c]
            let a = smoothed_ix2[[y, x]];
            let b = smoothed_ixy[[y, x]];
            let c = smoothed_iy2[[y, x]];

            // Minimum eigenvalue calculation
            // λ = (a + c - sqrt((a - c)² + 4b²)) / 2
            let trace = a + c;
            let _det = a * c - b * b;
            let discriminant = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
            let min_eigenvalue = (trace - discriminant) / 2.0;

            response[[y, x]] = min_eigenvalue;
        }
    }

    // Step 4: Threshold and extract corners
    let mut corners = Vec::new();

    for y in radius..(height - radius) {
        for x in radius..(width - radius) {
            if response[[y, x]] > threshold {
                corners.push((x, y, response[[y, x]]));
            }
        }
    }

    // Sort by response strength
    corners.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Filter by minimum distance
    let mut selected_corners = Vec::new();

    for (x, y, score) in corners {
        let mut too_close = false;

        for &(sx, sy, _) in &selected_corners {
            let dist_sq = ((x as i32 - sx as i32).pow(2) + (y as i32 - sy as i32).pow(2)) as usize;
            if dist_sq < min_distance * min_distance {
                too_close = true;
                break;
            }
        }

        if !too_close {
            selected_corners.push((x, y, score));

            if max_corners > 0 && selected_corners.len() >= max_corners {
                break;
            }
        }
    }

    // Create output image
    let mut output = Array2::zeros((height, width));
    for (x, y, _) in selected_corners {
        output[[y, x]] = 1.0;
    }

    crate::feature::array_to_image(&output)
}

/// Simplified Shi-Tomasi corner detection
///
/// Uses default parameters suitable for most applications.
///
/// # Arguments
///
/// * `img` - Input image
/// * `max_corners` - Maximum number of corners to return
///
/// # Returns
///
/// * Result containing corner points
pub fn shi_tomasi_corners_simple(img: &DynamicImage, max_corners: usize) -> Result<GrayImage> {
    shi_tomasi_corners(img, 3, 0.01, max_corners, 10)
}

/// Extract good features to track with sub-pixel accuracy
///
/// Returns corner coordinates with floating-point precision.
///
/// # Arguments
///
/// * `img` - Input image
/// * `block_size` - Size of the window for corner detection
/// * `threshold` - Threshold for corner detection
/// * `max_corners` - Maximum number of corners to return
/// * `min_distance` - Minimum distance between corners
///
/// # Returns
///
/// * Result containing vector of (x, y, score) tuples
pub fn good_features_to_track(
    img: &DynamicImage,
    block_size: usize,
    threshold: f32,
    max_corners: usize,
    min_distance: usize,
) -> Result<Vec<(f32, f32, f32)>> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Check if block_size is valid
    if block_size % 2 == 0 || block_size < 3 {
        return Err(VisionError::InvalidParameter(
            "block_size must be odd and at least 3".to_string(),
        ));
    }

    // Calculate gradients and response (same as above)
    let mut ix2 = Array2::zeros((height, width));
    let mut iy2 = Array2::zeros((height, width));
    let mut ixy = Array2::zeros((height, width));

    // Calculate gradients
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let gx = -1.0 * array[[y - 1, x - 1]]
                + 1.0 * array[[y - 1, x + 1]]
                + -2.0 * array[[y, x - 1]]
                + 2.0 * array[[y, x + 1]]
                + -1.0 * array[[y + 1, x - 1]]
                + 1.0 * array[[y + 1, x + 1]];

            let gy = -1.0 * array[[y - 1, x - 1]]
                + -2.0 * array[[y - 1, x]]
                + -1.0 * array[[y - 1, x + 1]]
                + 1.0 * array[[y + 1, x - 1]]
                + 2.0 * array[[y + 1, x]]
                + 1.0 * array[[y + 1, x + 1]];

            ix2[[y, x]] = gx * gx;
            iy2[[y, x]] = gy * gy;
            ixy[[y, x]] = gx * gy;
        }
    }

    // Apply window function
    let radius = block_size / 2;
    let mut response = Array2::zeros((height, width));

    for y in radius..(height - radius) {
        for x in radius..(width - radius) {
            let mut sum_ix2 = 0.0;
            let mut sum_iy2 = 0.0;
            let mut sum_ixy = 0.0;
            let mut count = 0;

            for dy in (y - radius)..=(y + radius) {
                for dx in (x - radius)..=(x + radius) {
                    sum_ix2 += ix2[[dy, dx]];
                    sum_iy2 += iy2[[dy, dx]];
                    sum_ixy += ixy[[dy, dx]];
                    count += 1;
                }
            }

            let a = sum_ix2 / count as f32;
            let b = sum_ixy / count as f32;
            let c = sum_iy2 / count as f32;

            // Minimum eigenvalue
            let trace = a + c;
            let discriminant = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
            response[[y, x]] = (trace - discriminant) / 2.0;
        }
    }

    // Extract corners with sub-pixel refinement
    let mut corners = Vec::new();

    for y in (radius + 1)..(height - radius - 1) {
        for x in (radius + 1)..(width - radius - 1) {
            let r = response[[y, x]];

            if r > threshold {
                // Check if local maximum
                let mut is_max = true;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dy == 0 && dx == 0 {
                            continue;
                        }
                        if response[[(y as i32 + dy) as usize, (x as i32 + dx) as usize]] >= r {
                            is_max = false;
                            break;
                        }
                    }
                    if !is_max {
                        break;
                    }
                }

                if is_max {
                    // Sub-pixel refinement using quadratic interpolation
                    let dx = (response[[y, x + 1]] - response[[y, x - 1]]) / 2.0;
                    let dy = (response[[y + 1, x]] - response[[y - 1, x]]) / 2.0;
                    let dxx = response[[y, x + 1]] - 2.0 * r + response[[y, x - 1]];
                    let dyy = response[[y + 1, x]] - 2.0 * r + response[[y - 1, x]];

                    let mut sub_x = x as f32;
                    let mut sub_y = y as f32;

                    if dxx.abs() > 1e-6 {
                        sub_x -= dx / dxx;
                    }
                    if dyy.abs() > 1e-6 {
                        sub_y -= dy / dyy;
                    }

                    corners.push((sub_x, sub_y, r));
                }
            }
        }
    }

    // Sort by response strength
    corners.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    // Filter by minimum distance
    let mut selected_corners = Vec::new();

    for (x, y, score) in corners {
        let mut too_close = false;

        for &(sx, sy, _) in &selected_corners {
            let x_diff: f32 = x - sx;
            let y_diff: f32 = y - sy;
            let dist_sq: f32 = x_diff.powi(2) + y_diff.powi(2);
            if dist_sq < (min_distance as f32 * min_distance as f32) {
                too_close = true;
                break;
            }
        }

        if !too_close {
            selected_corners.push((x, y, score));

            if max_corners > 0 && selected_corners.len() >= max_corners {
                break;
            }
        }
    }

    Ok(selected_corners)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_shi_tomasi_detection() {
        // Create a test image with corners
        let mut img = GrayImage::new(20, 20);

        // Background
        for y in 0..20 {
            for x in 0..20 {
                img.put_pixel(x, y, Luma([128u8]));
            }
        }

        // Create a bright square
        for y in 5..15 {
            for x in 5..15 {
                img.put_pixel(x, y, Luma([255u8]));
            }
        }

        let dynamic_img = DynamicImage::ImageLuma8(img);
        let result = shi_tomasi_corners_simple(&dynamic_img, 10);

        assert!(result.is_ok());
        let corners = result.unwrap();

        // Count detected corners
        let mut corner_count = 0;
        for y in 0..20 {
            for x in 0..20 {
                if corners.get_pixel(x, y)[0] > 0 {
                    corner_count += 1;
                }
            }
        }

        // Should detect corners of the square
        assert!(
            corner_count >= 4,
            "Should detect at least 4 corners, found {}",
            corner_count
        );
    }

    #[test]
    fn test_good_features_to_track() {
        let img = GrayImage::new(20, 20);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result = good_features_to_track(&dynamic_img, 3, 0.01, 10, 5);
        assert!(result.is_ok());

        let features = result.unwrap();
        assert!(features.len() <= 10, "Should not exceed max_corners");
    }

    #[test]
    fn test_invalid_block_size() {
        let img = GrayImage::new(20, 20);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        // Test even block size
        let result = shi_tomasi_corners(&dynamic_img, 4, 0.01, 10, 5);
        assert!(result.is_err());

        // Test too small block size
        let result = shi_tomasi_corners(&dynamic_img, 1, 0.01, 10, 5);
        assert!(result.is_err());
    }
}
