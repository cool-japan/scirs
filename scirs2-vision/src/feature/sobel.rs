//! Sobel edge detection with gradient orientation
//!
//! The Sobel operator is used to detect edges in images by computing
//! the gradient magnitude and optionally the gradient direction.

use crate::error::Result;
use crate::feature::image_to_array;
use image::{DynamicImage, GrayImage};
use ndarray::Array2;
use std::f32::consts::PI;

/// Sobel edge detection with gradient magnitude and orientation
///
/// Computes edge strength and direction using Sobel operators.
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Threshold for edge detection (0.0 to 1.0)
/// * `compute_orientation` - Whether to compute gradient orientation
///
/// # Returns
///
/// * Result containing tuple of (edge_image, Option<orientation_map>)
///   where orientation_map contains angles in radians [-π, π]
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::sobel_edges_oriented;
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let (edges, orientations) = sobel_edges_oriented(&img, 0.1, true)?;
/// # Ok(())
/// # }
/// ```
pub fn sobel_edges_oriented(
    img: &DynamicImage,
    threshold: f32,
    compute_orientation: bool,
) -> Result<(GrayImage, Option<Array2<f32>>)> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Sobel kernels
    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let mut gradient_x = Array2::zeros((height, width));
    let mut gradient_y = Array2::zeros((height, width));
    let mut magnitude = Array2::zeros((height, width));
    let mut orientation = if compute_orientation {
        Some(Array2::zeros((height, width)))
    } else {
        None
    };

    // Apply Sobel operators
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut gx = 0.0;
            let mut gy = 0.0;

            // Convolve with Sobel kernels
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = array[[y + ky - 1, x + kx - 1]];
                    gx += pixel * sobel_x[ky][kx];
                    gy += pixel * sobel_y[ky][kx];
                }
            }

            gradient_x[[y, x]] = gx;
            gradient_y[[y, x]] = gy;

            // Compute gradient magnitude
            let mag = (gx * gx + gy * gy).sqrt();
            magnitude[[y, x]] = mag;

            // Compute gradient orientation if requested
            if let Some(ref mut orient) = orientation {
                orient[[y, x]] = gy.atan2(gx);
            }
        }
    }

    // Create binary edge image
    let edges =
        crate::feature::array_to_image(&magnitude.mapv(|x| if x > threshold { 1.0 } else { 0.0 }))?;

    Ok((edges, orientation))
}

/// Simple Sobel edge detection (magnitude only)
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Threshold for edge detection
///
/// # Returns
///
/// * Result containing edge image
pub fn sobel_edges(img: &DynamicImage, threshold: f32) -> Result<GrayImage> {
    let (edges, _) = sobel_edges_oriented(img, threshold, false)?;
    Ok(edges)
}

/// Compute gradient magnitude and orientation
///
/// Returns the raw gradient magnitude and orientation maps without thresholding.
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// * Result containing tuple of (magnitude_map, orientation_map)
pub fn compute_gradients(img: &DynamicImage) -> Result<(Array2<f32>, Array2<f32>)> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Sobel kernels
    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];

    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    let mut magnitude = Array2::zeros((height, width));
    let mut orientation = Array2::zeros((height, width));

    // Apply Sobel operators
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let mut gx = 0.0;
            let mut gy = 0.0;

            // Convolve with Sobel kernels
            for ky in 0..3 {
                for kx in 0..3 {
                    let pixel = array[[y + ky - 1, x + kx - 1]];
                    gx += pixel * sobel_x[ky][kx];
                    gy += pixel * sobel_y[ky][kx];
                }
            }

            // Compute gradient magnitude
            magnitude[[y, x]] = (gx * gx + gy * gy).sqrt();

            // Compute gradient orientation
            orientation[[y, x]] = gy.atan2(gx);
        }
    }

    Ok((magnitude, orientation))
}

/// Create a color-coded orientation visualization
///
/// Maps gradient orientations to colors using HSV color space.
///
/// # Arguments
///
/// * `magnitude` - Gradient magnitude map
/// * `orientation` - Gradient orientation map (in radians)
/// * `mag_threshold` - Minimum magnitude to display
///
/// # Returns
///
/// * Result containing RGB visualization image
pub fn visualize_gradient_orientation(
    magnitude: &Array2<f32>,
    orientation: &Array2<f32>,
    mag_threshold: f32,
) -> Result<image::RgbImage> {
    use image::{Rgb, RgbImage};

    let (height, width) = magnitude.dim();
    let mut output = RgbImage::new(width as u32, height as u32);

    // Find max magnitude for normalization
    let max_mag = magnitude.iter().fold(0.0f32, |a, &b| a.max(b));

    for y in 0..height {
        for x in 0..width {
            let mag = magnitude[[y, x]];
            let angle = orientation[[y, x]];

            if mag > mag_threshold {
                // Map angle to hue (0-360 degrees)
                let hue = ((angle + PI) / (2.0 * PI)) * 360.0;

                // Use magnitude for saturation and value
                let saturation = (mag / max_mag).min(1.0);
                let value = saturation;

                // Convert HSV to RGB
                let (r, g, b) = hsv_to_rgb(hue, saturation, value);
                output.put_pixel(x as u32, y as u32, Rgb([r, g, b]));
            } else {
                output.put_pixel(x as u32, y as u32, Rgb([0, 0, 0]));
            }
        }
    }

    Ok(output)
}

/// Convert HSV to RGB
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (u8, u8, u8) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
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

    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Compute histogram of oriented gradients for a region
///
/// # Arguments
///
/// * `magnitude` - Gradient magnitude map
/// * `orientation` - Gradient orientation map
/// * `num_bins` - Number of orientation bins
///
/// # Returns
///
/// * Histogram of oriented gradients
pub fn compute_hog_histogram(
    magnitude: &Array2<f32>,
    orientation: &Array2<f32>,
    num_bins: usize,
) -> Vec<f32> {
    let mut histogram = vec![0.0; num_bins];
    let bin_size = 2.0 * PI / num_bins as f32;

    for (&mag, &angle) in magnitude.iter().zip(orientation.iter()) {
        if mag > 0.0 {
            // Normalize angle to [0, 2π]
            let normalized_angle = if angle < 0.0 { angle + 2.0 * PI } else { angle };

            // Compute bin index
            let bin = ((normalized_angle / bin_size) as usize).min(num_bins - 1);

            // Add magnitude to bin
            histogram[bin] += mag;
        }
    }

    // Normalize histogram
    let sum: f32 = histogram.iter().sum();
    if sum > 0.0 {
        for val in &mut histogram {
            *val /= sum;
        }
    }

    histogram
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;

    #[test]
    fn test_sobel_edges() {
        let img = DynamicImage::new_luma8(10, 10);
        let result = sobel_edges(&img, 0.1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sobel_edges_oriented() {
        let img = DynamicImage::new_luma8(10, 10);
        let result = sobel_edges_oriented(&img, 0.1, true);
        assert!(result.is_ok());

        let (edges, orientations) = result.unwrap();
        assert!(orientations.is_some());
        assert_eq!(edges.dimensions(), (10, 10));
    }

    #[test]
    fn test_compute_gradients() {
        // Create test image with vertical edge
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                img.put_pixel(x, y, image::Luma([if x < 5 { 0 } else { 255 }]));
            }
        }
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result = compute_gradients(&dynamic_img);
        assert!(result.is_ok());

        let (magnitude, orientation) = result.unwrap();
        assert_eq!(magnitude.dim(), (10, 10));
        assert_eq!(orientation.dim(), (10, 10));

        // Check that we detect edge around x=5
        assert!(magnitude[[5, 5]] > 0.0);
    }

    #[test]
    fn test_hsv_to_rgb() {
        // Test primary colors
        assert_eq!(hsv_to_rgb(0.0, 1.0, 1.0), (255, 0, 0)); // Red
        assert_eq!(hsv_to_rgb(120.0, 1.0, 1.0), (0, 255, 0)); // Green
        assert_eq!(hsv_to_rgb(240.0, 1.0, 1.0), (0, 0, 255)); // Blue

        // Test grayscale
        assert_eq!(hsv_to_rgb(0.0, 0.0, 0.5), (127, 127, 127)); // Gray
    }

    #[test]
    fn test_hog_histogram() {
        let magnitude =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0])
                .unwrap();

        let orientation = Array2::from_shape_vec(
            (3, 3),
            vec![0.0, 0.0, PI / 2.0, 0.0, PI, 0.0, -PI / 2.0, 0.0, PI],
        )
        .unwrap();

        let histogram = compute_hog_histogram(&magnitude, &orientation, 4);

        assert_eq!(histogram.len(), 4);

        // Check normalization
        let sum: f32 = histogram.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
