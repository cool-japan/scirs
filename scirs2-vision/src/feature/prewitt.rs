//! Prewitt edge detector implementation
//!
//! The Prewitt operator is used for edge detection in image processing.
//! It's a discrete differentiation operator, computing an approximation
//! of the gradient of the image intensity function.

use crate::error::Result;
use crate::feature::{array_to_image, image_to_array};
use image::{DynamicImage, GrayImage};
use ndarray::Array2;

/// Apply Prewitt edge detection
///
/// The Prewitt operator is similar to the Sobel operator but uses simpler
/// kernel coefficients. It detects edges in horizontal and vertical directions.
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Threshold value for edge detection
///
/// # Returns
///
/// * Result containing an edge image
///
/// # Example
///
/// ```rust,ignore
/// use scirs2_vision::feature::prewitt_edges;
/// use image::DynamicImage;
///
/// let img = image::open("input.jpg").unwrap();
/// let edges = prewitt_edges(&img, 0.1)?;
/// ```
pub fn prewitt_edges(img: &DynamicImage, threshold: f32) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Create output array
    let mut edges = Array2::zeros(array.dim());

    // Apply Prewitt operator
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Horizontal Prewitt kernel: [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
            let gx = -1.0 * array[[y - 1, x - 1]]
                + 0.0 * array[[y - 1, x]]
                + 1.0 * array[[y - 1, x + 1]]
                + -1.0 * array[[y, x - 1]]
                + 0.0 * array[[y, x]]
                + 1.0 * array[[y, x + 1]]
                + -1.0 * array[[y + 1, x - 1]]
                + 0.0 * array[[y + 1, x]]
                + 1.0 * array[[y + 1, x + 1]];

            // Vertical Prewitt kernel: [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
            let gy = -1.0 * array[[y - 1, x - 1]]
                + -1.0 * array[[y - 1, x]]
                + -1.0 * array[[y - 1, x + 1]]
                + 0.0 * array[[y, x - 1]]
                + 0.0 * array[[y, x]]
                + 0.0 * array[[y, x + 1]]
                + 1.0 * array[[y + 1, x - 1]]
                + 1.0 * array[[y + 1, x]]
                + 1.0 * array[[y + 1, x + 1]];

            // Calculate magnitude
            let magnitude = (gx * gx + gy * gy).sqrt();

            // Apply threshold
            if magnitude > threshold {
                edges[[y, x]] = 1.0;
            }
        }
    }

    // Convert to image
    array_to_image(&edges)
}

/// Apply Prewitt edge detection with magnitude and direction
///
/// This variant returns both the edge magnitude and direction arrays,
/// which can be useful for further processing like non-maximum suppression.
///
/// # Arguments
///
/// * `img` - Input image
///
/// # Returns
///
/// * Result containing a tuple of (magnitude array, direction array in radians)
pub fn prewitt_gradients(img: &DynamicImage) -> Result<(Array2<f32>, Array2<f32>)> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Create output arrays
    let mut magnitude = Array2::zeros(array.dim());
    let mut direction = Array2::zeros(array.dim());

    // Apply Prewitt operator
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            // Horizontal Prewitt kernel
            let gx = -1.0 * array[[y - 1, x - 1]]
                + 1.0 * array[[y - 1, x + 1]]
                + -1.0 * array[[y, x - 1]]
                + 1.0 * array[[y, x + 1]]
                + -1.0 * array[[y + 1, x - 1]]
                + 1.0 * array[[y + 1, x + 1]];

            // Vertical Prewitt kernel
            let gy = -1.0 * array[[y - 1, x - 1]]
                + -1.0 * array[[y - 1, x]]
                + -1.0 * array[[y - 1, x + 1]]
                + 1.0 * array[[y + 1, x - 1]]
                + 1.0 * array[[y + 1, x]]
                + 1.0 * array[[y + 1, x + 1]];

            // Calculate magnitude and direction
            magnitude[[y, x]] = (gx * gx + gy * gy).sqrt();
            direction[[y, x]] = gy.atan2(gx);
        }
    }

    Ok((magnitude, direction))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_prewitt_on_simple_edge() {
        // Create a simple test image with a vertical edge
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let value = if x < 5 { 0 } else { 255 };
                img.put_pixel(x, y, Luma([value]));
            }
        }

        let dynamic_img = DynamicImage::ImageLuma8(img);
        let result = prewitt_edges(&dynamic_img, 0.1);

        assert!(result.is_ok());
        let edges = result.unwrap();

        // Should detect edge around x=5
        let mut has_edge = false;
        for y in 1..9 {
            for x in 4..7 {
                if edges.get_pixel(x, y)[0] > 0 {
                    has_edge = true;
                    break;
                }
            }
        }
        assert!(has_edge, "Should detect vertical edge");
    }

    #[test]
    fn test_prewitt_gradients() {
        let img = GrayImage::new(10, 10);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result = prewitt_gradients(&dynamic_img);
        assert!(result.is_ok());

        let (magnitude, direction) = result.unwrap();
        assert_eq!(magnitude.dim(), (10, 10));
        assert_eq!(direction.dim(), (10, 10));
    }

    #[test]
    fn test_prewitt_vs_sobel_difference() {
        // Create a diagonal edge
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let value = if x > y { 255 } else { 0 };
                img.put_pixel(x, y, Luma([value]));
            }
        }

        let dynamic_img = DynamicImage::ImageLuma8(img);
        let prewitt = prewitt_edges(&dynamic_img, 0.1).unwrap();

        // Prewitt should detect diagonal edge
        let mut has_diagonal = false;
        for i in 2..8 {
            if prewitt.get_pixel(i, i)[0] > 0 {
                has_diagonal = true;
                break;
            }
        }
        assert!(has_diagonal, "Should detect diagonal edge");
    }
}
