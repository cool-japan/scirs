//! Laplacian edge detector implementation
//!
//! The Laplacian is a 2D isotropic measure of the 2nd spatial derivative
//! of an image. It highlights regions of rapid intensity change and is
//! therefore often used for edge detection.

use crate::error::Result;
use crate::feature::{array_to_image, image_to_array};
use image::{DynamicImage, GrayImage};
use ndarray::Array2;

/// Apply Laplacian edge detection
///
/// The Laplacian operator is a second-order derivative operator that
/// detects edges by finding the zero-crossings of the second derivative.
///
/// # Arguments
///
/// * `img` - Input image
/// * `threshold` - Threshold value for edge detection
/// * `use_diagonal` - If true, use 8-connected kernel, else 4-connected
///
/// # Returns
///
/// * Result containing an edge image
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::laplacian_edges;
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let edges = laplacian_edges(&img, 0.1, true)?;
/// # Ok(())
/// # }
/// ```
pub fn laplacian_edges(
    img: &DynamicImage,
    threshold: f32,
    use_diagonal: bool,
) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Create output array
    let mut edges = Array2::zeros(array.dim());

    // Apply Laplacian operator
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let laplacian = if use_diagonal {
                // 8-connected Laplacian kernel: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
                -1.0 * array[[y - 1, x - 1]]
                    + -1.0 * array[[y - 1, x]]
                    + -1.0 * array[[y - 1, x + 1]]
                    + -1.0 * array[[y, x - 1]]
                    + 8.0 * array[[y, x]]
                    + -1.0 * array[[y, x + 1]]
                    + -1.0 * array[[y + 1, x - 1]]
                    + -1.0 * array[[y + 1, x]]
                    + -1.0 * array[[y + 1, x + 1]]
            } else {
                // 4-connected Laplacian kernel: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
                0.0 * array[[y - 1, x - 1]]
                    + -1.0 * array[[y - 1, x]]
                    + 0.0 * array[[y - 1, x + 1]]
                    + -1.0 * array[[y, x - 1]]
                    + 4.0 * array[[y, x]]
                    + -1.0 * array[[y, x + 1]]
                    + 0.0 * array[[y + 1, x - 1]]
                    + -1.0 * array[[y + 1, x]]
                    + 0.0 * array[[y + 1, x + 1]]
            };

            // Apply threshold to absolute value
            if laplacian.abs() > threshold {
                edges[[y, x]] = 1.0;
            }
        }
    }

    // Convert to image
    array_to_image(&edges)
}

/// Apply Laplacian of Gaussian (LoG) edge detection
///
/// This combines Gaussian smoothing with Laplacian edge detection,
/// which helps reduce noise sensitivity.
///
/// # Arguments
///
/// * `img` - Input image
/// * `sigma` - Standard deviation for Gaussian smoothing
/// * `threshold` - Threshold value for edge detection
///
/// # Returns
///
/// * Result containing an edge image
pub fn laplacian_of_gaussian(img: &DynamicImage, sigma: f32, threshold: f32) -> Result<GrayImage> {
    use crate::preprocessing::gaussian_blur;

    // First apply Gaussian blur
    let blurred = gaussian_blur(img, sigma)?;

    // Then apply Laplacian
    laplacian_edges(&blurred, threshold, true)
}

/// Apply zero-crossing detection to find edges in Laplacian result
///
/// This method looks for sign changes in the Laplacian result to detect edges.
///
/// # Arguments
///
/// * `img` - Input image
/// * `use_diagonal` - If true, use 8-connected kernel, else 4-connected
///
/// # Returns
///
/// * Result containing an edge image with zero-crossings marked
pub fn laplacian_zero_crossing(img: &DynamicImage, use_diagonal: bool) -> Result<GrayImage> {
    let array = image_to_array(img)?;
    let (height, width) = array.dim();

    // Create Laplacian array
    let mut laplacian = Array2::zeros(array.dim());

    // Apply Laplacian operator
    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            laplacian[[y, x]] = if use_diagonal {
                // 8-connected Laplacian
                -1.0 * array[[y - 1, x - 1]]
                    + -1.0 * array[[y - 1, x]]
                    + -1.0 * array[[y - 1, x + 1]]
                    + -1.0 * array[[y, x - 1]]
                    + 8.0 * array[[y, x]]
                    + -1.0 * array[[y, x + 1]]
                    + -1.0 * array[[y + 1, x - 1]]
                    + -1.0 * array[[y + 1, x]]
                    + -1.0 * array[[y + 1, x + 1]]
            } else {
                // 4-connected Laplacian
                -1.0 * array[[y - 1, x]]
                    + -1.0 * array[[y, x - 1]]
                    + 4.0 * array[[y, x]]
                    + -1.0 * array[[y, x + 1]]
                    + -1.0 * array[[y + 1, x]]
            };
        }
    }

    // Find zero-crossings
    let mut edges = Array2::zeros(array.dim());

    for y in 1..(height - 1) {
        for x in 1..(width - 1) {
            let center = laplacian[[y, x]];

            // Check for zero-crossing in horizontal and vertical directions
            let has_crossing = (center * laplacian[[y - 1, x]] < 0.0)
                || (center * laplacian[[y + 1, x]] < 0.0)
                || (center * laplacian[[y, x - 1]] < 0.0)
                || (center * laplacian[[y, x + 1]] < 0.0);

            if use_diagonal {
                // Also check diagonal directions
                let has_diagonal_crossing = (center * laplacian[[y - 1, x - 1]] < 0.0)
                    || (center * laplacian[[y - 1, x + 1]] < 0.0)
                    || (center * laplacian[[y + 1, x - 1]] < 0.0)
                    || (center * laplacian[[y + 1, x + 1]] < 0.0);

                if has_crossing || has_diagonal_crossing {
                    edges[[y, x]] = 1.0;
                }
            } else if has_crossing {
                edges[[y, x]] = 1.0;
            }
        }
    }

    // Convert to image
    array_to_image(&edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn test_laplacian_4_connected() {
        // Create a simple test image with a square
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let value = if x >= 3 && x <= 6 && y >= 3 && y <= 6 {
                    255
                } else {
                    0
                };
                img.put_pixel(x, y, Luma([value]));
            }
        }

        let dynamic_img = DynamicImage::ImageLuma8(img);
        let result = laplacian_edges(&dynamic_img, 0.1, false);

        assert!(result.is_ok());
        let edges = result.unwrap();

        // Should detect edges around the square
        let mut has_edge = false;
        for y in 2..8 {
            if edges.get_pixel(2, y)[0] > 0 || edges.get_pixel(7, y)[0] > 0 {
                has_edge = true;
                break;
            }
        }
        assert!(has_edge, "Should detect square edges");
    }

    #[test]
    fn test_laplacian_8_connected() {
        let img = GrayImage::new(10, 10);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result_4 = laplacian_edges(&dynamic_img, 0.1, false);
        let result_8 = laplacian_edges(&dynamic_img, 0.1, true);

        assert!(result_4.is_ok());
        assert!(result_8.is_ok());
    }

    #[test]
    fn test_laplacian_of_gaussian() {
        let img = GrayImage::new(10, 10);
        let dynamic_img = DynamicImage::ImageLuma8(img);

        let result = laplacian_of_gaussian(&dynamic_img, 1.0, 0.1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_zero_crossing() {
        // Create an image with gradual intensity change
        let mut img = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let value = ((x * 255) / 10) as u8;
                img.put_pixel(x, y, Luma([value]));
            }
        }

        let dynamic_img = DynamicImage::ImageLuma8(img);
        let result = laplacian_zero_crossing(&dynamic_img, true);

        assert!(result.is_ok());
    }
}
