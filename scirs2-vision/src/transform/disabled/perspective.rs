//! Perspective (projective) transformations for image geometry
//!
//! This module provides functions for perspective transformations
//! such as homography estimation, perspective warping, and
//! perspective correction.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgba};
use ndarray::Array2;

/// Border handling methods for areas outside the image boundaries
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode {
    /// Fill with constant color
    Constant(Rgba<u8>),
    /// Reflect image content across edges
    Reflect,
    /// Replicate edge pixels
    Replicate,
    /// Wrap pixels around the opposite edge
    Wrap,
    /// Leave the area transparent (alpha channel set to 0)
    Transparent,
}

impl Default for BorderMode {
    fn default() -> Self {
        Self::Constant(Rgba([0, 0, 0, 255]))
    }
}

/// 3x3 Perspective transformation matrix
#[derive(Debug, Clone)]
pub struct PerspectiveTransform {
    /// Homography matrix
    pub matrix: Array2<f64>,
}

impl PerspectiveTransform {
    /// Create a new perspective transformation matrix from raw data
    pub fn new(data: [f64; 9]) -> Self {
        let matrix = Array2::from_shape_vec((3, 3), data.to_vec()).unwrap();
        Self { matrix }
    }

    /// Create an identity transformation that leaves the image unchanged
    pub fn identity() -> Self {
        Self::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    }

    /// Compute an affine transformation from three point correspondences
    pub fn from_points(src_points: &[(f64, f64)], dst_points: &[(f64, f64)]) -> Result<Self> {
        if src_points.len() != dst_points.len() {
            return Err(VisionError::InvalidParameter(
                "Source and destination point sets must have the same length".to_string(),
            ));
        }

        if src_points.len() < 4 {
            return Err(VisionError::InvalidParameter(
                "At least 4 point correspondences are required".to_string(),
            ));
        }

        // Create coefficient matrix for the homogeneous system
        let mut a = Array2::zeros((2 * src_points.len(), 9));

        for (i, (src, dst)) in src_points.iter().zip(dst_points.iter()).enumerate() {
            let (x, y) = *src;
            let (u, v) = *dst;

            // Set up the 2 rows in the coefficient matrix for this correspondence
            let idx = i * 2;
            a[[idx, 0]] = x;
            a[[idx, 1]] = y;
            a[[idx, 2]] = 1.0;
            a[[idx, 3]] = 0.0;
            a[[idx, 4]] = 0.0;
            a[[idx, 5]] = 0.0;
            a[[idx, 6]] = -x * u;
            a[[idx, 7]] = -y * u;
            a[[idx, 8]] = -u;

            let idx = i * 2 + 1;
            a[[idx, 0]] = 0.0;
            a[[idx, 1]] = 0.0;
            a[[idx, 2]] = 0.0;
            a[[idx, 3]] = x;
            a[[idx, 4]] = y;
            a[[idx, 5]] = 1.0;
            a[[idx, 6]] = -x * v;
            a[[idx, 7]] = -y * v;
            a[[idx, 8]] = -v;
        }

        // Solve the system using SVD
        use ndarray_linalg::svddc::SVDDC;
        let svd = a.svddc()
            .map_err(|e| VisionError::LinAlgError(e.to_string()))?;
        let (_, s, vt) = svd;

        // Find the smallest singular value index
        let min_idx = s
            .iter()
            .enumerate()
            .min_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(8);

        // Get the corresponding right singular vector
        let h = vt.row(min_idx);
        let h_data = h.to_vec();

        // Create homography matrix
        let h_matrix = Array2::from_shape_vec((3, 3), h_data)?;

        // Normalize so that h[2,2] = 1
        let norm_factor = h_matrix[[2, 2]];
        if norm_factor.abs() < 1e-10 {
            return Err(VisionError::OperationFailed(
                "Failed to compute homography - degenerate case".to_string(),
            ));
        }

        let h_normalized = h_matrix.mapv(|v| v / norm_factor);

        Ok(Self {
            matrix: h_normalized,
        })
    }

    /// Compute a perspective transformation to map a rectangle to a quadrilateral
    ///
    /// # Arguments
    ///
    /// * `src_rect` - Source rectangle as (x, y, width, height)
    /// * `dst_quad` - Destination quadrilateral as 4 points in clockwise order from top-left
    ///
    /// # Returns
    ///
    /// * The perspective transformation
    pub fn rect_to_quad(src_rect: (f64, f64, f64, f64), dst_quad: [(f64, f64); 4]) -> Result<Self> {
        let (x, y, width, height) = src_rect;

        let src_points = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ];

        Self::from_points(&src_points, &dst_quad)
    }

    /// Compute a perspective transformation to map a quadrilateral to a rectangle
    ///
    /// # Arguments
    ///
    /// * `src_quad` - Source quadrilateral as 4 points in clockwise order from top-left
    /// * `dst_rect` - Destination rectangle as (x, y, width, height)
    ///
    /// # Returns
    ///
    /// * The perspective transformation
    pub fn quad_to_rect(src_quad: [(f64, f64); 4], dst_rect: (f64, f64, f64, f64)) -> Result<Self> {
        let (x, y, width, height) = dst_rect;

        let dst_points = [
            (x, y),
            (x + width, y),
            (x + width, y + height),
            (x, y + height),
        ];

        Self::from_points(&src_quad, &dst_points)
    }

    /// Get the inverse transformation
    ///
    /// # Returns
    ///
    /// * The inverse perspective transformation
    pub fn inverse(&self) -> Result<Self> {
        use ndarray_linalg::solve::Inverse;
        let inv = self.matrix.inv()
            .map_err(|e| VisionError::LinAlgError(e.to_string()))?;
        Ok(Self { matrix: inv })
    }

    /// Transform a point using this perspective transformation
    ///
    /// # Arguments
    ///
    /// * `point` - The point to transform (x, y)
    ///
    /// # Returns
    ///
    /// * The transformed point (x', y')
    pub fn transform_point(&self, point: (f64, f64)) -> (f64, f64) {
        let (x, y) = point;
        let h = &self.matrix;

        let w = h[[2, 0]] * x + h[[2, 1]] * y + h[[2, 2]];
        let w_inv = if w.abs() > 1e-10 { 1.0 / w } else { 1.0 };

        let x_prime = (h[[0, 0]] * x + h[[0, 1]] * y + h[[0, 2]]) * w_inv;
        let y_prime = (h[[1, 0]] * x + h[[1, 1]] * y + h[[1, 2]]) * w_inv;

        (x_prime, y_prime)
    }
}

/// Warp an image using a perspective transformation
///
/// # Arguments
///
/// * `src` - Source image
/// * `transform` - Perspective transformation matrix
/// * `width` - Output width (if None, uses source width)
/// * `height` - Output height (if None, uses source height)
/// * `border_mode` - How to handle pixels outside image boundaries
///
/// # Returns
///
/// * Result containing the warped image
pub fn warp_perspective(
    src: &DynamicImage,
    transform: &PerspectiveTransform,
    width: Option<u32>,
    height: Option<u32>,
    border_mode: BorderMode,
) -> Result<DynamicImage> {
    let (src_width, src_height) = src.dimensions();
    let dst_width = width.unwrap_or(src_width);
    let dst_height = height.unwrap_or(src_height);

    // Create output image
    let mut dst: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(dst_width, dst_height);

    // Compute inverse transform for backward mapping
    let inv_transform = transform.inverse()?;

    // For each pixel in destination image, find corresponding pixel in source
    for y in 0..dst_height {
        for x in 0..dst_width {
            // Convert to floating point
            let dst_pt = (x as f64, y as f64);

            // Apply inverse transform to get source coordinates
            let (src_x, src_y) = inv_transform.transform_point(dst_pt);

            // Check if source point is inside image bounds
            if src_x >= 0.0 && src_x < src_width as f64 && src_y >= 0.0 && src_y < src_height as f64
            {
                // Bilinear interpolation for smoother results
                let color = bilinear_interpolate(src, src_x, src_y);
                dst.put_pixel(x, y, color);
            } else {
                // Handle out-of-bounds pixels according to border mode
                match border_mode {
                    BorderMode::Constant(color) => {
                        dst.put_pixel(x, y, color);
                    }
                    BorderMode::Reflect => {
                        // Reflect coordinates at image boundaries
                        let reflected_x = reflect_coordinate(src_x, src_width as f64);
                        let reflected_y = reflect_coordinate(src_y, src_height as f64);
                        let color = bilinear_interpolate(src, reflected_x, reflected_y);
                        dst.put_pixel(x, y, color);
                    }
                    BorderMode::Replicate => {
                        // Clamp coordinates to image boundaries
                        let clamped_x = src_x.max(0.0).min(src_width as f64 - 1.0);
                        let clamped_y = src_y.max(0.0).min(src_height as f64 - 1.0);
                        let color = bilinear_interpolate(src, clamped_x, clamped_y);
                        dst.put_pixel(x, y, color);
                    }
                    BorderMode::Wrap => {
                        // Wrap coordinates around image boundaries
                        let wrapped_x = modulo(src_x, src_width as f64);
                        let wrapped_y = modulo(src_y, src_height as f64);
                        let color = bilinear_interpolate(src, wrapped_x, wrapped_y);
                        dst.put_pixel(x, y, color);
                    }
                    BorderMode::Transparent => {
                        // Set transparent pixel (alpha = 0)
                        dst.put_pixel(x, y, Rgba([0, 0, 0, 0]));
                    }
                }
            }
        }
    }

    Ok(DynamicImage::ImageRgba8(dst))
}

/// Perform bilinear interpolation for a point in the image
///
/// # Arguments
///
/// * `img` - Source image
/// * `x` - X coordinate (floating point)
/// * `y` - Y coordinate (floating point)
///
/// # Returns
///
/// * Interpolated color value
pub fn bilinear_interpolate(img: &DynamicImage, x: f64, y: f64) -> Rgba<u8> {
    let (width, height) = img.dimensions();

    // Get integer and fractional parts
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    // Get the four surrounding pixels
    let p00 = img.get_pixel(x0, y0).to_rgba();
    let p01 = img.get_pixel(x0, y1).to_rgba();
    let p10 = img.get_pixel(x1, y0).to_rgba();
    let p11 = img.get_pixel(x1, y1).to_rgba();

    // Interpolate each channel separately
    let mut result = [0u8; 4];
    for c in 0..4 {
        // Bilinear interpolation formula
        let c00 = p00[c] as f64;
        let c01 = p01[c] as f64;
        let c10 = p10[c] as f64;
        let c11 = p11[c] as f64;

        let value = (1.0 - dx) * (1.0 - dy) * c00
            + dx * (1.0 - dy) * c10
            + (1.0 - dx) * dy * c01
            + dx * dy * c11;

        // Clamp to valid range and round
        result[c] = value.round().clamp(0.0, 255.0) as u8;
    }

    Rgba(result)
}

/// Reflect a coordinate at image boundaries
///
/// # Arguments
///
/// * `coord` - The coordinate to reflect
/// * `size` - The size of the dimension
///
/// # Returns
///
/// * The reflected coordinate
pub fn reflect_coordinate(coord: f64, size: f64) -> f64 {
    if coord < 0.0 {
        -coord
    } else if coord >= size {
        2.0 * size - coord - 1.0
    } else {
        coord
    }
}

/// Compute modulo for floating point (handles negative numbers)
///
/// # Arguments
///
/// * `a` - The dividend
/// * `b` - The divisor
///
/// # Returns
///
/// * The modulo result
pub fn modulo(a: f64, b: f64) -> f64 {
    ((a % b) + b) % b
}

/// Detect a quadrilateral in an image (e.g., document corners)
///
/// # Arguments
///
/// * `src` - Source image
/// * `threshold` - Edge detection threshold (0-255)
///
/// # Returns
///
/// * Result containing the found quadrilateral as 4 points
pub fn detect_quad(src: &DynamicImage, _threshold: u8) -> Result<[(f64, f64); 4]> {
    // This is a placeholder implementation
    // A real implementation would use edge detection, contour finding, etc.

    // Simple placeholder implementation returning the image corners
    let (width, height) = src.dimensions();
    let corners = [
        (0.0, 0.0),
        (width as f64, 0.0),
        (width as f64, height as f64),
        (0.0, height as f64),
    ];

    Ok(corners)
}

/// Correct perspective distortion in an image (e.g., document scanner)
///
/// # Arguments
///
/// * `src` - Source image
/// * `corners` - Four corners of the quadrilateral to rectify
/// * `aspect_ratio` - Desired aspect ratio of the output (if None, uses detected aspect ratio)
///
/// # Returns
///
/// * Result containing the rectified image
pub fn correct_perspective(
    src: &DynamicImage,
    corners: [(f64, f64); 4],
    aspect_ratio: Option<f64>,
) -> Result<DynamicImage> {
    // Calculate the desired width and height
    let width_top = distance(corners[0], corners[1]);
    let width_bottom = distance(corners[3], corners[2]);
    let height_left = distance(corners[0], corners[3]);
    let height_right = distance(corners[1], corners[2]);

    let max_width = width_top.max(width_bottom).round() as u32;
    let max_height = height_left.max(height_right).round() as u32;

    // Adjust for aspect ratio if provided
    let (dst_width, dst_height) = if let Some(ratio) = aspect_ratio {
        let current_ratio = max_width as f64 / max_height as f64;

        if current_ratio > ratio {
            // Width is too large relative to height
            let new_width = (max_height as f64 * ratio).round() as u32;
            (new_width, max_height)
        } else {
            // Height is too large relative to width
            let new_height = (max_width as f64 / ratio).round() as u32;
            (max_width, new_height)
        }
    } else {
        (max_width, max_height)
    };

    // Destination rectangle
    let dst_rect = (0.0, 0.0, dst_width as f64, dst_height as f64);

    // Compute perspective transform
    let transform = PerspectiveTransform::quad_to_rect(corners, dst_rect)?;

    // Warp the image
    warp_perspective(
        src,
        &transform,
        Some(dst_width),
        Some(dst_height),
        BorderMode::Transparent,
    )
}

/// Calculate Euclidean distance between two points
///
/// # Arguments
///
/// * `p1` - First point
/// * `p2` - Second point
///
/// # Returns
///
/// * The distance between the points
fn distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_perspective_identity() {
        let transform = PerspectiveTransform::identity();

        // Identity transform should leave points unchanged
        let point = (10.0, 20.0);
        let transformed = transform.transform_point(point);

        assert!((transformed.0 - point.0).abs() < 1e-10);
        assert!((transformed.1 - point.1).abs() < 1e-10);
    }

    #[test]
    fn test_perspective_transform_point() {
        // Create a perspective transform that scales by 2 in both dimensions
        let transform = PerspectiveTransform::new([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]);

        let point = (10.0, 20.0);
        let transformed = transform.transform_point(point);

        assert!((transformed.0 - 20.0).abs() < 1e-10);
        assert!((transformed.1 - 40.0).abs() < 1e-10);
    }

    #[test]
    fn test_perspective_inverse() {
        // Create a transform
        let transform = PerspectiveTransform::new([2.0, 1.0, 3.0, 0.0, 1.0, 5.0, 0.0, 0.0, 1.0]);

        // Get its inverse
        let inverse = transform.inverse().unwrap();

        // Transform a point and then transform it back
        let point = (10.0, 20.0);
        let transformed = transform.transform_point(point);
        let back = inverse.transform_point(transformed);

        // Should get original point back
        assert!((back.0 - point.0).abs() < 1e-10);
        assert!((back.1 - point.1).abs() < 1e-10);
    }

    #[test]
    fn test_warp_perspective_identity() {
        // Create a simple test image
        let width = 100;
        let height = 100;
        let mut img = RgbImage::new(width, height);

        // Fill with a diagonal pattern
        for y in 0..height {
            for x in 0..width {
                let color = if x == y {
                    Rgb([255, 0, 0]) // Red diagonal
                } else {
                    Rgb([0, 0, 0]) // Black background
                };
                img.put_pixel(x as u32, y as u32, color);
            }
        }

        let src = DynamicImage::ImageRgb8(img);

        // Apply identity transform
        let transform = PerspectiveTransform::identity();
        let result = warp_perspective(&src, &transform, None, None, BorderMode::default()).unwrap();

        // Dimensions should be the same
        assert_eq!(result.width(), width as u32);
        assert_eq!(result.height(), height as u32);

        // Red diagonal should be preserved
        for i in 0..width {
            let color = result.get_pixel(i as u32, i as u32).to_rgb();
            assert_eq!(color[0], 255);
            assert_eq!(color[1], 0);
            assert_eq!(color[2], 0);
        }
    }
}
