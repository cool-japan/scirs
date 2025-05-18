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

    /// Compute a perspective transformation from point correspondences
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

        // Use a simple approach for SVD computation
        // In a real implementation, we would use ndarray_linalg's SVDDC
        // but for compatibility, we'll implement a simpler approach
        let h = Self::compute_homography_from_system(&a)?;

        // Create homography matrix
        let h_matrix = Array2::from_shape_vec((3, 3), h.to_vec())?;

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
        // Compute the determinant to check invertibility
        let det = self.compute_determinant();
        if det.abs() < 1e-10 {
            return Err(VisionError::OperationFailed(
                "Matrix is singular, cannot compute inverse".to_string(),
            ));
        }

        // Compute the adjugate matrix
        let mut inv = Array2::zeros((3, 3));

        // Cofactors for the inverse
        inv[[0, 0]] =
            self.matrix[[1, 1]] * self.matrix[[2, 2]] - self.matrix[[1, 2]] * self.matrix[[2, 1]];
        inv[[0, 1]] =
            self.matrix[[0, 2]] * self.matrix[[2, 1]] - self.matrix[[0, 1]] * self.matrix[[2, 2]];
        inv[[0, 2]] =
            self.matrix[[0, 1]] * self.matrix[[1, 2]] - self.matrix[[0, 2]] * self.matrix[[1, 1]];
        inv[[1, 0]] =
            self.matrix[[1, 2]] * self.matrix[[2, 0]] - self.matrix[[1, 0]] * self.matrix[[2, 2]];
        inv[[1, 1]] =
            self.matrix[[0, 0]] * self.matrix[[2, 2]] - self.matrix[[0, 2]] * self.matrix[[2, 0]];
        inv[[1, 2]] =
            self.matrix[[0, 2]] * self.matrix[[1, 0]] - self.matrix[[0, 0]] * self.matrix[[1, 2]];
        inv[[2, 0]] =
            self.matrix[[1, 0]] * self.matrix[[2, 1]] - self.matrix[[1, 1]] * self.matrix[[2, 0]];
        inv[[2, 1]] =
            self.matrix[[0, 1]] * self.matrix[[2, 0]] - self.matrix[[0, 0]] * self.matrix[[2, 1]];
        inv[[2, 2]] =
            self.matrix[[0, 0]] * self.matrix[[1, 1]] - self.matrix[[0, 1]] * self.matrix[[1, 0]];

        // Divide by determinant
        inv.mapv_inplace(|v| v / det);

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

    // Helper method to compute determinant
    fn compute_determinant(&self) -> f64 {
        let m = &self.matrix;
        m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
            - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
            + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]])
    }

    // Helper method to compute homography from the linear system
    fn compute_homography_from_system(a: &Array2<f64>) -> Result<Vec<f64>> {
        // This is a simplified version to compute the homography
        // In a real implementation, we'd use SVD, but for now we'll
        // use a simpler approach based on the normal equations

        // Step 1: Compute A^T * A
        let (m, n) = a.dim();
        let mut ata = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                for k in 0..m {
                    ata[[i, j]] += a[[k, i]] * a[[k, j]];
                }
            }
        }

        // Find the smallest eigenvalue of A^T * A using power iteration
        let eigenvector = Self::find_smallest_eigenvector(&ata)?;

        Ok(eigenvector)
    }

    // Finds the eigenvector corresponding to the smallest eigenvalue
    fn find_smallest_eigenvector(matrix: &Array2<f64>) -> Result<Vec<f64>> {
        let n = matrix.shape()[0];

        // Start with a random vector
        let mut v = vec![1.0; n];

        // Normalize v
        let mut norm: f64 = 0.0;
        for val in &v {
            norm += val * val;
        }
        norm = norm.sqrt();

        for v_i in v.iter_mut().take(n) {
            *v_i /= norm;
        }

        // Iterate to find eigenvector
        for _ in 0..50 {
            // Matrix-vector multiplication: Mv
            let mut mv = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    mv[i] += matrix[[i, j]] * v[j];
                }
            }

            // Compute Rayleigh quotient: v^T * M * v / (v^T * v)
            let mut rayleigh = 0.0;
            for i in 0..n {
                rayleigh += v[i] * mv[i];
            }

            // Shift to find smallest eigenvalue: v = v - rayleigh*v
            for i in 0..n {
                v[i] = mv[i] - rayleigh * v[i];
            }

            // Renormalize
            norm = 0.0;
            for val in &v {
                norm += val * val;
            }
            norm = norm.sqrt();

            if norm < 1e-10 {
                // Use a different starting vector if we converge to zero
                for (i, v_i) in v.iter_mut().enumerate().take(n) {
                    *v_i = (i + 1) as f64;
                }

                norm = 0.0;
                for val in &v {
                    norm += val * val;
                }
                norm = norm.sqrt();
            }

            for v_i in v.iter_mut().take(n) {
                *v_i /= norm;
            }
        }

        Ok(v)
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

    let dx = x - f64::from(x0);
    let dy = y - f64::from(y0);

    // Get the four surrounding pixels
    let p00 = img.get_pixel(x0, y0).to_rgba();
    let p01 = img.get_pixel(x0, y1).to_rgba();
    let p10 = img.get_pixel(x1, y0).to_rgba();
    let p11 = img.get_pixel(x1, y1).to_rgba();

    // Interpolate each channel separately
    let mut result = [0u8; 4];
    for c in 0..4 {
        // Bilinear interpolation formula
        let c00 = f64::from(p00[c]);
        let c01 = f64::from(p01[c]);
        let c10 = f64::from(p10[c]);
        let c11 = f64::from(p11[c]);

        let value = (1.0 - dx) * (1.0 - dy) * c00
            + dx * (1.0 - dy) * c10
            + (1.0 - dx) * dy * c01
            + dx * dy * c11;

        // Clamp to valid range and round
        // Intentionally truncating float to integer for pixel values
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
#[must_use]
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
#[must_use]
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
///
/// # Errors
///
/// Returns an error if the quadrilateral detection fails
pub fn detect_quad(src: &DynamicImage, _threshold: u8) -> Result<[(f64, f64); 4]> {
    // This is a placeholder implementation
    // A real implementation would use edge detection, contour finding, etc.

    // Simple placeholder implementation returning the image corners
    let (width, height) = src.dimensions();
    let corners = [
        (0.0, 0.0),
        (f64::from(width), 0.0),
        (f64::from(width), f64::from(height)),
        (0.0, f64::from(height)),
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
///
/// # Errors
///
/// Returns an error if the perspective transformation cannot be calculated
/// or if the warping operation fails
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

    // Intentionally truncating float to integer for pixel dimensions
    let max_width = width_top.max(width_bottom).round() as u32;
    // Intentionally truncating float to integer for pixel dimensions
    let max_height = height_left.max(height_right).round() as u32;

    // Adjust for aspect ratio if provided
    let (dst_width, dst_height) = if let Some(ratio) = aspect_ratio {
        let current_ratio = f64::from(max_width) / f64::from(max_height);

        if current_ratio > ratio {
            // Width is too large relative to height
            // Intentionally truncating float to integer for pixel dimensions
            let new_width = (f64::from(max_height) * ratio).round() as u32;
            (new_width, max_height)
        } else {
            // Height is too large relative to width
            // Intentionally truncating float to integer for pixel dimensions
            let new_height = (f64::from(max_width) / ratio).round() as u32;
            (max_width, new_height)
        }
    } else {
        (max_width, max_height)
    };

    // Destination rectangle
    let dst_rect = (0.0, 0.0, f64::from(dst_width), f64::from(dst_height));

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
    // Imports for potential future color image support

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
}
