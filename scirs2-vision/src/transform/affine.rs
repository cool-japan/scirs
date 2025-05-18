//! Affine transformations for image geometry
//!
//! This module provides functions for affine transformations
//! such as translation, rotation, scaling, and shearing.

use crate::error::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgba};
use ndarray::{Array1, Array2};

/// 2D Affine transformation matrix (2x3)
#[derive(Debug, Clone)]
pub struct AffineTransform {
    /// Transformation matrix
    pub matrix: Array2<f64>,
}

impl AffineTransform {
    /// Create a new affine transformation matrix from raw data
    pub fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64) -> Self {
        let matrix = Array2::from_shape_vec((2, 3), vec![a, b, c, d, e, f]).unwrap();
        Self { matrix }
    }

    /// Create identity transformation
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    }

    /// Create translation transformation
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self::new(1.0, 0.0, tx, 0.0, 1.0, ty)
    }

    /// Create scaling transformation
    pub fn scaling(sx: f64, sy: f64) -> Self {
        Self::new(sx, 0.0, 0.0, 0.0, sy, 0.0)
    }

    /// Create rotation transformation
    pub fn rotation(angle: f64) -> Self {
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        Self::new(cos_angle, -sin_angle, 0.0, sin_angle, cos_angle, 0.0)
    }

    /// Create shearing transformation
    pub fn shearing(shx: f64, shy: f64) -> Self {
        Self::new(1.0, shx, 0.0, shy, 1.0, 0.0)
    }

    /// Compose with another transformation
    pub fn compose(&self, other: &Self) -> Self {
        // Extended matrices for composition (with implicit homogeneous coordinate)
        let mut result = Array2::zeros((2, 3));

        // Matrix multiplication
        for i in 0..2 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..2 {
                    sum += self.matrix[[i, k]] * other.matrix[[k, j]];
                }
                if j == 2 {
                    sum += self.matrix[[i, 2]]; // Translation component
                }
                result[[i, j]] = sum;
            }
        }

        Self { matrix: result }
    }

    /// Get inverse transformation
    pub fn inverse(&self) -> Result<Self> {
        // Extract components
        let a = self.matrix[[0, 0]];
        let b = self.matrix[[0, 1]];
        let c = self.matrix[[0, 2]];
        let d = self.matrix[[1, 0]];
        let e = self.matrix[[1, 1]];
        let f = self.matrix[[1, 2]];

        // Calculate determinant
        let det = a * e - b * d;

        if det.abs() < 1e-10 {
            return Err(crate::error::VisionError::OperationFailed(
                "Affine transformation is singular, cannot compute inverse".to_string(),
            ));
        }

        // Calculate inverse
        let inv_det = 1.0 / det;
        let a_inv = e * inv_det;
        let b_inv = -b * inv_det;
        let c_inv = (b * f - e * c) * inv_det;
        let d_inv = -d * inv_det;
        let e_inv = a * inv_det;
        let f_inv = (d * c - a * f) * inv_det;

        Ok(Self::new(a_inv, b_inv, c_inv, d_inv, e_inv, f_inv))
    }

    /// Apply transformation to a point
    pub fn transform_point(&self, x: f64, y: f64) -> (f64, f64) {
        let x_new = self.matrix[[0, 0]] * x + self.matrix[[0, 1]] * y + self.matrix[[0, 2]];
        let y_new = self.matrix[[1, 0]] * x + self.matrix[[1, 1]] * y + self.matrix[[1, 2]];

        (x_new, y_new)
    }
}

/// Warp an image using an affine transformation
///
/// # Arguments
///
/// * `src` - Source image
/// * `transform` - Affine transformation
/// * `width` - Output image width (if None, uses source width)
/// * `height` - Output image height (if None, uses source height)
/// * `border_mode` - Border handling mode
///
/// # Returns
///
/// * Result containing warped image
pub fn warp_affine(
    src: &DynamicImage,
    transform: &AffineTransform,
    width: Option<u32>,
    height: Option<u32>,
    border_mode: BorderMode,
) -> Result<DynamicImage> {
    let width = width.unwrap_or_else(|| src.width());
    let height = height.unwrap_or_else(|| src.height());

    let mut dst = ImageBuffer::new(width, height);

    // Compute inverse transform for backward mapping
    let inverse_transform = transform.inverse()?;

    // Warp image using inverse mapping
    for y in 0..height {
        for x in 0..width {
            // Apply inverse transform to get source coordinate
            let (src_x, src_y) = inverse_transform.transform_point(x as f64, y as f64);

            // Sample pixel using interpolation
            if let Some(color) = sample_pixel(src, src_x, src_y, border_mode) {
                dst.put_pixel(x, y, color);
            }
        }
    }

    Ok(DynamicImage::ImageRgba8(dst))
}

/// Border handling modes for image warping
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode {
    /// Transparent border
    Transparent,
    /// Replicate border pixels
    Replicate,
    /// Reflect border pixels
    Reflect,
    /// Wrap around
    Wrap,
    /// Constant color border
    Constant(Rgba<u8>),
}

/// Sample a pixel from an image using bilinear interpolation
fn sample_pixel(src: &DynamicImage, x: f64, y: f64, border_mode: BorderMode) -> Option<Rgba<u8>> {
    let width = src.width() as f64;
    let height = src.height() as f64;

    // Handle border cases
    let (x_adj, y_adj) = match border_mode {
        BorderMode::Transparent => {
            // Return None for out-of-bounds pixels
            if x < 0.0 || x >= width || y < 0.0 || y >= height {
                return None;
            }
            (x, y)
        }
        BorderMode::Replicate => {
            let x_clamped = x.max(0.0).min(width - 1.0);
            let y_clamped = y.max(0.0).min(height - 1.0);
            (x_clamped, y_clamped)
        }
        BorderMode::Reflect => {
            let x_reflected = if x < 0.0 {
                -x
            } else if x >= width {
                2.0 * width - x - 2.0
            } else {
                x
            };

            let y_reflected = if y < 0.0 {
                -y
            } else if y >= height {
                2.0 * height - y - 2.0
            } else {
                y
            };

            (x_reflected, y_reflected)
        }
        BorderMode::Wrap => {
            let x_wrapped = ((x % width) + width) % width;
            let y_wrapped = ((y % height) + height) % height;
            (x_wrapped, y_wrapped)
        }
        BorderMode::Constant(color) => {
            if x < 0.0 || x >= width || y < 0.0 || y >= height {
                return Some(color);
            }
            (x, y)
        }
    };

    // Bilinear interpolation
    let x0 = x_adj.floor() as u32;
    let y0 = y_adj.floor() as u32;
    let x1 = (x0 + 1).min(src.width() - 1);
    let y1 = (y0 + 1).min(src.height() - 1);

    let sx = x_adj - x0 as f64;
    let sy = y_adj - y0 as f64;

    let p00 = src.get_pixel(x0, y0).to_rgba();
    let p01 = src.get_pixel(x0, y1).to_rgba();
    let p10 = src.get_pixel(x1, y0).to_rgba();
    let p11 = src.get_pixel(x1, y1).to_rgba();

    // Interpolate
    let mut result = [0u8; 4];
    for c in 0..4 {
        let c00 = p00[c] as f64;
        let c01 = p01[c] as f64;
        let c10 = p10[c] as f64;
        let c11 = p11[c] as f64;

        let val = (1.0 - sx) * (1.0 - sy) * c00
            + sx * (1.0 - sy) * c10
            + (1.0 - sx) * sy * c01
            + sx * sy * c11;

        result[c] = val.round().clamp(0.0, 255.0) as u8;
    }

    Some(Rgba(result))
}

/// Estimate affine transformation between two sets of points using least squares
///
/// # Arguments
///
/// * `src_points` - Source points (at least 3)
/// * `dst_points` - Destination points (same length as src_points)
///
/// # Returns
///
/// * Result containing estimated affine transformation
pub fn estimate_affine_transform(
    src_points: &[(f64, f64)],
    dst_points: &[(f64, f64)],
) -> Result<AffineTransform> {
    if src_points.len() != dst_points.len() {
        return Err(crate::error::VisionError::InvalidParameter(
            "Source and destination point sets must have the same length".to_string(),
        ));
    }

    if src_points.len() < 3 {
        return Err(crate::error::VisionError::InvalidParameter(
            "At least 3 point correspondences are required".to_string(),
        ));
    }

    // Construct linear system for the least squares problem
    let n = src_points.len();
    let mut a = Array2::zeros((2 * n, 6));
    let mut b = Array1::zeros(2 * n);

    for (i, (src, dst)) in src_points.iter().zip(dst_points.iter()).enumerate() {
        let (x, y) = *src;
        let (u, v) = *dst;

        // First row for x coordinate
        a[[i * 2, 0]] = x;
        a[[i * 2, 1]] = y;
        a[[i * 2, 2]] = 1.0;
        b[i * 2] = u;

        // Second row for y coordinate
        a[[i * 2 + 1, 3]] = x;
        a[[i * 2 + 1, 4]] = y;
        a[[i * 2 + 1, 5]] = 1.0;
        b[i * 2 + 1] = v;
    }

    // Solve using least squares (pseudo-inverse)
    let solution = solve_least_squares(&a, &b)?;

    // Construct affine transformation matrix
    let transform = AffineTransform::new(
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
    );

    Ok(transform)
}

/// Solve linear least squares problem using normal equations
fn solve_least_squares(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let (m, n) = a.dim();

    if m < n {
        return Err(crate::error::VisionError::InvalidParameter(
            "Underconstrained system (fewer equations than unknowns)".to_string(),
        ));
    }

    if b.len() != m {
        return Err(crate::error::VisionError::InvalidParameter(
            "Dimensions of A and b do not match".to_string(),
        ));
    }

    // Compute A^T * A
    let mut ata = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += a[[k, i]] * a[[k, j]];
            }
            ata[[i, j]] = sum;
        }
    }

    // Compute A^T * b
    let mut atb = Array1::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for k in 0..m {
            sum += a[[k, i]] * b[k];
        }
        atb[i] = sum;
    }

    // Solve the normal equation (A^T * A) * x = A^T * b
    // Simplified Gaussian elimination
    solve_linear_system(&ata, &atb)
}

/// Solve linear system Ax = b
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let (n, m) = a.dim();
    if n != m {
        return Err(crate::error::VisionError::InvalidParameter(
            "Matrix A must be square".to_string(),
        ));
    }

    if b.len() != n {
        return Err(crate::error::VisionError::InvalidParameter(
            "Dimensions of A and b do not match".to_string(),
        ));
    }

    // Gaussian elimination with pivoting
    let mut a_copy = a.clone();
    let mut b_copy = b.clone();

    for i in 0..n {
        // Find pivot
        let mut max_idx = i;
        let mut max_val = a_copy[[i, i]].abs();

        for j in (i + 1)..n {
            let val = a_copy[[j, i]].abs();
            if val > max_val {
                max_idx = j;
                max_val = val;
            }
        }

        if max_val < 1e-10 {
            return Err(crate::error::VisionError::OperationFailed(
                "Matrix is singular or nearly singular".to_string(),
            ));
        }

        // Swap rows if needed
        if max_idx != i {
            for j in 0..n {
                let temp = a_copy[[i, j]];
                a_copy[[i, j]] = a_copy[[max_idx, j]];
                a_copy[[max_idx, j]] = temp;
            }

            let temp = b_copy[i];
            b_copy[i] = b_copy[max_idx];
            b_copy[max_idx] = temp;
        }

        // Eliminate below
        for j in (i + 1)..n {
            let factor = a_copy[[j, i]] / a_copy[[i, i]];

            for k in i..n {
                a_copy[[j, k]] -= factor * a_copy[[i, k]];
            }

            b_copy[j] -= factor * b_copy[i];
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);

    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += a_copy[[i, j]] * x[j];
        }

        x[i] = (b_copy[i] - sum) / a_copy[[i, i]];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_identity() {
        let transform = AffineTransform::identity();

        // Identity should not transform points
        let (x, y) = (10.0, 20.0);
        let (x_transformed, y_transformed) = transform.transform_point(x, y);

        assert!((x_transformed - x).abs() < 1e-10);
        assert!((y_transformed - y).abs() < 1e-10);
    }

    #[test]
    fn test_affine_translation() {
        let transform = AffineTransform::translation(5.0, 7.0);

        let (x, y) = (10.0, 20.0);
        let (x_transformed, y_transformed) = transform.transform_point(x, y);

        assert!((x_transformed - (x + 5.0)).abs() < 1e-10);
        assert!((y_transformed - (y + 7.0)).abs() < 1e-10);
    }

    #[test]
    fn test_affine_scaling() {
        let transform = AffineTransform::scaling(2.0, 0.5);

        let (x, y) = (10.0, 20.0);
        let (x_transformed, y_transformed) = transform.transform_point(x, y);

        assert!((x_transformed - 2.0 * x).abs() < 1e-10);
        assert!((y_transformed - 0.5 * y).abs() < 1e-10);
    }

    #[test]
    fn test_affine_rotation() {
        // 90 degree rotation
        let transform = AffineTransform::rotation(std::f64::consts::PI / 2.0);

        let (x, y) = (10.0, 0.0);
        let (x_transformed, y_transformed) = transform.transform_point(x, y);

        assert!((x_transformed - 0.0).abs() < 1e-10);
        assert!((y_transformed - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_affine_composition() {
        let translate = AffineTransform::translation(5.0, 7.0);
        let scale = AffineTransform::scaling(2.0, 0.5);

        // Translate then scale
        let combined = scale.compose(&translate);

        let (x, y) = (10.0, 20.0);
        let (x_transformed, y_transformed) = combined.transform_point(x, y);

        assert!((x_transformed - 2.0 * (x + 5.0)).abs() < 1e-10);
        assert!((y_transformed - 0.5 * (y + 7.0)).abs() < 1e-10);
    }

    #[test]
    fn test_affine_inverse() {
        let transform = AffineTransform::new(2.0, 1.0, 3.0, 0.5, 3.0, 2.0);
        let inverse = transform.inverse().unwrap();

        // Apply transform then inverse should give original point
        let (x, y) = (10.0, 20.0);
        let (x_transformed, y_transformed) = transform.transform_point(x, y);
        let (x_recovered, y_recovered) = inverse.transform_point(x_transformed, y_transformed);

        assert!((x_recovered - x).abs() < 1e-10);
        assert!((y_recovered - y).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_affine_transform() {
        // Source points
        let src_points = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];

        // Destination points (apply scaling + rotation + translation)
        let angle = std::f64::consts::PI / 4.0; // 45 degrees
        let scale_x = 2.0;
        let scale_y = 1.5;
        let tx = 10.0;
        let ty = 5.0;

        let cos_angle = angle.cos();
        let sin_angle = angle.sin();

        let dst_points: Vec<_> = src_points
            .iter()
            .map(|&(x, y)| {
                let x_rot = cos_angle * x - sin_angle * y;
                let y_rot = sin_angle * x + cos_angle * y;

                (scale_x * x_rot + tx, scale_y * y_rot + ty)
            })
            .collect();

        // Estimate transformation
        let transform = estimate_affine_transform(&src_points, &dst_points).unwrap();

        // Check that the estimated transform correctly maps points
        for (i, &(x, y)) in src_points.iter().enumerate() {
            let (x_transformed, y_transformed) = transform.transform_point(x, y);
            let (x_expected, y_expected) = dst_points[i];

            assert!((x_transformed - x_expected).abs() < 1e-10);
            assert!((y_transformed - y_expected).abs() < 1e-10);
        }
    }
}
