//! Non-rigid transformation methods for image warping
//!
//! This module provides functions for non-rigid image transformations
//! such as thin-plate splines and elastic deformations. These transformations
//! allow for local deformations that are not possible with global transformations
//! like affine or perspective transforms.

use crate::error::{Result, VisionError};
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgba};
use ndarray::{Array1, Array2};
// SVD is used in solver imports
use rand::distr::Uniform;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::f64::consts::PI;

/// Non-rigid transformation interface
pub trait NonRigidTransform {
    /// Transform a point using this non-rigid transformation
    ///
    /// # Arguments
    ///
    /// * `point` - The point to transform (x, y)
    ///
    /// # Returns
    ///
    /// * The transformed point (x', y')
    fn transform_point(&self, point: (f64, f64)) -> (f64, f64);
}

/// Thin-Plate Spline transformation
///
/// This implements a smooth deformation that minimizes bending energy
/// while interpolating between control points.
#[derive(Debug, Clone)]
pub struct ThinPlateSpline {
    /// Control points in source image
    source_points: Vec<(f64, f64)>,

    /// Control points in target image
    target_points: Vec<(f64, f64)>,

    /// Coefficients for x-coordinate mapping
    coef_x: Array1<f64>,

    /// Coefficients for y-coordinate mapping
    coef_y: Array1<f64>,

    /// Regularization parameter (lambda)
    lambda: f64,
}

impl ThinPlateSpline {
    /// Create a new thin-plate spline transformation
    ///
    /// # Arguments
    ///
    /// * `source_points` - Control points in the source image
    /// * `target_points` - Corresponding control points in the target image
    /// * `lambda` - Regularization parameter (defaults to 0.0)
    ///
    /// # Returns
    ///
    /// * Result containing the thin-plate spline transformation
    pub fn new(
        source_points: &[(f64, f64)],
        target_points: &[(f64, f64)],
        lambda: Option<f64>,
    ) -> Result<Self> {
        if source_points.len() != target_points.len() {
            return Err(VisionError::InvalidParameter(
                "Source and target point sets must have the same length".to_string(),
            ));
        }

        if source_points.len() < 3 {
            return Err(VisionError::InvalidParameter(
                "At least 3 control point pairs are required".to_string(),
            ));
        }

        let lambda = lambda.unwrap_or(0.0);
        let n = source_points.len();

        // Compute the TPS kernel matrix
        let mut k = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist_squared = squared_distance(source_points[i], source_points[j]);
                    // U(r) = r^2 * log(r^2), where r^2 is squared distance
                    k[[i, j]] = if dist_squared > 0.0 {
                        dist_squared * dist_squared.ln()
                    } else {
                        0.0
                    };
                }
            }
        }

        // Create the L matrix (n+3 x n+3)
        let mut l = Array2::zeros((n + 3, n + 3));

        // Copy K to the upper left
        for i in 0..n {
            for j in 0..n {
                l[[i, j]] = k[[i, j]];

                // Add regularization to the diagonal
                if i == j {
                    l[[i, j]] += lambda;
                }
            }
        }

        // Fill in P parts (affine components)
        for i in 0..n {
            l[[i, n]] = 1.0;
            l[[i, n + 1]] = source_points[i].0;
            l[[i, n + 2]] = source_points[i].1;

            l[[n, i]] = 1.0;
            l[[n + 1, i]] = source_points[i].0;
            l[[n + 2, i]] = source_points[i].1;
        }

        // Extract x and y coordinates from target points
        let mut target_x = Array1::zeros(n + 3);
        let mut target_y = Array1::zeros(n + 3);

        for i in 0..n {
            target_x[i] = target_points[i].0;
            target_y[i] = target_points[i].1;
        }

        // Solve the linear system for x and y mappings
        use ndarray_linalg::solve::Solve;
        
        let coef_x = Solve::solve(&l, &target_x).map_err(|e| {
            VisionError::LinAlgError(format!("Failed to solve for x coefficients: {}", e))
        })?;

        let coef_y = Solve::solve(&l, &target_y).map_err(|e| {
            VisionError::LinAlgError(format!("Failed to solve for y coefficients: {}", e))
        })?;

        Ok(Self {
            source_points: source_points.to_vec(),
            target_points: target_points.to_vec(),
            coef_x,
            coef_y,
            lambda,
        })
    }

    /// Get the regularization parameter
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Get a reference to the source control points
    pub fn source_points(&self) -> &[(f64, f64)] {
        &self.source_points
    }

    /// Get a reference to the target control points
    pub fn target_points(&self) -> &[(f64, f64)] {
        &self.target_points
    }
}

impl NonRigidTransform for ThinPlateSpline {
    fn transform_point(&self, point: (f64, f64)) -> (f64, f64) {
        let n = self.source_points.len();
        let (x, y) = point;

        // Initialize with affine component
        let mut x_new = self.coef_x[n] + self.coef_x[n + 1] * x + self.coef_x[n + 2] * y;
        let mut y_new = self.coef_y[n] + self.coef_y[n + 1] * x + self.coef_y[n + 2] * y;

        // Add the non-linear part from each control point
        for i in 0..n {
            let dist_squared = squared_distance(point, self.source_points[i]);
            let basis_value = if dist_squared > 0.0 {
                dist_squared * dist_squared.ln()
            } else {
                0.0
            };

            x_new += self.coef_x[i] * basis_value;
            y_new += self.coef_y[i] * basis_value;
        }

        (x_new, y_new)
    }
}

/// Generate a grid of control points for the entire image
///
/// # Arguments
///
/// * `width` - Image width
/// * `height` - Image height
/// * `x_count` - Number of points in x direction
/// * `y_count` - Number of points in y direction
///
/// # Returns
///
/// * Vector of control points (x, y)
pub fn generate_grid_points(
    width: u32,
    height: u32,
    x_count: u32,
    y_count: u32,
) -> Vec<(f64, f64)> {
    let width = width as f64;
    let height = height as f64;
    let mut points = Vec::with_capacity((x_count * y_count) as usize);

    // Add corner points
    points.push((0.0, 0.0));
    points.push((width - 1.0, 0.0));
    points.push((width - 1.0, height - 1.0));
    points.push((0.0, height - 1.0));

    // Add inner grid points
    for y in 1..y_count - 1 {
        let y_pos = y as f64 * height / (y_count - 1) as f64;
        for x in 1..x_count - 1 {
            let x_pos = x as f64 * width / (x_count - 1) as f64;
            points.push((x_pos, y_pos));
        }
    }

    points
}

/// Elastic deformation transformation
///
/// This implements a random displacement field for elastic-like deformations.
#[derive(Debug, Clone)]
pub struct ElasticDeformation {
    /// Displacement field for x coordinates
    dx_map: Array2<f64>,

    /// Displacement field for y coordinates
    dy_map: Array2<f64>,

    /// Image width
    width: u32,

    /// Image height
    height: u32,
}

impl ElasticDeformation {
    /// Create a new elastic deformation with random displacements
    ///
    /// # Arguments
    ///
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `alpha` - Scaling factor for displacements (higher values = more deformation)
    /// * `sigma` - Smoothing factor (higher values = smoother deformation)
    /// * `seed` - Optional random seed for reproducibility
    ///
    /// # Returns
    ///
    /// * Result containing the elastic deformation
    pub fn new(width: u32, height: u32, alpha: f64, sigma: f64, seed: Option<u64>) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(VisionError::InvalidParameter(
                "Width and height must be positive".to_string(),
            ));
        }

        if alpha <= 0.0 || sigma <= 0.0 {
            return Err(VisionError::InvalidParameter(
                "Alpha and sigma must be positive".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = if let Some(seed_value) = seed {
            StdRng::seed_from_u64(seed_value)
        } else {
            // According to CLAUDE.md, should use rng() instead of thread_rng() for rand 0.9.0
            let mut thread_rng = rand::rng();
            StdRng::from_rng(&mut thread_rng).expect("Failed to create StdRng from ThreadRng")
        };

        // Generate random displacement fields
        let dist = Uniform::new(-1.0, 1.0).unwrap();
        let mut dx_map = Array2::zeros((height as usize, width as usize));
        let mut dy_map = Array2::zeros((height as usize, width as usize));

        for y in 0..height as usize {
            for x in 0..width as usize {
                dx_map[[y, x]] = rng.sample(dist);
                dy_map[[y, x]] = rng.sample(dist);
            }
        }

        // Apply Gaussian filter to smooth the displacement fields
        dx_map = gaussian_filter(&dx_map, sigma)?;
        dy_map = gaussian_filter(&dy_map, sigma)?;

        // Scale the displacement fields
        dx_map.mapv_inplace(|v| v * alpha);
        dy_map.mapv_inplace(|v| v * alpha);

        Ok(Self {
            dx_map,
            dy_map,
            width,
            height,
        })
    }
}

impl NonRigidTransform for ElasticDeformation {
    fn transform_point(&self, point: (f64, f64)) -> (f64, f64) {
        let (x, y) = point;

        // Check bounds
        if x < 0.0 || y < 0.0 || x >= self.width as f64 || y >= self.height as f64 {
            return point;
        }

        // Bilinearly interpolate displacement values
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width as usize - 1);
        let y1 = (y0 + 1).min(self.height as usize - 1);

        let dx = x - x0 as f64;
        let dy = y - y0 as f64;

        // Interpolate x displacement
        let dx00 = self.dx_map[[y0, x0]];
        let dx01 = self.dx_map[[y1, x0]];
        let dx10 = self.dx_map[[y0, x1]];
        let dx11 = self.dx_map[[y1, x1]];

        let dx_interp = (1.0 - dx) * (1.0 - dy) * dx00
            + dx * (1.0 - dy) * dx10
            + (1.0 - dx) * dy * dx01
            + dx * dy * dx11;

        // Interpolate y displacement
        let dy00 = self.dy_map[[y0, x0]];
        let dy01 = self.dy_map[[y1, x0]];
        let dy10 = self.dy_map[[y0, x1]];
        let dy11 = self.dy_map[[y1, x1]];

        let dy_interp = (1.0 - dx) * (1.0 - dy) * dy00
            + dx * (1.0 - dy) * dy10
            + (1.0 - dx) * dy * dy01
            + dx * dy * dy11;

        // Apply displacement
        (x + dx_interp, y + dy_interp)
    }
}

/// Apply a non-rigid transformation to an image
///
/// # Arguments
///
/// * `src` - Source image
/// * `transform` - Non-rigid transformation
/// * `border_mode` - How to handle pixels outside image boundaries
///
/// # Returns
///
/// * Result containing the transformed image
pub fn warp_non_rigid(
    src: &DynamicImage,
    transform: &impl NonRigidTransform,
    border_mode: crate::transform::perspective::BorderMode,
) -> Result<DynamicImage> {
    let (src_width, src_height) = src.dimensions();

    // Create output image with same dimensions
    let mut dst: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(src_width, src_height);

    // For each pixel in destination image, find corresponding pixel in source
    for y in 0..src_height {
        for x in 0..src_width {
            // Get source coordinates by inverse mapping
            let dst_pt = (x as f64, y as f64);
            let (src_x, src_y) = transform.transform_point(dst_pt);

            // Check if source point is inside image bounds
            if src_x >= 0.0 && src_x < src_width as f64 && src_y >= 0.0 && src_y < src_height as f64
            {
                // Bilinear interpolation for smoother results
                let color = crate::transform::perspective::bilinear_interpolate(src, src_x, src_y);
                dst.put_pixel(x, y, color);
            } else {
                // Handle out-of-bounds pixels according to border mode
                match border_mode {
                    crate::transform::perspective::BorderMode::Constant(color) => {
                        dst.put_pixel(x, y, color);
                    }
                    crate::transform::perspective::BorderMode::Reflect => {
                        // Reflect coordinates at image boundaries
                        let reflected_x =
                            crate::transform::perspective::reflect_coordinate(src_x, src_width as f64);
                        let reflected_y =
                            crate::transform::perspective::reflect_coordinate(src_y, src_height as f64);
                        let color =
                            crate::transform::perspective::bilinear_interpolate(src, reflected_x, reflected_y);
                        dst.put_pixel(x, y, color);
                    }
                    crate::transform::perspective::BorderMode::Replicate => {
                        // Clamp coordinates to image boundaries
                        let clamped_x = src_x.max(0.0).min(src_width as f64 - 1.0);
                        let clamped_y = src_y.max(0.0).min(src_height as f64 - 1.0);
                        let color =
                            crate::transform::perspective::bilinear_interpolate(src, clamped_x, clamped_y);
                        dst.put_pixel(x, y, color);
                    }
                    crate::transform::perspective::BorderMode::Wrap => {
                        // Wrap coordinates around image boundaries
                        let wrapped_x = crate::transform::perspective::modulo(src_x, src_width as f64);
                        let wrapped_y = crate::transform::perspective::modulo(src_y, src_height as f64);
                        let color =
                            crate::transform::perspective::bilinear_interpolate(src, wrapped_x, wrapped_y);
                        dst.put_pixel(x, y, color);
                    }
                    crate::transform::perspective::BorderMode::Transparent => {
                        // Set transparent pixel (alpha = 0)
                        dst.put_pixel(x, y, Rgba([0, 0, 0, 0]));
                    }
                }
            }
        }
    }

    Ok(DynamicImage::ImageRgba8(dst))
}

/// Apply a thin-plate spline transformation to an image
///
/// # Arguments
///
/// * `src` - Source image
/// * `source_points` - Control points in the source image
/// * `target_points` - Corresponding control points in the target image
/// * `lambda` - Regularization parameter (defaults to 0.0)
/// * `border_mode` - How to handle pixels outside image boundaries
///
/// # Returns
///
/// * Result containing the warped image
pub fn warp_thin_plate_spline(
    src: &DynamicImage,
    source_points: Vec<(f64, f64)>,
    target_points: Vec<(f64, f64)>,
    lambda: Option<f64>,
    border_mode: crate::transform::perspective::BorderMode,
) -> Result<DynamicImage> {
    // Create the thin-plate spline transformation
    let tps = ThinPlateSpline::new(&source_points, &target_points, lambda)?;

    // Apply the transformation
    warp_non_rigid(src, &tps, border_mode)
}

/// Apply an elastic deformation to an image
///
/// # Arguments
///
/// * `src` - Source image
/// * `alpha` - Scaling factor for displacements (higher values = more deformation)
/// * `sigma` - Smoothing factor (higher values = smoother deformation)
/// * `seed` - Optional random seed for reproducibility
/// * `border_mode` - How to handle pixels outside image boundaries
///
/// # Returns
///
/// * Result containing the warped image
pub fn warp_elastic(
    src: &DynamicImage,
    alpha: f64,
    sigma: f64,
    seed: Option<u64>,
    border_mode: crate::transform::perspective::BorderMode,
) -> Result<DynamicImage> {
    let (width, height) = src.dimensions();

    // Create the elastic deformation
    let elastic = ElasticDeformation::new(width, height, alpha, sigma, seed)?;

    // Apply the transformation
    warp_non_rigid(src, &elastic, border_mode)
}

/// Apply a face warping effect (fun example of thin-plate splines)
///
/// # Arguments
///
/// * `src` - Source image
/// * `feature_points` - Feature points of the face (eyes, nose, mouth, etc.)
/// * `warp_type` - Warp type (e.g., "smile", "surprise", "narrow", etc.)
/// * `intensity` - Intensity of the warping effect (0.0-1.0)
/// * `border_mode` - How to handle pixels outside image boundaries
///
/// # Returns
///
/// * Result containing the warped image
pub fn warp_face(
    src: &DynamicImage,
    _feature_points: Vec<(f64, f64)>,
    _warp_type: &str,
    _intensity: f64,
    _border_mode: crate::transform::perspective::BorderMode,
) -> Result<DynamicImage> {
    // This is a placeholder implementation
    // A real implementation would use the feature points and warp type to
    // create appropriate target points for the thin-plate spline

    // For now, just return the original image
    Ok(src.clone())
}

// Helper functions

/// Calculate squared Euclidean distance between two points
///
/// # Arguments
///
/// * `p1` - First point
/// * `p2` - Second point
///
/// # Returns
///
/// * The squared distance between the points
fn squared_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    let (x1, y1) = p1;
    let (x2, y2) = p2;

    (x2 - x1).powi(2) + (y2 - y1).powi(2)
}

/// Apply a Gaussian filter to a 2D array
///
/// # Arguments
///
/// * `input` - Input array
/// * `sigma` - Standard deviation of the Gaussian kernel
///
/// # Returns
///
/// * Result containing the filtered array
fn gaussian_filter(input: &Array2<f64>, sigma: f64) -> Result<Array2<f64>> {
    let (height, width) = input.dim();

    // Determine kernel size (odd number, approximately 3*sigma on each side)
    let kernel_radius = (3.0 * sigma).ceil() as usize;
    let kernel_size = 2 * kernel_radius + 1;

    // Create 1D Gaussian kernel
    let mut kernel = Array1::zeros(kernel_size);
    let scale = 1.0 / (2.0 * PI * sigma * sigma).sqrt();

    for i in 0..kernel_size {
        let x = (i as isize - kernel_radius as isize) as f64;
        kernel[i] = scale * (-x * x / (2.0 * sigma * sigma)).exp();
    }

    // Normalize kernel
    let kernel_sum = kernel.sum();
    kernel.mapv_inplace(|v| v / kernel_sum);

    // Apply separable convolution (first rows, then columns)

    // First pass: convolve rows
    let mut temp = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for k in 0..kernel_size {
                let offset = k as isize - kernel_radius as isize;
                let x_pos = x as isize + offset;

                if x_pos >= 0 && x_pos < width as isize {
                    let kernel_value = kernel[k];
                    sum += input[[y, x_pos as usize]] * kernel_value;
                    weight_sum += kernel_value;
                }
            }

            temp[[y, x]] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };
        }
    }

    // Second pass: convolve columns
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;

            for k in 0..kernel_size {
                let offset = k as isize - kernel_radius as isize;
                let y_pos = y as isize + offset;

                if y_pos >= 0 && y_pos < height as isize {
                    let kernel_value = kernel[k];
                    sum += temp[[y_pos as usize, x]] * kernel_value;
                    weight_sum += kernel_value;
                }
            }

            output[[y, x]] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Pixel, Rgb, RgbImage};

    #[test]
    fn test_thin_plate_spline_identity() {
        // Create control points in a grid (identity mapping)
        let source_points = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
            (50.0, 50.0),
        ];

        // Target points are identical (no transformation)
        let target_points = source_points.clone();

        // Create the thin-plate spline
        let tps = ThinPlateSpline::new(&source_points, &target_points, None).unwrap();

        // Test some points
        let test_points = [(25.0, 25.0), (75.0, 75.0), (10.0, 90.0)];

        for point in &test_points {
            let transformed = tps.transform_point(*point);

            // Points should remain unchanged (within numerical precision)
            assert!((transformed.0 - point.0).abs() < 1e-10);
            assert!((transformed.1 - point.1).abs() < 1e-10);
        }
    }

    #[test]
    fn test_thin_plate_spline_interpolation() {
        // Create control points
        let source_points = vec![
            (0.0, 0.0),
            (100.0, 0.0),
            (0.0, 100.0),
            (100.0, 100.0),
            (50.0, 50.0),
        ];

        // Target points move the center point
        let mut target_points = source_points.clone();
        target_points[4] = (60.0, 40.0); // Move the center point

        // Create the thin-plate spline
        let tps = ThinPlateSpline::new(&source_points, &target_points, None).unwrap();

        // Corner points should map exactly
        for i in 0..4 {
            let transformed = tps.transform_point(source_points[i]);
            assert!((transformed.0 - target_points[i].0).abs() < 1e-10);
            assert!((transformed.1 - target_points[i].1).abs() < 1e-10);
        }

        // Center point should map exactly
        let center_transformed = tps.transform_point(source_points[4]);
        assert!((center_transformed.0 - target_points[4].0).abs() < 1e-10);
        assert!((center_transformed.1 - target_points[4].1).abs() < 1e-10);

        // Points in between should be smoothly interpolated
        // (we don't test exact values, just verify they're reasonable)
        let mid_point = (25.0, 25.0);
        let transformed = tps.transform_point(mid_point);

        // Should be shifted a bit toward the new center
        assert!(transformed.0 > mid_point.0);
        assert!(transformed.1 < mid_point.1);
    }

    #[test]
    fn test_elastic_deformation() {
        // Create a consistent elastic deformation with a fixed seed
        let width = 100;
        let height = 100;
        let alpha = 10.0; // Significant deformation
        let sigma = 10.0; // Smooth deformation
        let seed = Some(42); // Fixed seed for reproducibility

        let elastic = ElasticDeformation::new(width, height, alpha, sigma, seed).unwrap();

        // Test that points are moved
        let original_point = (50.0, 50.0);
        let transformed = elastic.transform_point(original_point);

        // Point should be different but within reasonable bounds
        assert!(transformed.0 != original_point.0 || transformed.1 != original_point.1);
        assert!(transformed.0 >= 30.0 && transformed.0 <= 70.0);
        assert!(transformed.1 >= 30.0 && transformed.1 <= 70.0);

        // Test reproducibility with same seed
        let elastic2 = ElasticDeformation::new(width, height, alpha, sigma, seed).unwrap();
        let transformed2 = elastic2.transform_point(original_point);

        // Should get exactly the same result with the same seed
        assert!((transformed.0 - transformed2.0).abs() < 1e-10);
        assert!((transformed.1 - transformed2.1).abs() < 1e-10);
    }

    #[test]
    fn test_warp_non_rigid() {
        // Create a simple test image with a pattern
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

        // Create a thin-plate spline that shifts the center of the image
        let source_points = vec![
            (0.0, 0.0),
            (width as f64 - 1.0, 0.0),
            (0.0, height as f64 - 1.0),
            (width as f64 - 1.0, height as f64 - 1.0),
            (width as f64 / 2.0, height as f64 / 2.0),
        ];

        let mut target_points = source_points.clone();
        target_points[4] = (width as f64 / 2.0 + 20.0, height as f64 / 2.0); // Shift center to the right

        // Create and apply transformation  
        let tps = ThinPlateSpline::new(&source_points, &target_points, None).unwrap();
        let result = warp_non_rigid(&src, &tps, crate::transform::perspective::BorderMode::default()).unwrap();

        // Check that dimensions are the same
        assert_eq!(result.width(), width);
        assert_eq!(result.height(), height);

        // Check that corners are unchanged
        for &(x, y) in &[
            (0.0, 0.0),
            (width as f64 - 1.0, 0.0),
            (0.0, height as f64 - 1.0),
            (width as f64 - 1.0, height as f64 - 1.0),
        ] {
            let src_color = src.get_pixel(x as u32, y as u32).to_rgb();
            let dst_color = result.get_pixel(x as u32, y as u32).to_rgb();

            // Colors should be identical at corners
            assert_eq!(src_color, dst_color);
        }

        // Check that the center has moved
        // The red diagonal should no longer pass through the center
        let center_x = width / 2;
        let center_y = height / 2;
        let center_color = result.get_pixel(center_x as u32, center_y as u32).to_rgb();

        // Center should no longer be red (diagonal has shifted)
        assert_ne!(center_color[0], 255);
    }
}
