//! Image warping and resampling functions
//!
//! This module provides functionality for transforming images using various
//! interpolation methods and geometric transformations.

use crate::error::{Result, VisionError};
use crate::registration::{transform_point, Point2D, TransformMatrix};
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage};
use ndarray::Array2;

/// Interpolation method for image resampling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Nearest neighbor interpolation
    NearestNeighbor,
    /// Bilinear interpolation
    Bilinear,
    /// Bicubic interpolation
    Bicubic,
}

/// Boundary handling method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryMethod {
    /// Use zero values outside image bounds
    Zero,
    /// Use constant value outside image bounds
    Constant(f32),
    /// Reflect values at image boundaries
    Reflect,
    /// Wrap around at image boundaries
    Wrap,
    /// Clamp to edge values
    Clamp,
}

/// Warp a grayscale image using a transformation matrix
///
/// # Arguments
///
/// * `image` - Input grayscale image
/// * `transform` - 3x3 transformation matrix
/// * `output_size` - Output image dimensions (width, height)
/// * `interpolation` - Interpolation method
/// * `boundary` - Boundary handling method
///
/// # Returns
///
/// * Result containing the warped image
pub fn warp_image(
    image: &GrayImage,
    transform: &TransformMatrix,
    output_size: (u32, u32),
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
) -> Result<GrayImage> {
    let (out_width, out_height) = output_size;
    let (in_width, in_height) = image.dimensions();

    // Create output image
    let mut output = GrayImage::new(out_width, out_height);

    // Invert transformation for backwards mapping
    use ndarray_linalg::Inverse;
    let inv_transform = transform.inv().map_err(|e| {
        VisionError::OperationError(format!("Failed to invert transformation: {}", e))
    })?;

    // For each pixel in output image
    for y in 0..out_height {
        for x in 0..out_width {
            // Map output coordinates to input coordinates
            let out_point = Point2D::new(x as f64, y as f64);
            let in_point = transform_point(out_point, &inv_transform);

            // Sample input image at mapped coordinates
            let intensity = sample_image(
                image,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );

            output.put_pixel(x, y, Luma([intensity as u8]));
        }
    }

    Ok(output)
}

/// Warp an RGB image using a transformation matrix
pub fn warp_rgb_image(
    image: &RgbImage,
    transform: &TransformMatrix,
    output_size: (u32, u32),
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
) -> Result<RgbImage> {
    let (out_width, out_height) = output_size;
    let (in_width, in_height) = image.dimensions();

    // Create output image
    let mut output = RgbImage::new(out_width, out_height);

    // Invert transformation for backwards mapping
    use ndarray_linalg::Inverse;
    let inv_transform = transform.inv().map_err(|e| {
        VisionError::OperationError(format!("Failed to invert transformation: {}", e))
    })?;

    // For each pixel in output image
    for y in 0..out_height {
        for x in 0..out_width {
            // Map output coordinates to input coordinates
            let out_point = Point2D::new(x as f64, y as f64);
            let in_point = transform_point(out_point, &inv_transform);

            // Sample each color channel
            let r = sample_rgb_image(
                image,
                0,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );
            let g = sample_rgb_image(
                image,
                1,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );
            let b = sample_rgb_image(
                image,
                2,
                in_point.x as f32,
                in_point.y as f32,
                interpolation,
                boundary,
                in_width,
                in_height,
            );

            output.put_pixel(x, y, Rgb([r as u8, g as u8, b as u8]));
        }
    }

    Ok(output)
}

/// Sample a grayscale image at fractional coordinates
fn sample_image(
    image: &GrayImage,
    x: f32,
    y: f32,
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    match interpolation {
        InterpolationMethod::NearestNeighbor => {
            let ix = x.round() as i32;
            let iy = y.round() as i32;
            get_pixel_value(image, ix, iy, boundary, width, height)
        }
        InterpolationMethod::Bilinear => {
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let v00 = get_pixel_value(image, x0, y0, boundary, width, height);
            let v01 = get_pixel_value(image, x0, y1, boundary, width, height);
            let v10 = get_pixel_value(image, x1, y0, boundary, width, height);
            let v11 = get_pixel_value(image, x1, y1, boundary, width, height);

            let v0 = v00 * (1.0 - fx) + v10 * fx;
            let v1 = v01 * (1.0 - fx) + v11 * fx;

            v0 * (1.0 - fy) + v1 * fy
        }
        InterpolationMethod::Bicubic => {
            // Simplified bicubic interpolation
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let mut sum = 0.0;
            for j in -1..3 {
                for i in -1..3 {
                    let weight = cubic_kernel(fx - i as f32) * cubic_kernel(fy - j as f32);
                    let value = get_pixel_value(image, x0 + i, y0 + j, boundary, width, height);
                    sum += weight * value;
                }
            }

            sum.clamp(0.0, 255.0)
        }
    }
}

/// Sample an RGB image at fractional coordinates for a specific channel
fn sample_rgb_image(
    image: &RgbImage,
    channel: usize,
    x: f32,
    y: f32,
    interpolation: InterpolationMethod,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    match interpolation {
        InterpolationMethod::NearestNeighbor => {
            let ix = x.round() as i32;
            let iy = y.round() as i32;
            get_rgb_pixel_value(image, channel, ix, iy, boundary, width, height)
        }
        InterpolationMethod::Bilinear => {
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;
            let x1 = x0 + 1;
            let y1 = y0 + 1;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let v00 = get_rgb_pixel_value(image, channel, x0, y0, boundary, width, height);
            let v01 = get_rgb_pixel_value(image, channel, x0, y1, boundary, width, height);
            let v10 = get_rgb_pixel_value(image, channel, x1, y0, boundary, width, height);
            let v11 = get_rgb_pixel_value(image, channel, x1, y1, boundary, width, height);

            let v0 = v00 * (1.0 - fx) + v10 * fx;
            let v1 = v01 * (1.0 - fx) + v11 * fx;

            v0 * (1.0 - fy) + v1 * fy
        }
        InterpolationMethod::Bicubic => {
            let x0 = x.floor() as i32;
            let y0 = y.floor() as i32;

            let fx = x - x0 as f32;
            let fy = y - y0 as f32;

            let mut sum = 0.0;
            for j in -1..3 {
                for i in -1..3 {
                    let weight = cubic_kernel(fx - i as f32) * cubic_kernel(fy - j as f32);
                    let value = get_rgb_pixel_value(
                        image,
                        channel,
                        x0 + i,
                        y0 + j,
                        boundary,
                        width,
                        height,
                    );
                    sum += weight * value;
                }
            }

            sum.clamp(0.0, 255.0)
        }
    }
}

/// Get pixel value with boundary handling
fn get_pixel_value(
    image: &GrayImage,
    x: i32,
    y: i32,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    let (nx, ny) = handle_boundary(x, y, boundary, width, height);

    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
        image.get_pixel(nx as u32, ny as u32)[0] as f32
    } else {
        match boundary {
            BoundaryMethod::Zero => 0.0,
            BoundaryMethod::Constant(value) => value,
            _ => 0.0, // Fallback
        }
    }
}

/// Get RGB pixel value with boundary handling
fn get_rgb_pixel_value(
    image: &RgbImage,
    channel: usize,
    x: i32,
    y: i32,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> f32 {
    let (nx, ny) = handle_boundary(x, y, boundary, width, height);

    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
        image.get_pixel(nx as u32, ny as u32)[channel] as f32
    } else {
        match boundary {
            BoundaryMethod::Zero => 0.0,
            BoundaryMethod::Constant(value) => value,
            _ => 0.0, // Fallback
        }
    }
}

/// Handle boundary conditions
fn handle_boundary(
    x: i32,
    y: i32,
    boundary: BoundaryMethod,
    width: u32,
    height: u32,
) -> (i32, i32) {
    let w = width as i32;
    let h = height as i32;

    match boundary {
        BoundaryMethod::Zero | BoundaryMethod::Constant(_) => (x, y),
        BoundaryMethod::Reflect => {
            let nx = if x < 0 {
                -x - 1
            } else if x >= w {
                2 * w - x - 1
            } else {
                x
            };

            let ny = if y < 0 {
                -y - 1
            } else if y >= h {
                2 * h - y - 1
            } else {
                y
            };

            (nx.clamp(0, w - 1), ny.clamp(0, h - 1))
        }
        BoundaryMethod::Wrap => {
            let nx = ((x % w) + w) % w;
            let ny = ((y % h) + h) % h;
            (nx, ny)
        }
        BoundaryMethod::Clamp => (x.clamp(0, w - 1), y.clamp(0, h - 1)),
    }
}

/// Cubic interpolation kernel
fn cubic_kernel(t: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        1.5 * t * t * t - 2.5 * t * t + 1.0
    } else if t <= 2.0 {
        -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}

/// Create a mesh grid for transformation mapping
pub fn create_mesh_grid(width: u32, height: u32) -> (Array2<f64>, Array2<f64>) {
    let mut x_grid = Array2::zeros((height as usize, width as usize));
    let mut y_grid = Array2::zeros((height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            x_grid[[y as usize, x as usize]] = x as f64;
            y_grid[[y as usize, x as usize]] = y as f64;
        }
    }

    (x_grid, y_grid)
}

/// Apply perspective correction to an image
pub fn perspective_correct(
    image: &DynamicImage,
    corners: &[Point2D; 4],
    output_size: (u32, u32),
) -> Result<DynamicImage> {
    // Define target rectangle corners
    let (width, height) = output_size;
    let target_corners = [
        Point2D::new(0.0, 0.0),
        Point2D::new(width as f64 - 1.0, 0.0),
        Point2D::new(width as f64 - 1.0, height as f64 - 1.0),
        Point2D::new(0.0, height as f64 - 1.0),
    ];

    // Create matches for homography estimation
    let matches: Vec<_> = corners
        .iter()
        .zip(target_corners.iter())
        .map(|(&src, &tgt)| crate::registration::PointMatch {
            source: src,
            target: tgt,
            confidence: 1.0,
        })
        .collect();

    // Estimate homography
    use crate::registration::estimate_homography_transform;
    let transform = estimate_homography_transform(&matches)?;

    // Warp image
    match image {
        DynamicImage::ImageLuma8(gray) => {
            let warped = warp_image(
                gray,
                &transform,
                output_size,
                InterpolationMethod::Bilinear,
                BoundaryMethod::Zero,
            )?;
            Ok(DynamicImage::ImageLuma8(warped))
        }
        DynamicImage::ImageRgb8(rgb) => {
            let warped = warp_rgb_image(
                rgb,
                &transform,
                output_size,
                InterpolationMethod::Bilinear,
                BoundaryMethod::Zero,
            )?;
            Ok(DynamicImage::ImageRgb8(warped))
        }
        _ => {
            // Convert to RGB and process
            let rgb = image.to_rgb8();
            let warped = warp_rgb_image(
                &rgb,
                &transform,
                output_size,
                InterpolationMethod::Bilinear,
                BoundaryMethod::Zero,
            )?;
            Ok(DynamicImage::ImageRgb8(warped))
        }
    }
}

/// Rectify stereo image pair using fundamental matrix
pub fn rectify_stereo_pair(
    left_image: &DynamicImage,
    right_image: &DynamicImage,
    _fundamental_matrix: &TransformMatrix,
) -> Result<(DynamicImage, DynamicImage)> {
    // This is a simplified rectification
    // In practice, this would involve computing rectification transforms
    // from the fundamental matrix and camera parameters

    // For now, return the original images
    // TODO: Implement proper stereo rectification
    Ok((left_image.clone(), right_image.clone()))
}

/// Create a panorama by stitching multiple images
pub fn stitch_images(
    images: &[DynamicImage],
    transforms: &[TransformMatrix],
    output_size: (u32, u32),
) -> Result<DynamicImage> {
    if images.len() != transforms.len() {
        return Err(VisionError::InvalidParameter(
            "Number of images must match number of transforms".to_string(),
        ));
    }

    let (width, height) = output_size;
    let mut output = RgbImage::new(width, height);
    let mut weight_map = Array2::<f32>::zeros((height as usize, width as usize));

    // Initialize output with zeros
    for y in 0..height {
        for x in 0..width {
            output.put_pixel(x, y, Rgb([0, 0, 0]));
        }
    }

    // Blend each image
    for (image, transform) in images.iter().zip(transforms.iter()) {
        let rgb_image = image.to_rgb8();
        let warped = warp_rgb_image(
            &rgb_image,
            transform,
            output_size,
            InterpolationMethod::Bilinear,
            BoundaryMethod::Zero,
        )?;

        // Simple averaging blend
        for y in 0..height {
            for x in 0..width {
                let warped_pixel = warped.get_pixel(x, y);
                let output_pixel = output.get_pixel_mut(x, y);

                // Check if warped pixel is not black (indicating valid data)
                if warped_pixel[0] > 0 || warped_pixel[1] > 0 || warped_pixel[2] > 0 {
                    let weight = weight_map[[y as usize, x as usize]];
                    let new_weight = weight + 1.0;

                    for c in 0..3 {
                        let old_value = output_pixel[c] as f32;
                        let new_value = warped_pixel[c] as f32;
                        let blended: f32 = (old_value * weight + new_value) / new_weight;
                        output_pixel[c] = blended as u8;
                    }

                    weight_map[[y as usize, x as usize]] = new_weight;
                }
            }
        }
    }

    Ok(DynamicImage::ImageRgb8(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registration::identity_transform;

    fn create_test_image() -> GrayImage {
        let mut image = GrayImage::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                image.put_pixel(x, y, Luma([((x + y) * 25) as u8]));
            }
        }
        image
    }

    #[test]
    fn test_identity_warp() {
        let image = create_test_image();
        let transform = identity_transform();

        let warped = warp_image(
            &image,
            &transform,
            (10, 10),
            InterpolationMethod::NearestNeighbor,
            BoundaryMethod::Zero,
        )
        .unwrap();

        // Should be identical to original
        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(image.get_pixel(x, y)[0], warped.get_pixel(x, y)[0]);
            }
        }
    }

    #[test]
    fn test_translation_warp() {
        let image = create_test_image();
        let mut transform = identity_transform();
        transform[[0, 2]] = 1.0; // Translate by 1 pixel in x

        let warped = warp_image(
            &image,
            &transform,
            (10, 10),
            InterpolationMethod::NearestNeighbor,
            BoundaryMethod::Zero,
        )
        .unwrap();

        // Check that translation occurred
        assert_eq!(warped.get_pixel(0, 0)[0], 0); // Should be zero (background)
        assert_eq!(warped.get_pixel(1, 0)[0], image.get_pixel(0, 0)[0]);
    }

    #[test]
    fn test_interpolation_methods() {
        let image = create_test_image();
        let transform = identity_transform();

        // Test all interpolation methods
        for &method in &[
            InterpolationMethod::NearestNeighbor,
            InterpolationMethod::Bilinear,
            InterpolationMethod::Bicubic,
        ] {
            let result = warp_image(&image, &transform, (10, 10), method, BoundaryMethod::Zero);
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_boundary_methods() {
        let image = create_test_image();
        let mut transform = identity_transform();
        transform[[0, 2]] = -5.0; // Translate outside bounds

        // Test all boundary methods
        for &method in &[
            BoundaryMethod::Zero,
            BoundaryMethod::Constant(128.0),
            BoundaryMethod::Reflect,
            BoundaryMethod::Wrap,
            BoundaryMethod::Clamp,
        ] {
            let result = warp_image(
                &image,
                &transform,
                (10, 10),
                InterpolationMethod::NearestNeighbor,
                method,
            );
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_perspective_correction() {
        let image = DynamicImage::ImageLuma8(create_test_image());

        // Define a simple quadrilateral
        let corners = [
            Point2D::new(1.0, 1.0),
            Point2D::new(8.0, 1.0),
            Point2D::new(8.0, 8.0),
            Point2D::new(1.0, 8.0),
        ];

        let result = perspective_correct(&image, &corners, (100, 100));
        assert!(result.is_ok());
    }
}
