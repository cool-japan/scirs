//! Local Binary Patterns (LBP) for texture analysis
//!
//! LBP is a powerful feature for texture classification that labels the pixels
//! of an image by thresholding the neighborhood of each pixel and considers
//! the result as a binary number.

use crate::error::Result;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use ndarray::Array1;
use std::collections::HashMap;

/// LBP variant types
#[derive(Debug, Clone, Copy)]
pub enum LBPType {
    /// Original LBP with fixed 3x3 neighborhood
    Original,
    /// Extended LBP with configurable radius and points
    Extended {
        /// Radius of the circular pattern
        radius: f32,
        /// Number of sampling points
        points: usize,
    },
    /// Uniform LBP (patterns with at most 2 transitions)
    Uniform {
        /// Radius of the circular pattern
        radius: f32,
        /// Number of sampling points
        points: usize,
    },
    /// Rotation invariant LBP
    RotationInvariant {
        /// Radius of the circular pattern
        radius: f32,
        /// Number of sampling points
        points: usize,
    },
}

/// Compute Local Binary Pattern for an image
///
/// # Arguments
///
/// * `img` - Input grayscale image
/// * `lbp_type` - Type of LBP to compute
///
/// # Returns
///
/// * Result containing LBP image
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::{lbp, LBPType};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let lbp_img = lbp(&img, LBPType::Original)?;
/// # Ok(())
/// # }
/// ```
pub fn lbp(img: &DynamicImage, lbp_type: LBPType) -> Result<GrayImage> {
    let gray = img.to_luma8();
    let (_width, _height) = gray.dimensions();

    match lbp_type {
        LBPType::Original => compute_lbp_original(&gray),
        LBPType::Extended { radius, points } => compute_lbp_extended(&gray, radius, points),
        LBPType::Uniform { radius, points } => compute_lbp_uniform(&gray, radius, points),
        LBPType::RotationInvariant { radius, points } => {
            compute_lbp_rotation_invariant(&gray, radius, points)
        }
    }
}

/// Compute original 3x3 LBP
fn compute_lbp_original(gray: &GrayImage) -> Result<GrayImage> {
    let (width, height) = gray.dimensions();
    let mut lbp_img = ImageBuffer::new(width, height);

    // Define 3x3 neighborhood offsets
    let offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let center = gray.get_pixel(x, y)[0];
            let mut pattern = 0u8;

            // Compare with neighbors
            for (i, &(dy, dx)) in offsets.iter().enumerate() {
                let ny = (y as i32 + dy) as u32;
                let nx = (x as i32 + dx) as u32;
                let neighbor = gray.get_pixel(nx, ny)[0];

                if neighbor >= center {
                    pattern |= 1 << i;
                }
            }

            lbp_img.put_pixel(x, y, Luma([pattern]));
        }
    }

    // Handle borders by copying original values
    for x in 0..width {
        lbp_img.put_pixel(x, 0, Luma([0]));
        lbp_img.put_pixel(x, height - 1, Luma([0]));
    }
    for y in 0..height {
        lbp_img.put_pixel(0, y, Luma([0]));
        lbp_img.put_pixel(width - 1, y, Luma([0]));
    }

    Ok(lbp_img)
}

/// Compute extended LBP with circular neighborhood
fn compute_lbp_extended(gray: &GrayImage, radius: f32, points: usize) -> Result<GrayImage> {
    let (width, height) = gray.dimensions();
    let mut lbp_img = ImageBuffer::new(width, height);

    // Compute neighbor positions
    let neighbors = compute_circular_neighbors(radius, points);

    let border = radius.ceil() as u32;

    for y in border..height - border {
        for x in border..width - border {
            let center = gray.get_pixel(x, y)[0];
            let mut pattern = 0u32;

            // Sample neighbors using bilinear interpolation
            for (i, &(dy, dx)) in neighbors.iter().enumerate() {
                let ny = y as f32 + dy;
                let nx = x as f32 + dx;
                let neighbor = bilinear_interpolate(gray, nx, ny);

                if neighbor >= center as f32 {
                    pattern |= 1 << i;
                }
            }

            // Map to 8-bit range
            let value = if points <= 8 {
                pattern as u8
            } else {
                // For more than 8 points, use modulo to fit in u8
                (pattern % 256) as u8
            };

            lbp_img.put_pixel(x, y, Luma([value]));
        }
    }

    Ok(lbp_img)
}

/// Compute uniform LBP (patterns with at most 2 transitions)
fn compute_lbp_uniform(gray: &GrayImage, radius: f32, points: usize) -> Result<GrayImage> {
    let (width, height) = gray.dimensions();
    let mut lbp_img = ImageBuffer::new(width, height);

    // Compute neighbor positions
    let neighbors = compute_circular_neighbors(radius, points);

    // Build uniform pattern lookup table
    let uniform_map = build_uniform_pattern_map(points);

    let border = radius.ceil() as u32;

    for y in border..height - border {
        for x in border..width - border {
            let center = gray.get_pixel(x, y)[0];
            let mut pattern = 0u32;

            // Sample neighbors
            for (i, &(dy, dx)) in neighbors.iter().enumerate() {
                let ny = y as f32 + dy;
                let nx = x as f32 + dx;
                let neighbor = bilinear_interpolate(gray, nx, ny);

                if neighbor >= center as f32 {
                    pattern |= 1 << i;
                }
            }

            // Map to uniform pattern
            let value = uniform_map
                .get(&pattern)
                .copied()
                .unwrap_or(points as u8 + 1);

            lbp_img.put_pixel(x, y, Luma([value]));
        }
    }

    Ok(lbp_img)
}

/// Compute rotation invariant LBP
fn compute_lbp_rotation_invariant(
    gray: &GrayImage,
    radius: f32,
    points: usize,
) -> Result<GrayImage> {
    let (width, height) = gray.dimensions();
    let mut lbp_img = ImageBuffer::new(width, height);

    // Compute neighbor positions
    let neighbors = compute_circular_neighbors(radius, points);

    let border = radius.ceil() as u32;

    for y in border..height - border {
        for x in border..width - border {
            let center = gray.get_pixel(x, y)[0];
            let mut pattern = 0u32;

            // Sample neighbors
            for (i, &(dy, dx)) in neighbors.iter().enumerate() {
                let ny = y as f32 + dy;
                let nx = x as f32 + dx;
                let neighbor = bilinear_interpolate(gray, nx, ny);

                if neighbor >= center as f32 {
                    pattern |= 1 << i;
                }
            }

            // Find minimum rotation
            let min_pattern = find_min_rotation(pattern, points);

            // Map to 8-bit range
            let value = if points <= 8 {
                min_pattern as u8
            } else {
                (min_pattern % 256) as u8
            };

            lbp_img.put_pixel(x, y, Luma([value]));
        }
    }

    Ok(lbp_img)
}

/// Compute circular neighbor positions
fn compute_circular_neighbors(radius: f32, points: usize) -> Vec<(f32, f32)> {
    let mut neighbors = Vec::with_capacity(points);

    for i in 0..points {
        let angle = 2.0 * std::f32::consts::PI * i as f32 / points as f32;
        let dy = -radius * angle.cos();
        let dx = radius * angle.sin();
        neighbors.push((dy, dx));
    }

    neighbors
}

/// Bilinear interpolation for sub-pixel sampling
fn bilinear_interpolate(img: &GrayImage, x: f32, y: f32) -> f32 {
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x0 as f32;
    let fy = y - y0 as f32;

    let (width, height) = img.dimensions();
    let x1 = x1.min(width - 1);
    let y1 = y1.min(height - 1);

    let v00 = img.get_pixel(x0, y0)[0] as f32;
    let v10 = img.get_pixel(x1, y0)[0] as f32;
    let v01 = img.get_pixel(x0, y1)[0] as f32;
    let v11 = img.get_pixel(x1, y1)[0] as f32;

    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;

    v0 * (1.0 - fy) + v1 * fy
}

/// Build uniform pattern lookup table
fn build_uniform_pattern_map(points: usize) -> HashMap<u32, u8> {
    let mut map = HashMap::new();
    let mut label = 0u8;

    for pattern in 0..(1 << points) {
        if is_uniform_pattern(pattern, points) {
            map.insert(pattern, label);
            label += 1;
        }
    }

    map
}

/// Check if a pattern is uniform (at most 2 transitions)
fn is_uniform_pattern(pattern: u32, points: usize) -> bool {
    let mut transitions = 0;

    for i in 0..points {
        let bit1 = (pattern >> i) & 1;
        let bit2 = (pattern >> ((i + 1) % points)) & 1;

        if bit1 != bit2 {
            transitions += 1;
        }
    }

    transitions <= 2
}

/// Find minimum rotation of a pattern
fn find_min_rotation(pattern: u32, points: usize) -> u32 {
    let mut min_pattern = pattern;

    for i in 1..points {
        let rotated = rotate_pattern(pattern, i, points);
        if rotated < min_pattern {
            min_pattern = rotated;
        }
    }

    min_pattern
}

/// Rotate a pattern by n positions
fn rotate_pattern(pattern: u32, n: usize, points: usize) -> u32 {
    let mask = (1 << points) - 1;
    ((pattern >> n) | (pattern << (points - n))) & mask
}

/// Compute LBP histogram for texture analysis
///
/// # Arguments
///
/// * `lbp_img` - LBP image
/// * `n_bins` - Number of histogram bins
/// * `normalize` - Whether to normalize the histogram
///
/// # Returns
///
/// * Result containing histogram
pub fn lbp_histogram(lbp_img: &GrayImage, n_bins: usize, normalize: bool) -> Result<Array1<f32>> {
    let mut histogram = Array1::zeros(n_bins);
    let scale = 256.0 / n_bins as f32;

    // Count occurrences
    for pixel in lbp_img.pixels() {
        let bin = (pixel[0] as f32 / scale).floor() as usize;
        let bin = bin.min(n_bins - 1);
        histogram[bin] += 1.0;
    }

    // Normalize if requested
    if normalize {
        let sum = histogram.sum();
        if sum > 0.0 {
            histogram /= sum;
        }
    }

    Ok(histogram)
}

/// Compute multi-scale LBP features
///
/// # Arguments
///
/// * `img` - Input image
/// * `scales` - List of (radius, points) pairs
///
/// # Returns
///
/// * Result containing concatenated histogram features
pub fn multi_scale_lbp(img: &DynamicImage, scales: &[(f32, usize)]) -> Result<Array1<f32>> {
    let mut features = Vec::new();

    for &(radius, points) in scales {
        let lbp_img = lbp(img, LBPType::Uniform { radius, points })?;

        // Uniform patterns + 1 non-uniform bin
        let n_bins = points * (points - 1) + 3;
        let hist = lbp_histogram(&lbp_img, n_bins, true)?;

        features.extend(hist.iter());
    }

    Ok(Array1::from_vec(features))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbp_original() {
        let img = DynamicImage::new_luma8(10, 10);
        let result = lbp(&img, LBPType::Original);
        assert!(result.is_ok());

        let lbp_img = result.unwrap();
        assert_eq!(lbp_img.dimensions(), (10, 10));
    }

    #[test]
    fn test_lbp_extended() {
        let img = DynamicImage::new_luma8(20, 20);
        let result = lbp(
            &img,
            LBPType::Extended {
                radius: 1.5,
                points: 8,
            },
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_uniform_pattern() {
        assert!(is_uniform_pattern(0b00000000, 8)); // All zeros
        assert!(is_uniform_pattern(0b11111111, 8)); // All ones
        assert!(is_uniform_pattern(0b00000111, 8)); // One transition region
        assert!(is_uniform_pattern(0b00111000, 8)); // Two transitions
        assert!(!is_uniform_pattern(0b01010101, 8)); // Many transitions
    }

    #[test]
    fn test_rotation_invariant() {
        assert_eq!(find_min_rotation(0b00001111, 8), 0b00001111);
        assert_eq!(find_min_rotation(0b11110000, 8), 0b00001111);
        assert_eq!(find_min_rotation(0b00111100, 8), 0b00001111);
    }

    #[test]
    fn test_lbp_histogram() {
        let mut img = GrayImage::new(10, 10);
        // Fill with some pattern values
        for (i, pixel) in img.pixels_mut().enumerate() {
            *pixel = Luma([(i % 256) as u8]);
        }

        let hist = lbp_histogram(&img, 16, true).unwrap();
        assert_eq!(hist.len(), 16);
        assert!((hist.sum() - 1.0).abs() < 1e-6); // Should sum to 1 when normalized
    }
}
