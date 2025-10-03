//! SLIC (Simple Linear Iterative Clustering) superpixel segmentation
//!
//! SLIC is a superpixel algorithm that clusters pixels in the combined
//! five-dimensional color and image plane space to efficiently generate
//! compact, nearly uniform superpixels.
//!
//! # Performance Characteristics
//!
//! - Time complexity: O(N) where N is the number of pixels (linear in image size)
//! - Space complexity: O(N + K) where K is the number of superpixels
//! - The algorithm typically converges in 10 iterations
//! - Search region is limited to 2S × 2S area around each cluster center (where S = sqrt(N/K))
//!
//! # References
//!
//! - Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P. and Süsstrunk, S., 2012. SLIC superpixels compared to state-of-the-art superpixel methods. IEEE transactions on pattern analysis and machine intelligence, 34(11), pp.2274-2282.

use crate::error::{Result, VisionError};
use image::{DynamicImage, RgbImage};
use scirs2_core::ndarray::{Array2, Array3};

/// SLIC superpixel center
#[derive(Debug, Clone)]
struct SuperpixelCenter {
    /// Position in image coordinates
    y: f32,
    x: f32,
    /// Lab color values
    l: f32,
    a: f32,
    b: f32,
    /// Number of pixels in this superpixel
    pixel_count: usize,
}

/// SLIC superpixel segmentation
///
/// Segments an image into superpixels using the SLIC algorithm.
///
/// # Arguments
///
/// * `img` - Input RGB image
/// * `nsegments` - Approximate number of superpixels
/// * `compactness` - Balance between color and spatial distance (default: 10.0)
/// * `max_iterations` - Maximum number of iterations (default: 10)
/// * `sigma` - Width of Gaussian smoothing kernel (0 to disable)
///
/// # Returns
///
/// * Result containing labeled superpixel map
///
/// # Example
///
/// ```rust
/// use scirs2_vision::segmentation::slic;
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let superpixels = slic(&img, 100, 10.0, 10, 0.0)?;
/// # Ok(())
/// # }
/// ```
#[allow(dead_code)]
pub fn slic(
    img: &DynamicImage,
    nsegments: usize,
    compactness: f32,
    max_iterations: usize,
    sigma: f32,
) -> Result<Array2<u32>> {
    if nsegments == 0 {
        return Err(VisionError::InvalidParameter(
            "Number of _segments must be positive".to_string(),
        ));
    }

    let rgb = img.to_rgb8();
    let (width_, height) = rgb.dimensions();

    // Apply Gaussian smoothing if requested
    let smoothed = if sigma > 0.0 {
        gaussian_smooth_rgb(&rgb, sigma)?
    } else {
        rgb.clone()
    };

    // Convert to Lab color space
    let lab = rgb_to_lab_array(&smoothed);

    // Initialize cluster centers
    let gridstep = ((width_ * height) as f32 / nsegments as f32).sqrt();
    let mut centers = initialize_centers(&lab, gridstep as usize);

    // Move centers to lowest gradient position
    perturb_centers(&mut centers, &lab);

    // Initialize labels and distances
    let mut labels = Array2::from_elem((height as usize, width_ as usize), u32::MAX);
    let mut distances = Array2::from_elem((height as usize, width_ as usize), f32::INFINITY);

    // Main SLIC iteration
    for _iter in 0..max_iterations {
        // Reset distances
        distances.fill(f32::INFINITY);

        // Assign pixels to nearest center
        for (k, center) in centers.iter().enumerate() {
            let y_min = (center.y - 2.0 * gridstep).max(0.0) as usize;
            let y_max = (center.y + 2.0 * gridstep).min(height as f32) as usize;
            let x_min = (center.x - 2.0 * gridstep).max(0.0) as usize;
            let x_max = (center.x + 2.0 * gridstep).min(width_ as f32) as usize;

            for y in y_min..y_max {
                for x in x_min..x_max {
                    let dist = compute_distance(&lab, y, x, center, gridstep, compactness);

                    if dist < distances[[y, x]] {
                        distances[[y, x]] = dist;
                        labels[[y, x]] = k as u32;
                    }
                }
            }
        }

        // Update centers
        update_centers(&mut centers, &lab, &labels);
    }

    // Post-processing: enforce connectivity
    enforce_connectivity(&mut labels, nsegments);

    Ok(labels)
}

/// Initialize superpixel centers on a regular grid
#[allow(dead_code)]
fn initialize_centers(lab: &Array3<f32>, gridstep: usize) -> Vec<SuperpixelCenter> {
    let (height, width_, _) = lab.dim();
    let mut centers = Vec::new();

    let half_step = gridstep / 2;

    for y in (half_step..height).step_by(gridstep) {
        for x in (half_step..width_).step_by(gridstep) {
            centers.push(SuperpixelCenter {
                y: y as f32,
                x: x as f32,
                l: lab[[y, x, 0]],
                a: lab[[y, x, 1]],
                b: lab[[y, x, 2]],
                pixel_count: 0,
            });
        }
    }

    centers
}

/// Move centers to positions with lowest gradient
#[allow(dead_code)]
fn perturb_centers(centers: &mut [SuperpixelCenter], lab: &Array3<f32>) {
    let (height, width_, _) = lab.dim();

    for center in centers.iter_mut() {
        let y = center.y as usize;
        let x = center.x as usize;

        // Search in 3x3 neighborhood
        let mut min_gradient = f32::INFINITY;
        let mut best_y = y;
        let mut best_x = x;

        for dy in -1..=1 {
            for dx in -1..=1 {
                let ny = (y as i32 + dy).max(1).min(height as i32 - 2) as usize;
                let nx = (x as i32 + dx).max(1).min(width_ as i32 - 2) as usize;

                let gradient = compute_gradient(lab, ny, nx);

                if gradient < min_gradient {
                    min_gradient = gradient;
                    best_y = ny;
                    best_x = nx;
                }
            }
        }

        // Update center position
        center.y = best_y as f32;
        center.x = best_x as f32;
        center.l = lab[[best_y, best_x, 0]];
        center.a = lab[[best_y, best_x, 1]];
        center.b = lab[[best_y, best_x, 2]];
    }
}

/// Compute gradient magnitude at a pixel
#[allow(dead_code)]
fn compute_gradient(lab: &Array3<f32>, y: usize, x: usize) -> f32 {
    let mut gradient = 0.0;

    for c in 0..3 {
        let dx = lab[[y, x + 1, c]] - lab[[y, x - 1, c]];
        let dy = lab[[y + 1, x, c]] - lab[[y - 1, x, c]];
        gradient += dx * dx + dy * dy;
    }

    gradient.sqrt()
}

/// Compute distance between pixel and superpixel center
#[allow(dead_code)]
fn compute_distance(
    lab: &Array3<f32>,
    y: usize,
    x: usize,
    center: &SuperpixelCenter,
    gridstep: f32,
    compactness: f32,
) -> f32 {
    // Color distance
    let dl = lab[[y, x, 0]] - center.l;
    let da = lab[[y, x, 1]] - center.a;
    let db = lab[[y, x, 2]] - center.b;
    let color_dist = (dl * dl + da * da + db * db).sqrt();

    // Spatial distance
    let dy = y as f32 - center.y;
    let dx = x as f32 - center.x;
    let spatial_dist = (dy * dy + dx * dx).sqrt();

    // Combined distance
    color_dist + (compactness / gridstep) * spatial_dist
}

/// Update superpixel centers based on assigned pixels
#[allow(dead_code)]
fn update_centers(centers: &mut [SuperpixelCenter], lab: &Array3<f32>, labels: &Array2<u32>) {
    let (height, width_, _) = lab.dim();

    // Reset _centers
    for center in centers.iter_mut() {
        center.y = 0.0;
        center.x = 0.0;
        center.l = 0.0;
        center.a = 0.0;
        center.b = 0.0;
        center.pixel_count = 0;
    }

    // Accumulate values
    for y in 0..height {
        for x in 0..width_ {
            let label = labels[[y, x]] as usize;
            if label < centers.len() {
                centers[label].y += y as f32;
                centers[label].x += x as f32;
                centers[label].l += lab[[y, x, 0]];
                centers[label].a += lab[[y, x, 1]];
                centers[label].b += lab[[y, x, 2]];
                centers[label].pixel_count += 1;
            }
        }
    }

    // Compute means
    for center in centers.iter_mut() {
        if center.pixel_count > 0 {
            let count = center.pixel_count as f32;
            center.y /= count;
            center.x /= count;
            center.l /= count;
            center.a /= count;
            center.b /= count;
        }
    }
}

/// Enforce connectivity of superpixels
#[allow(dead_code)]
fn enforce_connectivity(labels: &mut Array2<u32>, nsegments: usize) {
    let (height, width_) = labels.dim();
    let min_size = (height * width_) / (nsegments * 4);

    // Find and merge small _segments
    let mut new_label = 0;
    let mut visited = Array2::from_elem((height, width_), false);

    for y in 0..height {
        for x in 0..width_ {
            if !visited[[y, x]] {
                let old_label = labels[[y, x]];
                let size = flood_fill(labels, &mut visited, y, x, old_label, new_label);

                if size >= min_size {
                    new_label += 1;
                } else {
                    // Merge with nearest neighbor
                    merge_small_segment(labels, y, x, new_label);
                }
            }
        }
    }
}

/// Flood fill to count and relabel connected components
#[allow(dead_code)]
fn flood_fill(
    labels: &mut Array2<u32>,
    visited: &mut Array2<bool>,
    start_y: usize,
    start_x: usize,
    old_label: u32,
    new_label: u32,
) -> usize {
    let (height, width_) = labels.dim();
    let mut stack = vec![(start_y, start_x)];
    let mut size = 0;

    while let Some((y, x)) = stack.pop() {
        if visited[[y, x]] || labels[[y, x]] != old_label {
            continue;
        }

        visited[[y, x]] = true;
        labels[[y, x]] = new_label;
        size += 1;

        // Check 4-neighbors
        if y > 0 {
            stack.push((y - 1, x));
        }
        if y < height - 1 {
            stack.push((y + 1, x));
        }
        if x > 0 {
            stack.push((y, x - 1));
        }
        if x < width_ - 1 {
            stack.push((y, x + 1));
        }
    }

    size
}

/// Merge small segment with neighbor
#[allow(dead_code)]
fn merge_small_segment(labels: &mut Array2<u32>, y: usize, x: usize, currentlabel: u32) {
    let (height, width_) = labels.dim();
    let neighbors = [
        (y.wrapping_sub(1), x),
        (y + 1, x),
        (y, x.wrapping_sub(1)),
        (y, x + 1),
    ];

    for &(ny, nx) in neighbors.iter() {
        if ny < height && nx < width_ && labels[[ny, nx]] != currentlabel {
            // Replace current segment with neighbor's _label
            let neighbor_label = labels[[ny, nx]];
            flood_fill_replace(labels, y, x, currentlabel, neighbor_label);
            break;
        }
    }
}

/// Replace all pixels with old_label with new_label using flood fill
#[allow(dead_code)]
fn flood_fill_replace(
    labels: &mut Array2<u32>,
    start_y: usize,
    start_x: usize,
    old_label: u32,
    new_label: u32,
) {
    let (height, width_) = labels.dim();
    let mut stack = vec![(start_y, start_x)];

    while let Some((y, x)) = stack.pop() {
        if labels[[y, x]] != old_label {
            continue;
        }

        labels[[y, x]] = new_label;

        if y > 0 {
            stack.push((y - 1, x));
        }
        if y < height - 1 {
            stack.push((y + 1, x));
        }
        if x > 0 {
            stack.push((y, x - 1));
        }
        if x < width_ - 1 {
            stack.push((y, x + 1));
        }
    }
}

/// Convert RGB image to Lab color space array
#[allow(dead_code)]
fn rgb_to_lab_array(img: &RgbImage) -> Array3<f32> {
    let (width_, height) = img.dimensions();
    let mut lab = Array3::zeros((height as usize, width_ as usize, 3));

    for y in 0..height {
        for x in 0..width_ {
            let rgb = img.get_pixel(x, y);
            let (l, a, b) = rgb_to_lab(rgb[0], rgb[1], rgb[2]);
            lab[[y as usize, x as usize, 0]] = l;
            lab[[y as usize, x as usize, 1]] = a;
            lab[[y as usize, x as usize, 2]] = b;
        }
    }

    lab
}

/// Convert RGB to Lab color space
#[allow(dead_code)]
fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    // Convert to linear RGB
    let r = srgb_to_linear(r as f32 / 255.0);
    let g = srgb_to_linear(g as f32 / 255.0);
    let b = srgb_to_linear(b as f32 / 255.0);

    // Convert to XYZ
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.119_192 * g + 0.9503041 * b;

    // Normalize by D65 illuminant
    let x = x / 0.95047;
    let y = y / 1.00000;
    let z = z / 1.08883;

    // Convert to Lab
    let fx = if x > 0.008856 {
        x.powf(1.0 / 3.0)
    } else {
        7.787 * x + 16.0 / 116.0
    };
    let fy = if y > 0.008856 {
        y.powf(1.0 / 3.0)
    } else {
        7.787 * y + 16.0 / 116.0
    };
    let fz = if z > 0.008856 {
        z.powf(1.0 / 3.0)
    } else {
        7.787 * z + 16.0 / 116.0
    };

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    (l, a, b)
}

/// Convert sRGB to linear RGB
#[allow(dead_code)]
fn srgb_to_linear(val: f32) -> f32 {
    if val <= 0.04045 {
        val / 12.92
    } else {
        ((val + 0.055) / 1.055).powf(2.4)
    }
}

/// Apply Gaussian smoothing to RGB image
#[allow(dead_code)]
fn gaussian_smooth_rgb(img: &RgbImage, sigma: f32) -> Result<RgbImage> {
    // For simplicity, we'll use a box blur approximation
    // In production, you'd want to use a proper Gaussian kernel
    let (width_, height) = img.dimensions();
    let mut smoothed = RgbImage::new(width_, height);
    let radius = (sigma * 2.0) as i32;

    for y in 0..height {
        for x in 0..width_ {
            let mut r_sum = 0.0;
            let mut g_sum = 0.0;
            let mut b_sum = 0.0;
            let mut count = 0.0;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let ny = (y as i32 + dy).max(0).min(height as i32 - 1) as u32;
                    let nx = (x as i32 + dx).max(0).min(width_ as i32 - 1) as u32;

                    let pixel = img.get_pixel(nx, ny);
                    r_sum += pixel[0] as f32;
                    g_sum += pixel[1] as f32;
                    b_sum += pixel[2] as f32;
                    count += 1.0;
                }
            }

            smoothed.put_pixel(
                x,
                y,
                image::Rgb([
                    (r_sum / count) as u8,
                    (g_sum / count) as u8,
                    (b_sum / count) as u8,
                ]),
            );
        }
    }

    Ok(smoothed)
}

/// Draw superpixel boundaries on an image
///
/// # Arguments
///
/// * `img` - Original image
/// * `labels` - Superpixel labels from SLIC
/// * `boundary_color` - Color for boundaries (RGB)
///
/// # Returns
///
/// * Image with superpixel boundaries drawn
#[allow(dead_code)]
pub fn draw_superpixel_boundaries(
    img: &DynamicImage,
    labels: &Array2<u32>,
    boundary_color: [u8; 3],
) -> RgbImage {
    let rgb = img.to_rgb8();
    let (width_, height) = rgb.dimensions();
    let mut result = rgb.clone();

    // Find boundaries
    for y in 0..height as usize - 1 {
        for x in 0..width_ as usize - 1 {
            let current = labels[[y, x]];

            // Check right and bottom neighbors
            if labels[[y, x + 1]] != current || labels[[y + 1, x]] != current {
                result.put_pixel(x as u32, y as u32, image::Rgb(boundary_color));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::DynamicImage;
    use scirs2_core::ndarray::s;

    #[test]
    fn test_slic_basic() {
        let img = DynamicImage::new_rgb8(50, 50);
        let result = slic(&img, 25, 10.0, 10, 0.0);
        assert!(result.is_ok());

        let labels = result.unwrap();
        assert_eq!(labels.dim(), (50, 50));
    }

    #[test]
    fn test_slic_invalid_segments() {
        let img = DynamicImage::new_rgb8(10, 10);
        let result = slic(&img, 0, 10.0, 10, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_rgb_to_lab() {
        // Test white
        let (l, a, b) = rgb_to_lab(255, 255, 255);
        assert!(l > 99.0 && l < 101.0); // L should be ~100 for white
        assert!(a.abs() < 1.0);
        assert!(b.abs() < 1.0);

        // Test black
        let (l, _a, _b) = rgb_to_lab(0, 0, 0);
        assert!(l < 1.0); // L should be ~0 for black
    }

    #[test]
    fn test_draw_boundaries() {
        let img = DynamicImage::new_rgb8(10, 10);
        let mut labels = Array2::zeros((10, 10));

        // Create two regions
        labels.slice_mut(s![..5, ..]).fill(0);
        labels.slice_mut(s![5.., ..]).fill(1);

        let result = draw_superpixel_boundaries(&img, &labels, [255, 0, 0]);

        // Check that boundary was drawn
        assert_eq!(result.get_pixel(0, 4), &image::Rgb([255, 0, 0]));
    }
}
