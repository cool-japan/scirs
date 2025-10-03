//! Difference of Gaussians (DoG) blob detector
//!
//! The DoG detector finds blob-like structures at multiple scales
//! by computing the difference between two Gaussian blurs.

use crate::error::Result;
use image::{DynamicImage, GrayImage, Luma};
use scirs2_core::ndarray::Array2;
use std::cmp::Ordering;

/// Configuration for DoG detection
#[derive(Clone, Debug)]
pub struct DogConfig {
    /// Number of scale levels
    pub num_octaves: usize,
    /// Number of scale levels per octave
    pub scales_per_octave: usize,
    /// Initial sigma for Gaussian blurring
    pub initial_sigma: f64,
    /// Threshold for blob detection
    pub threshold: f32,
    /// Whether to use non-maximum suppression
    pub non_max_suppression: bool,
}

impl Default for DogConfig {
    fn default() -> Self {
        Self {
            num_octaves: 4,
            scales_per_octave: 4,
            initial_sigma: 1.6,
            threshold: 0.01,
            non_max_suppression: true,
        }
    }
}

/// Represents a detected blob
#[derive(Copy, Clone, Debug)]
pub struct Blob {
    /// X coordinate of the blob center
    pub x: usize,
    /// Y coordinate of the blob center
    pub y: usize,
    /// Scale at which the blob was detected
    pub scale: f64,
    /// Response strength of the blob
    pub response: f32,
}

/// Detect blobs using Difference of Gaussians
#[allow(dead_code)]
pub fn dog_detect(img: &DynamicImage, config: DogConfig) -> Result<Vec<Blob>> {
    let gray = img.to_luma8();
    let (height, width) = (gray.height() as usize, gray.width() as usize);

    // Convert to ndarray
    let mut array = Array2::<f32>::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            array[[y, x]] = gray.get_pixel(x as u32, y as u32).0[0] as f32 / 255.0;
        }
    }

    let mut blobs = Vec::new();

    let mut current_array = array.clone();

    // Process each octave
    for octave in 0..config.num_octaves {
        let scale_factor = 2.0f64.powi(octave as i32);
        let (current_height, current_width) = current_array.dim();
        let mut scale_space = Vec::new();

        // Build scale space for this octave
        for s in 0..=config.scales_per_octave {
            let sigma =
                config.initial_sigma * (2.0f64).powf(s as f64 / config.scales_per_octave as f64);
            let blurred = gaussian_blur(&current_array, sigma as f32)?;
            scale_space.push(blurred);
        }

        // Compute DoG images
        let mut dog_images = Vec::new();
        for i in 0..scale_space.len() - 1 {
            let mut dog = Array2::<f32>::zeros((current_height, current_width));
            for y in 0..current_height {
                for x in 0..current_width {
                    dog[[y, x]] = scale_space[i + 1][[y, x]] - scale_space[i][[y, x]];
                }
            }
            dog_images.push(dog);
        }

        // Find local extrema
        for (scale_idx, windows) in dog_images.windows(3).enumerate() {
            let prev = &windows[0];
            let curr = &windows[1];
            let next = &windows[2];

            // Check each pixel
            for y in 1..current_height - 1 {
                for x in 1..current_width - 1 {
                    let val = curr[[y, x]];

                    if val.abs() < config.threshold {
                        continue;
                    }

                    // Check if it's a local extremum
                    if is_local_extremum(prev, curr, next, x, y) {
                        let sigma = config.initial_sigma
                            * scale_factor
                            * (2.0f64)
                                .powf((scale_idx + 1) as f64 / config.scales_per_octave as f64);

                        blobs.push(Blob {
                            x: x * scale_factor as usize,
                            y: y * scale_factor as usize,
                            scale: sigma,
                            response: val.abs(),
                        });
                    }
                }
            }
        }

        // Downsample for next octave
        if octave < config.num_octaves - 1 {
            current_array = downsample(&current_array);
        }
    }

    // Apply non-maximum suppression if requested
    if config.non_max_suppression {
        blobs = non_max_suppression(blobs);
    }

    Ok(blobs)
}

/// Convert blobs to image
#[allow(dead_code)]
pub fn blobs_to_image(blobs: &[Blob], width: u32, height: u32) -> Result<GrayImage> {
    let mut img = GrayImage::new(width, height);

    for blob in blobs {
        // Draw a circle around each blob
        let radius = (blob.scale * 2.0) as i32;
        draw_circle(&mut img, blob.x as i32, blob.y as i32, radius);
    }

    Ok(img)
}

// Helper functions

#[allow(dead_code)]
fn gaussian_blur(img: &Array2<f32>, sigma: f32) -> Result<Array2<f32>> {
    let kernel_size = ((6.0 * sigma) as usize) | 1; // Make it odd
    let kernel = gaussian_kernel(kernel_size, sigma);

    // Apply separable convolution
    let temp = convolve_1d_horizontal(img, &kernel)?;
    convolve_1d_vertical(&temp, &kernel)
}

#[allow(dead_code)]
fn gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
    let mut kernel = vec![0.0; size];
    let center = size / 2;
    let s2 = 2.0 * sigma * sigma;

    let mut sum = 0.0;
    for (i, k) in kernel.iter_mut().enumerate().take(size) {
        let x = i as i32 - center as i32;
        let val = (-((x * x) as f32) / s2).exp();
        *k = val;
        sum += val;
    }

    // Normalize
    for val in &mut kernel {
        *val /= sum;
    }

    kernel
}

#[allow(dead_code)]
fn convolve_1d_horizontal(img: &Array2<f32>, kernel: &[f32]) -> Result<Array2<f32>> {
    let (height, width) = img.dim();
    let mut result = Array2::zeros((height, width));
    let ksize = kernel.len();
    let kcenter = ksize / 2;

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for (k, kernel_val) in kernel.iter().enumerate().take(ksize) {
                let ix = x as i32 + k as i32 - kcenter as i32;
                if ix >= 0 && ix < width as i32 {
                    sum += img[[y, ix as usize]] * kernel_val;
                }
            }

            result[[y, x]] = sum;
        }
    }

    Ok(result)
}

#[allow(dead_code)]
fn convolve_1d_vertical(img: &Array2<f32>, kernel: &[f32]) -> Result<Array2<f32>> {
    let (height, width) = img.dim();
    let mut result = Array2::zeros((height, width));
    let ksize = kernel.len();
    let kcenter = ksize / 2;

    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for (k, kernel_val) in kernel.iter().enumerate().take(ksize) {
                let iy = y as i32 + k as i32 - kcenter as i32;
                if iy >= 0 && iy < height as i32 {
                    sum += img[[iy as usize, x]] * kernel_val;
                }
            }

            result[[y, x]] = sum;
        }
    }

    Ok(result)
}

#[allow(dead_code)]
fn is_local_extremum(
    prev: &Array2<f32>,
    curr: &Array2<f32>,
    next: &Array2<f32>,
    x: usize,
    y: usize,
) -> bool {
    let val = curr[[y, x]];
    let is_max = val > 0.0;

    // Check 26 neighbors (3x3x3 - center)
    for dy in -1..=1 {
        for dx in -1..=1 {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            // Current scale
            if !(dx == 0 && dy == 0)
                && ((is_max && curr[[ny, nx]] >= val) || (!is_max && curr[[ny, nx]] <= val))
            {
                return false;
            }

            // Previous scale
            if (is_max && prev[[ny, nx]] >= val) || (!is_max && prev[[ny, nx]] <= val) {
                return false;
            }

            // Next scale
            if (is_max && next[[ny, nx]] >= val) || (!is_max && next[[ny, nx]] <= val) {
                return false;
            }
        }
    }

    true
}

#[allow(dead_code)]
fn downsample(img: &Array2<f32>) -> Array2<f32> {
    let (height, width) = img.dim();
    let new_height = height / 2;
    let new_width = width / 2;
    let mut result = Array2::zeros((new_height, new_width));

    for y in 0..new_height {
        for x in 0..new_width {
            result[[y, x]] = img[[y * 2, x * 2]];
        }
    }

    result
}

#[allow(dead_code)]
fn non_max_suppression(mut blobs: Vec<Blob>) -> Vec<Blob> {
    // Sort by response
    blobs.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(Ordering::Equal)
    });

    let mut kept = Vec::new();
    let mut suppressed = vec![false; blobs.len()];

    for i in 0..blobs.len() {
        if suppressed[i] {
            continue;
        }

        kept.push(blobs[i]);

        // Suppress nearby blobs
        for j in i + 1..blobs.len() {
            if suppressed[j] {
                continue;
            }

            let dx = blobs[i].x as f64 - blobs[j].x as f64;
            let dy = blobs[i].y as f64 - blobs[j].y as f64;
            let dist = (dx * dx + dy * dy).sqrt();

            // Suppress if too close (considering scale)
            let min_dist = (blobs[i].scale + blobs[j].scale) * 2.0;
            if dist < min_dist {
                suppressed[j] = true;
            }
        }
    }

    kept
}

#[allow(dead_code)]
fn draw_circle(img: &mut GrayImage, cx: i32, cy: i32, radius: i32) {
    let (width, height) = (img.width() as i32, img.height() as i32);

    for angle in 0..360 {
        let rad = (angle as f32).to_radians();
        let x = cx + (radius as f32 * rad.cos()) as i32;
        let y = cy + (radius as f32 * rad.sin()) as i32;

        if x >= 0 && x < width && y >= 0 && y < height {
            img.put_pixel(x as u32, y as u32, Luma([255]));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_kernel() {
        let kernel = gaussian_kernel(5, 1.0);
        assert_eq!(kernel.len(), 5);

        // Kernel should sum to 1
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be symmetric
        assert!((kernel[0] - kernel[4]).abs() < 1e-6);
        assert!((kernel[1] - kernel[3]).abs() < 1e-6);
    }

    #[test]
    fn test_dog_config_default() {
        let config = DogConfig::default();
        assert_eq!(config.num_octaves, 4);
        assert_eq!(config.scales_per_octave, 4);
        assert_eq!(config.initial_sigma, 1.6);
        assert_eq!(config.threshold, 0.01);
        assert!(config.non_max_suppression);
    }

    #[test]
    fn test_blob_creation() {
        let blob = Blob {
            x: 100,
            y: 150,
            scale: 2.5,
            response: 0.95,
        };

        assert_eq!(blob.x, 100);
        assert_eq!(blob.y, 150);
        assert_eq!(blob.scale, 2.5);
        assert_eq!(blob.response, 0.95);
    }
}
