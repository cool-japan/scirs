//! Laplacian of Gaussian (LoG) blob detector
//!
//! The LoG detector finds blob-like structures by combining Gaussian smoothing
//! with Laplacian edge detection.

use crate::error::Result;
use image::{DynamicImage, GrayImage, Luma};
use scirs2_core::ndarray::Array2;

/// Configuration for LoG blob detection
#[derive(Clone, Debug)]
pub struct LogBlobConfig {
    /// Number of scale levels
    pub num_scales: usize,
    /// Minimum sigma for Gaussian
    pub min_sigma: f64,
    /// Maximum sigma for Gaussian
    pub max_sigma: f64,
    /// Threshold for blob detection
    pub threshold: f32,
    /// Whether to use non-maximum suppression
    pub non_max_suppression: bool,
}

impl Default for LogBlobConfig {
    fn default() -> Self {
        Self {
            num_scales: 10,
            min_sigma: 1.0,
            max_sigma: 50.0,
            threshold: 0.01,
            non_max_suppression: true,
        }
    }
}

/// Represents a detected blob
#[derive(Clone, Debug)]
pub struct LogBlob {
    /// X coordinate of the blob center
    pub x: usize,
    /// Y coordinate of the blob center
    pub y: usize,
    /// Sigma value (scale) at which the blob was detected
    pub sigma: f64,
    /// Response strength of the blob
    pub response: f32,
}

/// Detect blobs using Laplacian of Gaussian
#[allow(dead_code)]
pub fn log_blob_detect(img: &DynamicImage, config: LogBlobConfig) -> Result<Vec<LogBlob>> {
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
    let mut scale_space = Vec::new();

    // Generate scale space
    for i in 0..config.num_scales {
        let sigma = config.min_sigma
            * (config.max_sigma / config.min_sigma).powf(i as f64 / (config.num_scales - 1) as f64);
        let log_response = apply_log(&array, sigma as f32)?;
        scale_space.push((log_response, sigma));
    }

    // Find local maxima in scale space
    for scale_idx in 1..scale_space.len() - 1 {
        let (curr_response, sigma) = &scale_space[scale_idx];
        let (prev_response_, _) = &scale_space[scale_idx - 1];
        let (next_response_, _) = &scale_space[scale_idx + 1];

        // Find local maxima in current scale
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let val = curr_response[[y, x]];

                if val.abs() < config.threshold {
                    continue;
                }

                // Check if it's a local maximum in scale space
                if is_local_maximum(prev_response_, curr_response, next_response_, x, y) {
                    blobs.push(LogBlob {
                        x,
                        y,
                        sigma: *sigma,
                        response: val.abs(),
                    });
                }
            }
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
pub fn log_blobs_to_image(blobs: &[LogBlob], width: u32, height: u32) -> Result<GrayImage> {
    let mut img = GrayImage::new(width, height);

    for blob in blobs {
        // Draw a circle around each blob, radius proportional to sigma
        let radius = (blob.sigma * std::f64::consts::SQRT_2) as i32;
        draw_circle(&mut img, blob.x as i32, blob.y as i32, radius);
    }

    Ok(img)
}

// Helper functions

#[allow(dead_code)]
fn apply_log(img: &Array2<f32>, sigma: f32) -> Result<Array2<f32>> {
    // Apply Gaussian blur
    let blurred = gaussian_blur(img, sigma)?;

    // Apply Laplacian
    let log_result = apply_laplacian(&blurred)?;

    // Normalize by sigma^2 for scale invariance
    let mut normalized = log_result;
    let sigma2 = sigma * sigma;
    for val in normalized.iter_mut() {
        *val *= sigma2;
    }

    Ok(normalized)
}

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
fn apply_laplacian(img: &Array2<f32>) -> Result<Array2<f32>> {
    let (height, width) = img.dim();
    let mut result = Array2::zeros((height, width));

    // 3x3 Laplacian kernel (8-connected)
    let kernel = [[-1.0, -1.0, -1.0], [-1.0, 8.0, -1.0], [-1.0, -1.0, -1.0]];

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut sum = 0.0;

            for (ky, row) in kernel.iter().enumerate() {
                for (kx, &k_val) in row.iter().enumerate() {
                    let px = (x as i32 + kx as i32 - 1) as usize;
                    let py = (y as i32 + ky as i32 - 1) as usize;
                    sum += img[[py, px]] * k_val;
                }
            }

            result[[y, x]] = sum;
        }
    }

    Ok(result)
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

            for (k, &kernel_val) in kernel.iter().enumerate().take(ksize) {
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

            for (k, &kernel_val) in kernel.iter().enumerate().take(ksize) {
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
fn is_local_maximum(
    prev: &Array2<f32>,
    curr: &Array2<f32>,
    next: &Array2<f32>,
    x: usize,
    y: usize,
) -> bool {
    let val = curr[[y, x]];

    // Check 8-connected neighbors in current scale
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }

            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if curr[[ny, nx]] >= val {
                return false;
            }
        }
    }

    // Check center pixel in adjacent scales
    if prev[[y, x]] >= val || next[[y, x]] >= val {
        return false;
    }

    true
}

#[allow(dead_code)]
fn non_max_suppression(mut blobs: Vec<LogBlob>) -> Vec<LogBlob> {
    // Sort by response
    blobs.sort_by(|a, b| {
        b.response
            .partial_cmp(&a.response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept = Vec::new();
    let mut suppressed = vec![false; blobs.len()];

    for i in 0..blobs.len() {
        if suppressed[i] {
            continue;
        }

        kept.push(blobs[i].clone());

        // Suppress nearby blobs
        for j in i + 1..blobs.len() {
            if suppressed[j] {
                continue;
            }

            let dx = blobs[i].x as f64 - blobs[j].x as f64;
            let dy = blobs[i].y as f64 - blobs[j].y as f64;
            let dist = (dx * dx + dy * dy).sqrt();

            // Suppress if too close (considering scale)
            let min_dist = (blobs[i].sigma + blobs[j].sigma) * std::f64::consts::SQRT_2;
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
    fn test_log_blob_config_default() {
        let config = LogBlobConfig::default();
        assert_eq!(config.num_scales, 10);
        assert_eq!(config.min_sigma, 1.0);
        assert_eq!(config.max_sigma, 50.0);
        assert_eq!(config.threshold, 0.01);
        assert!(config.non_max_suppression);
    }

    #[test]
    fn test_gaussian_kernel_properties() {
        let kernel = gaussian_kernel(5, 1.0);
        assert_eq!(kernel.len(), 5);

        // Kernel should sum to 1
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Should be symmetric
        assert!((kernel[0] - kernel[4]).abs() < 1e-6);
        assert!((kernel[1] - kernel[3]).abs() < 1e-6);

        // Center should be the maximum
        assert!(kernel[2] > kernel[1]);
        assert!(kernel[2] > kernel[3]);
    }

    #[test]
    fn test_log_blob_creation() {
        let blob = LogBlob {
            x: 100,
            y: 150,
            sigma: 3.5,
            response: 0.85,
        };

        assert_eq!(blob.x, 100);
        assert_eq!(blob.y, 150);
        assert_eq!(blob.sigma, 3.5);
        assert_eq!(blob.response, 0.85);
    }
}
