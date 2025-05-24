//! Gabor filters for texture analysis and feature extraction
//!
//! Gabor filters are linear filters used for texture analysis, edge detection,
//! and feature extraction. They are especially useful for analyzing textures
//! with specific orientations and frequencies.

use crate::error::Result;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use ndarray::{Array2, Array3};
use std::f32::consts::PI;

/// Parameters for Gabor filter
#[derive(Debug, Clone, Copy)]
pub struct GaborParams {
    /// Wavelength of the sinusoidal factor
    pub wavelength: f32,
    /// Orientation of the normal to the parallel stripes (in radians)
    pub orientation: f32,
    /// Phase offset
    pub phase: f32,
    /// Spatial aspect ratio
    pub aspect_ratio: f32,
    /// Standard deviation of the Gaussian envelope
    pub sigma: f32,
}

impl Default for GaborParams {
    fn default() -> Self {
        Self {
            wavelength: 10.0,
            orientation: 0.0,
            phase: 0.0,
            aspect_ratio: 0.5,
            sigma: 5.0,
        }
    }
}

/// Generate a Gabor kernel
///
/// # Arguments
///
/// * `params` - Gabor filter parameters
/// * `size` - Size of the kernel (should be odd)
///
/// # Returns
///
/// * 2D array containing the Gabor kernel
pub fn gabor_kernel(params: &GaborParams, size: usize) -> Array2<f32> {
    let half_size = size as i32 / 2;
    let mut kernel = Array2::zeros((size, size));

    let sigma_x = params.sigma;
    let sigma_y = params.sigma / params.aspect_ratio;

    // Pre-compute rotation values
    let cos_theta = params.orientation.cos();
    let sin_theta = params.orientation.sin();

    for y in -half_size..=half_size {
        for x in -half_size..=half_size {
            // Rotation
            let x_theta = x as f32 * cos_theta + y as f32 * sin_theta;
            let y_theta = -x as f32 * sin_theta + y as f32 * cos_theta;

            // Gaussian envelope
            let gaussian = (-0.5
                * (x_theta.powi(2) / sigma_x.powi(2) + y_theta.powi(2) / sigma_y.powi(2)))
            .exp();

            // Sinusoidal carrier
            let sinusoid = (2.0 * PI * x_theta / params.wavelength + params.phase).cos();

            let value = gaussian * sinusoid;
            kernel[[(y + half_size) as usize, (x + half_size) as usize]] = value;
        }
    }

    // Normalize kernel to have zero mean
    let mean = kernel.mean().unwrap_or(0.0);
    kernel.mapv_inplace(|v| v - mean);

    kernel
}

/// Apply Gabor filter to an image
///
/// # Arguments
///
/// * `img` - Input image
/// * `params` - Gabor filter parameters
///
/// # Returns
///
/// * Result containing filtered image
///
/// # Example
///
/// ```rust
/// use scirs2_vision::feature::{gabor_filter, GaborParams};
/// use image::DynamicImage;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let img = image::open("examples/input/input.jpg").unwrap();
/// let params = GaborParams {
///     wavelength: 10.0,
///     orientation: std::f32::consts::PI / 4.0,
///     ..Default::default()
/// };
/// let filtered = gabor_filter(&img, &params)?;
/// # Ok(())
/// # }
/// ```
pub fn gabor_filter(img: &DynamicImage, params: &GaborParams) -> Result<GrayImage> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    // Determine kernel size based on sigma
    let kernel_size = (6.0 * params.sigma).ceil() as usize;
    let kernel_size = if kernel_size % 2 == 0 {
        kernel_size + 1
    } else {
        kernel_size
    };

    let kernel = gabor_kernel(params, kernel_size);
    let half_size = kernel_size / 2;

    let mut result = ImageBuffer::new(width, height);

    // Apply convolution
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;

            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    let ix =
                        (x as i32 + kx as i32 - half_size as i32).clamp(0, width as i32 - 1) as u32;
                    let iy = (y as i32 + ky as i32 - half_size as i32).clamp(0, height as i32 - 1)
                        as u32;

                    let pixel_val = gray.get_pixel(ix, iy)[0] as f32 / 255.0;
                    sum += pixel_val * kernel[[ky, kx]];
                }
            }

            // Map to 0-255 range with proper scaling
            let value = ((sum + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
            result.put_pixel(x, y, Luma([value]));
        }
    }

    Ok(result)
}

/// Gabor filter bank for multi-scale texture analysis
#[derive(Debug, Clone)]
pub struct GaborBank {
    /// Wavelengths to use
    pub wavelengths: Vec<f32>,
    /// Orientations to use (in radians)
    pub orientations: Vec<f32>,
    /// Base parameters (phase, aspect_ratio, sigma)
    pub base_params: GaborParams,
}

impl Default for GaborBank {
    fn default() -> Self {
        Self {
            wavelengths: vec![4.0, 8.0, 16.0, 32.0],
            orientations: (0..8).map(|i| i as f32 * PI / 8.0).collect(),
            base_params: GaborParams::default(),
        }
    }
}

/// Apply Gabor filter bank to an image
///
/// # Arguments
///
/// * `img` - Input image
/// * `bank` - Gabor filter bank parameters
///
/// # Returns
///
/// * Result containing 3D array of filter responses [n_filters, height, width]
pub fn gabor_filter_bank(img: &DynamicImage, bank: &GaborBank) -> Result<Array3<f32>> {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();

    let n_filters = bank.wavelengths.len() * bank.orientations.len();
    let mut responses = Array3::zeros((n_filters, height as usize, width as usize));

    let mut filter_idx = 0;
    for &wavelength in &bank.wavelengths {
        for &orientation in &bank.orientations {
            let params = GaborParams {
                wavelength,
                orientation,
                phase: bank.base_params.phase,
                aspect_ratio: bank.base_params.aspect_ratio,
                sigma: wavelength / 2.0, // Adjust sigma based on wavelength
            };

            let filtered = gabor_filter(img, &params)?;

            // Copy to responses array
            for y in 0..height as usize {
                for x in 0..width as usize {
                    responses[[filter_idx, y, x]] =
                        filtered.get_pixel(x as u32, y as u32)[0] as f32 / 255.0;
                }
            }

            filter_idx += 1;
        }
    }

    Ok(responses)
}

/// Extract Gabor features from an image region
///
/// # Arguments
///
/// * `responses` - Gabor filter bank responses
/// * `x` - X coordinate of region center
/// * `y` - Y coordinate of region center
/// * `window_size` - Size of the window to extract features from
///
/// # Returns
///
/// * Feature vector containing mean and standard deviation of each filter response
pub fn extract_gabor_features(
    responses: &Array3<f32>,
    x: usize,
    y: usize,
    window_size: usize,
) -> Vec<f32> {
    let (n_filters, height, width) = responses.dim();
    let half_window = window_size / 2;

    let mut features = Vec::with_capacity(n_filters * 2);

    for f in 0..n_filters {
        let mut values = Vec::new();

        // Extract window values
        for dy in 0..window_size {
            for dx in 0..window_size {
                let px = (x + dx).saturating_sub(half_window).min(width - 1);
                let py = (y + dy).saturating_sub(half_window).min(height - 1);
                values.push(responses[[f, py, px]]);
            }
        }

        // Compute mean and standard deviation
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        features.push(mean);
        features.push(std_dev);
    }

    features
}

/// Compute Gabor energy image
///
/// # Arguments
///
/// * `img` - Input image
/// * `params` - Gabor filter parameters
///
/// # Returns
///
/// * Result containing energy image (magnitude of complex Gabor response)
pub fn gabor_energy(img: &DynamicImage, params: &GaborParams) -> Result<GrayImage> {
    // Real part (cosine)
    let real_response = gabor_filter(img, params)?;

    // Imaginary part (sine)
    let mut sine_params = *params;
    sine_params.phase = params.phase + PI / 2.0;
    let imag_response = gabor_filter(img, &sine_params)?;

    let (width, height) = real_response.dimensions();
    let mut energy = ImageBuffer::new(width, height);

    // Compute energy as magnitude of complex response
    for y in 0..height {
        for x in 0..width {
            let real = real_response.get_pixel(x, y)[0] as f32 / 255.0 - 0.5;
            let imag = imag_response.get_pixel(x, y)[0] as f32 / 255.0 - 0.5;
            let magnitude = (real.powi(2) + imag.powi(2)).sqrt();
            let value = (magnitude * 255.0).clamp(0.0, 255.0) as u8;
            energy.put_pixel(x, y, Luma([value]));
        }
    }

    Ok(energy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gabor_kernel() {
        let params = GaborParams::default();
        let kernel = gabor_kernel(&params, 21);

        assert_eq!(kernel.dim(), (21, 21));
        // Kernel should have approximately zero mean
        assert!(kernel.mean().unwrap_or(0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gabor_filter() {
        let img = DynamicImage::new_luma8(50, 50);
        let params = GaborParams::default();
        let result = gabor_filter(&img, &params);

        assert!(result.is_ok());
        let filtered = result.unwrap();
        assert_eq!(filtered.dimensions(), (50, 50));
    }

    #[test]
    fn test_gabor_bank() {
        let img = DynamicImage::new_luma8(30, 30);
        let bank = GaborBank {
            wavelengths: vec![8.0, 16.0],
            orientations: vec![0.0, PI / 4.0],
            base_params: GaborParams::default(),
        };

        let result = gabor_filter_bank(&img, &bank);
        assert!(result.is_ok());

        let responses = result.unwrap();
        assert_eq!(responses.dim(), (4, 30, 30)); // 2 wavelengths × 2 orientations
    }

    #[test]
    fn test_gabor_features() {
        let mut responses = Array3::zeros((4, 20, 20));
        // Fill with some test data
        for i in 0..4 {
            responses
                .slice_mut(ndarray::s![i, .., ..])
                .fill(i as f32 * 0.25);
        }

        let features = extract_gabor_features(&responses, 10, 10, 5);
        assert_eq!(features.len(), 8); // 4 filters × 2 features (mean, std)
    }
}
