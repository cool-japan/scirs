//! Interpolation methods for image resampling
//!
//! This module provides various interpolation methods for image
//! resampling, including nearest-neighbor, bilinear, bicubic,
//! Lanczos, and edge-preserving interpolation.
//!
//! The edge-preserving interpolation method is particularly useful for:
//!
//! 1. Maintaining sharp edges while reducing noise
//! 2. Preserving detailed structures during upscaling
//! 3. Reducing artifacts in images with text or line art
//! 4. Improving quality of resized natural images with distinct boundaries

use crate::error::Result;
use image::{DynamicImage, GenericImageView, ImageBuffer, Pixel, Rgba};
use ndarray::{Array1, Array2};

/// Interpolation methods for image resampling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// Nearest-neighbor interpolation (fast but low quality)
    Nearest,
    /// Bilinear interpolation (good balance of speed and quality)
    Bilinear,
    /// Bicubic interpolation (higher quality)
    Bicubic,
    /// Lanczos interpolation (best quality)
    Lanczos3,
    /// Edge-preserving interpolation (preserves edges while smoothing)
    EdgePreserving,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        Self::Bilinear
    }
}

/// Resize an image using the specified interpolation method
///
/// # Arguments
///
/// * `src` - Source image
/// * `width` - New width
/// * `height` - New height
/// * `method` - Interpolation method
///
/// # Returns
///
/// * Result containing resized image
pub fn resize(
    src: &DynamicImage,
    width: u32,
    height: u32,
    method: InterpolationMethod,
) -> Result<DynamicImage> {
    // Get source dimensions
    let (src_width, src_height) = src.dimensions();

    // If dimensions are the same, just return a copy
    if src_width == width && src_height == height {
        return Ok(src.clone());
    }

    // Create output image
    let mut dst: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    // Scale factors
    let scale_x = src_width as f64 / width as f64;
    let scale_y = src_height as f64 / height as f64;

    match method {
        InterpolationMethod::Nearest => {
            // Nearest neighbor interpolation
            for y in 0..height {
                for x in 0..width {
                    // Calculate source coordinates
                    let src_x = (x as f64 * scale_x).floor() as u32;
                    let src_y = (y as f64 * scale_y).floor() as u32;

                    // Clamp to source dimensions
                    let src_x = src_x.min(src_width - 1);
                    let src_y = src_y.min(src_height - 1);

                    // Sample nearest pixel
                    let pixel = src.get_pixel(src_x, src_y);
                    dst.put_pixel(x, y, pixel);
                }
            }
        }
        InterpolationMethod::Bilinear => {
            // Bilinear interpolation
            for y in 0..height {
                for x in 0..width {
                    // Calculate source coordinates
                    let src_x = x as f64 * scale_x;
                    let src_y = y as f64 * scale_y;

                    // Get pixel value using bilinear interpolation
                    let pixel = bilinear_interpolate(src, src_x, src_y);
                    dst.put_pixel(x, y, pixel);
                }
            }
        }
        InterpolationMethod::Bicubic => {
            // Bicubic interpolation
            for y in 0..height {
                for x in 0..width {
                    // Calculate source coordinates
                    let src_x = x as f64 * scale_x;
                    let src_y = y as f64 * scale_y;

                    // Get pixel value using bicubic interpolation
                    let pixel = bicubic_interpolate(src, src_x, src_y);
                    dst.put_pixel(x, y, pixel);
                }
            }
        }
        InterpolationMethod::Lanczos3 => {
            // Lanczos interpolation (a=3)
            for y in 0..height {
                for x in 0..width {
                    // Calculate source coordinates
                    let src_x = x as f64 * scale_x;
                    let src_y = y as f64 * scale_y;

                    // Get pixel value using Lanczos interpolation
                    let pixel = lanczos_interpolate(src, src_x, src_y, 3);
                    dst.put_pixel(x, y, pixel);
                }
            }
        }
        InterpolationMethod::EdgePreserving => {
            // Edge-preserving interpolation
            return resize_edge_preserving(src, width, height);
        }
    }

    Ok(DynamicImage::ImageRgba8(dst))
}

/// Bilinear interpolation for a single pixel
///
/// # Arguments
///
/// * `src` - Source image
/// * `x` - X coordinate (fractional)
/// * `y` - Y coordinate (fractional)
///
/// # Returns
///
/// * Interpolated pixel value
fn bilinear_interpolate(src: &DynamicImage, x: f64, y: f64) -> Rgba<u8> {
    let (width, height) = src.dimensions();

    // Get integer and fractional components
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let dx = x - x0 as f64;
    let dy = y - y0 as f64;

    // Get the four surrounding pixels
    let p00 = src.get_pixel(x0, y0).to_rgba();
    let p01 = src.get_pixel(x0, y1).to_rgba();
    let p10 = src.get_pixel(x1, y0).to_rgba();
    let p11 = src.get_pixel(x1, y1).to_rgba();

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

        // Clamp to valid range
        result[c] = value.round().clamp(0.0, 255.0) as u8;
    }

    Rgba(result)
}

/// Cubic Hermite spline
fn cubic_hermite(x: f64) -> f64 {
    // Cubic Hermite spline kernel
    let x = x.abs();
    if x < 1.0 {
        return (2.0 - x * x * (3.0 - 2.0 * x)).clamp(0.0, 1.0);
    } else if x < 2.0 {
        return (4.0 - 8.0 * x + 5.0 * x * x - x * x * x).clamp(0.0, 1.0) / 2.0;
    }
    0.0
}

/// Bicubic interpolation for a single pixel
///
/// # Arguments
///
/// * `src` - Source image
/// * `x` - X coordinate (fractional)
/// * `y` - Y coordinate (fractional)
///
/// # Returns
///
/// * Interpolated pixel value
fn bicubic_interpolate(src: &DynamicImage, x: f64, y: f64) -> Rgba<u8> {
    let (width, height) = src.dimensions();

    // Base coordinates
    let x_base = x.floor() as i32;
    let y_base = y.floor() as i32;

    // Calculate weights
    let dx = x - x_base as f64;
    let dy = y - y_base as f64;

    // Pre-calculate cubic weights
    let wx = [
        cubic_hermite(dx + 1.0),
        cubic_hermite(dx),
        cubic_hermite(1.0 - dx),
        cubic_hermite(2.0 - dx),
    ];

    let wy = [
        cubic_hermite(dy + 1.0),
        cubic_hermite(dy),
        cubic_hermite(1.0 - dy),
        cubic_hermite(2.0 - dy),
    ];

    // Sample 4x4 grid around the point
    let mut result = [0.0; 4];

    for c in 0..4 {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for ky in 0..4 {
            let y_sample = y_base - 1 + ky;

            // Skip outside pixels
            if y_sample < 0 || y_sample >= height as i32 {
                continue;
            }

            for kx in 0..4 {
                let x_sample = x_base - 1 + kx;

                // Skip outside pixels
                if x_sample < 0 || x_sample >= width as i32 {
                    continue;
                }

                // Calculate weight
                let weight = wx[kx as usize] * wy[ky as usize];
                if weight > 0.0 {
                    // Get pixel value
                    let pixel = src.get_pixel(x_sample as u32, y_sample as u32).to_rgba();
                    sum += weight * pixel[c] as f64;
                    weight_sum += weight;
                }
            }
        }

        // Normalize and clamp
        if weight_sum > 0.0 {
            result[c] = (sum / weight_sum).round();
            result[c] = result[c].clamp(0.0, 255.0);
        }
    }

    Rgba([
        result[0] as u8,
        result[1] as u8,
        result[2] as u8,
        result[3] as u8,
    ])
}

/// Lanczos kernel function
fn lanczos(x: f64, a: i32) -> f64 {
    if x.abs() < f64::EPSILON {
        return 1.0;
    }
    if x.abs() >= a as f64 {
        return 0.0;
    }

    let a_f64 = a as f64;
    let pi_x = std::f64::consts::PI * x;
    (a_f64 * (pi_x / a_f64).sin() * (pi_x).sin()) / (pi_x * pi_x)
}

/// Lanczos interpolation for a single pixel
///
/// # Arguments
///
/// * `src` - Source image
/// * `x` - X coordinate (fractional)
/// * `y` - Y coordinate (fractional)
/// * `a` - Size of the Lanczos kernel (typically 2 or 3)
///
/// # Returns
///
/// * Interpolated pixel value
fn lanczos_interpolate(src: &DynamicImage, x: f64, y: f64, a: i32) -> Rgba<u8> {
    let (width, height) = src.dimensions();

    // Base coordinates
    let x_base = x.floor() as i32;
    let y_base = y.floor() as i32;

    // Calculate fractional parts
    let dx = x - x_base as f64;
    let dy = y - y_base as f64;

    // Setup kernel size
    let kernel_width = 2 * a;
    let mut weights_x = vec![0.0; kernel_width as usize];
    let mut weights_y = vec![0.0; kernel_width as usize];

    // Compute weights for x direction
    for k in 0..kernel_width {
        let kx = k - a + 1;
        weights_x[k as usize] = lanczos(dx - kx as f64, a);
    }

    // Compute weights for y direction
    for k in 0..kernel_width {
        let ky = k - a + 1;
        weights_y[k as usize] = lanczos(dy - ky as f64, a);
    }

    // Normalize weights
    let sum_wx: f64 = weights_x.iter().sum();
    let sum_wy: f64 = weights_y.iter().sum();

    for k in 0..kernel_width as usize {
        weights_x[k] /= sum_wx;
        weights_y[k] /= sum_wy;
    }

    // Compute interpolated color
    let mut result = [0.0; 4];

    for c in 0..4 {
        let mut sum = 0.0;
        let mut weight_sum = 0.0;

        for ky in 0..kernel_width {
            let y_sample = y_base + ky - a + 1;

            // Skip outside pixels
            if y_sample < 0 || y_sample >= height as i32 {
                continue;
            }

            for kx in 0..kernel_width {
                let x_sample = x_base + kx - a + 1;

                // Skip outside pixels
                if x_sample < 0 || x_sample >= width as i32 {
                    continue;
                }

                // Calculate separable kernel weight
                let weight = weights_x[kx as usize] * weights_y[ky as usize];

                // Add weighted contribution
                let pixel = src.get_pixel(x_sample as u32, y_sample as u32).to_rgba();
                sum += weight * pixel[c] as f64;
                weight_sum += weight;
            }
        }

        // Normalize and clamp
        if weight_sum > 0.0 {
            result[c] = (sum / weight_sum).round();
            result[c] = result[c].clamp(0.0, 255.0);
        }
    }

    Rgba([
        result[0] as u8,
        result[1] as u8,
        result[2] as u8,
        result[3] as u8,
    ])
}

/// Helper function to create 1D kernels for separable convolution
///
/// # Arguments
///
/// * `kernel_size` - Size of the kernel
///
/// # Returns
///
/// * Kernel as a 1D array
fn create_kernel(kernel_func: fn(f64) -> f64, kernel_size: usize, scale: f64) -> Array1<f64> {
    let mut kernel = Array1::zeros(kernel_size);
    let radius = (kernel_size as f64 - 1.0) / 2.0;

    for i in 0..kernel_size {
        let x = (i as f64 - radius) / scale;
        kernel[i] = kernel_func(x);
    }

    // Normalize kernel
    let sum = kernel.sum();
    if sum > 0.0 {
        kernel.mapv_inplace(|x| x / sum);
    }

    kernel
}

/// Calculate convolution of a row or column with a 1D kernel
///
/// # Arguments
///
/// * `src` - Image to convolve
/// * `kernel` - 1D convolution kernel
/// * `horizontal` - If true, convolve along rows (x), otherwise along columns (y)
///
/// # Returns
///
/// * Convolved image
pub fn convolve_1d(
    src: &Array2<f64>,
    kernel: &Array1<f64>,
    horizontal: bool,
) -> Result<Array2<f64>> {
    let (height, width) = src.dim();
    let mut dst = Array2::zeros((height, width));

    let k_size = kernel.len();
    let k_radius = (k_size / 2) as isize;

    if horizontal {
        // Convolve rows (horizontally)
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in 0..k_size {
                    let kx = x as isize + (k as isize - k_radius);

                    // Handle boundary
                    if kx >= 0 && kx < width as isize {
                        let w = kernel[k];
                        sum += w * src[[y, kx as usize]];
                        weight_sum += w;
                    }
                }

                // Normalize
                if weight_sum > 0.0 {
                    dst[[y, x]] = sum / weight_sum;
                }
            }
        }
    } else {
        // Convolve columns (vertically)
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for k in 0..k_size {
                    let ky = y as isize + (k as isize - k_radius);

                    // Handle boundary
                    if ky >= 0 && ky < height as isize {
                        let w = kernel[k];
                        sum += w * src[[ky as usize, x]];
                        weight_sum += w;
                    }
                }

                // Normalize
                if weight_sum > 0.0 {
                    dst[[y, x]] = sum / weight_sum;
                }
            }
        }
    }

    Ok(dst)
}

/// Resize image using separable convolution method
///
/// This provides high-quality resizing by first convolving the image
/// with an appropriate filter kernel and then sampling at the desired locations.
///
/// # Arguments
///
/// * `src` - Source image
/// * `width` - Target width
/// * `height` - Target height
/// * `kernel_func` - Function implementing the filter kernel
/// * `kernel_size` - Size of the kernel
///
/// # Returns
///
/// * Resized image
pub fn resize_convolution(
    src: &DynamicImage,
    width: u32,
    height: u32,
    kernel_func: fn(f64) -> f64,
    kernel_size: usize,
) -> Result<DynamicImage> {
    let (src_width, src_height) = src.dimensions();

    // If dimensions are the same, just return a copy
    if src_width == width && src_height == height {
        return Ok(src.clone());
    }

    // Calculate scale factors
    let scale_x = width as f64 / src_width as f64;
    let scale_y = height as f64 / src_height as f64;

    // Create destination image
    let mut dst: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    // Process each channel separately
    for c in 0..4 {
        // Extract channel
        let mut channel = Array2::zeros((src_height as usize, src_width as usize));
        for y in 0..src_height {
            for x in 0..src_width {
                let pixel = src.get_pixel(x, y).to_rgba();
                channel[[y as usize, x as usize]] = pixel[c] as f64;
            }
        }

        // Create kernels for horizontal and vertical convolution
        let scale_factor_x = if scale_x < 1.0 { scale_x } else { 1.0 };
        let scale_factor_y = if scale_y < 1.0 { scale_y } else { 1.0 };

        let kernel_x = create_kernel(kernel_func, kernel_size, scale_factor_x);
        let kernel_y = create_kernel(kernel_func, kernel_size, scale_factor_y);

        // Apply horizontal convolution
        let temp = convolve_1d(&channel, &kernel_x, true)?;

        // Apply vertical convolution
        let filtered = convolve_1d(&temp, &kernel_y, false)?;

        // Sample the filtered image at target resolution
        for y in 0..height {
            for x in 0..width {
                // Calculate source coordinates
                let src_x = (x as f64 + 0.5) / scale_x - 0.5;
                let src_y = (y as f64 + 0.5) / scale_y - 0.5;

                // Convert to integer and fractional components
                let x0 = src_x.floor() as i32;
                let y0 = src_y.floor() as i32;
                let dx = src_x - x0 as f64;
                let dy = src_y - y0 as f64;

                // Calculate bilinear interpolation weights
                let w00 = (1.0 - dx) * (1.0 - dy);
                let w01 = (1.0 - dx) * dy;
                let w10 = dx * (1.0 - dy);
                let w11 = dx * dy;

                // Get pixel values (with boundary handling)
                let mut value = 0.0;
                let mut weight_sum = 0.0;

                // Check and accumulate all valid samples
                for ky in 0..2 {
                    let y_index = y0 + ky;
                    if y_index >= 0 && y_index < src_height as i32 {
                        for kx in 0..2 {
                            let x_index = x0 + kx;
                            if x_index >= 0 && x_index < src_width as i32 {
                                let w = match (kx, ky) {
                                    (0, 0) => w00,
                                    (0, 1) => w01,
                                    (1, 0) => w10,
                                    (1, 1) => w11,
                                    _ => 0.0,
                                };

                                if w > 0.0 {
                                    value += w * filtered[[y_index as usize, x_index as usize]];
                                    weight_sum += w;
                                }
                            }
                        }
                    }
                }

                // Normalize if needed
                if weight_sum > 0.0 {
                    value /= weight_sum;
                }

                // Update pixel in destination image
                let mut pixel = dst.get_pixel_mut(x, y).to_rgba();
                pixel[c] = value.round().clamp(0.0, 255.0) as u8;
                dst.put_pixel(x, y, pixel);
            }
        }
    }

    Ok(DynamicImage::ImageRgba8(dst))
}

// Lanczos filter kernel function
fn lanczos_kernel(x: f64) -> f64 {
    lanczos(x, 3)
}

// Bicubic filter kernel function
fn bicubic_kernel(x: f64) -> f64 {
    cubic_hermite(x.abs())
}

/// Resize image using high-quality Lanczos resampling
///
/// # Arguments
///
/// * `src` - Source image
/// * `width` - Target width
/// * `height` - Target height
///
/// # Returns
///
/// * Resized image
pub fn resize_lanczos(src: &DynamicImage, width: u32, height: u32) -> Result<DynamicImage> {
    resize_convolution(src, width, height, lanczos_kernel, 7)
}

/// Resize image using high-quality bicubic resampling
///
/// # Arguments
///
/// * `src` - Source image
/// * `width` - Target width
/// * `height` - Target height
///
/// # Returns
///
/// * Resized image
pub fn resize_bicubic(src: &DynamicImage, width: u32, height: u32) -> Result<DynamicImage> {
    resize_convolution(src, width, height, bicubic_kernel, 5)
}

/// Guided filter for edge-preserving smoothing
///
/// # Arguments
///
/// * `guide` - Guide image (typically the input image itself)
/// * `src` - Source image to be filtered
/// * `radius` - Filter radius
/// * `epsilon` - Regularization parameter
///
/// # Returns
///
/// * Filtered image
fn guided_filter(
    guide: &Array2<f64>,
    src: &Array2<f64>,
    radius: usize,
    epsilon: f64,
) -> Result<Array2<f64>> {
    let (height, width) = guide.dim();

    // These variables will be filled later
    // Mean of guide image (I) in each local window
    let mean_i;
    // Mean of source image (p) in each local window
    let mean_p;
    // Mean of I*I in each local window
    let mean_ii;
    // Mean of I*p in each local window
    let mean_ip;

    // Create box filter kernel
    let box_kernel = Array1::from_elem(2 * radius + 1, 1.0 / ((2 * radius + 1) as f64));

    // Compute local means using box filter
    let mut temp_i = guide.clone();
    let mut temp_p = src.clone();

    // Compute horizontal convolution
    temp_i = convolve_1d(&temp_i, &box_kernel, true)?;
    temp_p = convolve_1d(&temp_p, &box_kernel, true)?;

    // Compute vertical convolution
    mean_i = convolve_1d(&temp_i, &box_kernel, false)?;
    mean_p = convolve_1d(&temp_p, &box_kernel, false)?;

    // Compute mean of I*I and I*p
    let mut ii = Array2::zeros((height, width));
    let mut ip = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            ii[[y, x]] = guide[[y, x]] * guide[[y, x]];
            ip[[y, x]] = guide[[y, x]] * src[[y, x]];
        }
    }

    // Apply box filter to ii and ip
    let mut temp_ii = ii.clone();
    let mut temp_ip = ip.clone();

    // Compute horizontal convolution
    temp_ii = convolve_1d(&temp_ii, &box_kernel, true)?;
    temp_ip = convolve_1d(&temp_ip, &box_kernel, true)?;

    // Compute vertical convolution
    mean_ii = convolve_1d(&temp_ii, &box_kernel, false)?;
    mean_ip = convolve_1d(&temp_ip, &box_kernel, false)?;

    // Compute covariance of (I, p) in each local patch
    let mut cov_ip = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            cov_ip[[y, x]] = mean_ip[[y, x]] - mean_i[[y, x]] * mean_p[[y, x]];
        }
    }

    // Compute variance of I in each local patch
    let mut var_i = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            var_i[[y, x]] = mean_ii[[y, x]] - mean_i[[y, x]] * mean_i[[y, x]];
        }
    }

    // Compute a and b for the linear model
    let mut a = Array2::zeros((height, width));
    let mut b = Array2::zeros((height, width));

    for y in 0..height {
        for x in 0..width {
            a[[y, x]] = cov_ip[[y, x]] / (var_i[[y, x]] + epsilon);
            b[[y, x]] = mean_p[[y, x]] - a[[y, x]] * mean_i[[y, x]];
        }
    }

    // Apply box filter to a and b
    // Compute horizontal convolution
    let temp_a = convolve_1d(&a, &box_kernel, true)?;
    let temp_b = convolve_1d(&b, &box_kernel, true)?;

    // Compute vertical convolution
    let mean_a = convolve_1d(&temp_a, &box_kernel, false)?;
    let mean_b = convolve_1d(&temp_b, &box_kernel, false)?;

    // Compute the output
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            output[[y, x]] = mean_a[[y, x]] * guide[[y, x]] + mean_b[[y, x]];
        }
    }

    Ok(output)
}

/// Resize image using edge-preserving interpolation
///
/// This method first resizes the image using Lanczos interpolation,
/// then applies a guided filter to preserve edges and reduce artifacts.
///
/// # Arguments
///
/// * `src` - Source image
/// * `width` - Target width
/// * `height` - Target height
///
/// # Returns
///
/// * Edge-preserving resized image
pub fn resize_edge_preserving(src: &DynamicImage, width: u32, height: u32) -> Result<DynamicImage> {
    // First resize using high-quality Lanczos
    let initial_resize = resize_lanczos(src, width, height)?;

    // Set guided filter parameters
    let radius = 2;
    let epsilon = 0.01;

    // Convert image to grayscale for guide
    let grayscale = initial_resize.to_luma8();

    // Create output image
    let mut dst: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    // Process each channel with guided filter
    for c in 0..3 {
        // Skip alpha channel
        // Extract channel from initial resize
        let mut channel = Array2::zeros((height as usize, width as usize));
        let mut guide = Array2::zeros((height as usize, width as usize));

        for y in 0..height {
            for x in 0..width {
                let pixel = initial_resize.get_pixel(x, y).to_rgba();
                channel[[y as usize, x as usize]] = pixel[c] as f64;

                // Use grayscale as guide
                let guide_value = grayscale.get_pixel(x, y)[0];
                guide[[y as usize, x as usize]] = guide_value as f64;
            }
        }

        // Apply guided filter
        let filtered = guided_filter(&guide, &channel, radius, epsilon)?;

        // Update output image
        for y in 0..height {
            for x in 0..width {
                let mut pixel = dst.get_pixel_mut(x, y).to_rgba();
                pixel[c] = filtered[[y as usize, x as usize]].round().clamp(0.0, 255.0) as u8;
                dst.put_pixel(x, y, pixel);
            }
        }
    }

    // Copy alpha channel directly
    for y in 0..height {
        for x in 0..width {
            let alpha = initial_resize.get_pixel(x, y).to_rgba()[3];
            let mut pixel = dst.get_pixel_mut(x, y).to_rgba();
            pixel[3] = alpha;
            dst.put_pixel(x, y, pixel);
        }
    }

    Ok(DynamicImage::ImageRgba8(dst))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, ImageBuffer, Rgba};

    #[test]
    fn test_lanczos_kernel() {
        // Lanczos kernel should be 1.0 at x=0
        assert!((lanczos_kernel(0.0) - 1.0).abs() < 1e-10);

        // Lanczos kernel should be 0.0 at x=3 (for a=3)
        assert!(lanczos_kernel(3.0).abs() < 1e-10);

        // Lanczos kernel should be symmetric
        for x in [0.5, 1.0, 1.5, 2.0, 2.5] {
            assert!((lanczos_kernel(x) - lanczos_kernel(-x)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_bicubic_kernel() {
        // Bicubic kernel should be 1.0 at x=0
        assert!(bicubic_kernel(0.0) - 1.0 < 1e-10);

        // Bicubic kernel should be 0.0 at x=2 or beyond
        assert!(bicubic_kernel(2.0).abs() < 1e-10);
        assert!(bicubic_kernel(3.0).abs() < 1e-10);

        // Bicubic kernel should be symmetric
        for x in [0.5, 1.0, 1.5, 1.9] {
            assert!((bicubic_kernel(x) - bicubic_kernel(-x)).abs() < 1e-10);
        }
    }

    #[test]
    fn test_guided_filter() {
        // Create a simple test input
        let width = 10;
        let height = 10;
        let mut guide = Array2::zeros((height, width));
        let mut src = Array2::zeros((height, width));

        // Setup test data with an edge
        for y in 0..height {
            for x in 0..width {
                // Create a vertical edge in the middle
                if x < width / 2 {
                    guide[[y, x]] = 50.0;
                    src[[y, x]] = 50.0;
                } else {
                    guide[[y, x]] = 150.0;
                    src[[y, x]] = 150.0;
                }

                // Add some noise to the source
                if x % 2 == 0 && y % 2 == 0 {
                    src[[y, x]] += 20.0;
                }
            }
        }

        // Apply guided filter
        let radius = 1;
        let epsilon = 0.1;
        let result = guided_filter(&guide, &src, radius, epsilon).unwrap();

        // The filter should denoise but preserve the edge
        // Check that the edge is preserved
        let edge_preserved = result[[5, 4]] < 100.0 && result[[5, 5]] > 100.0;
        assert!(edge_preserved);

        // Check that noise is reduced
        let noise_reduced = (result[[2, 2]] - result[[2, 0]]).abs() < 10.0;
        assert!(noise_reduced);
    }

    #[test]
    fn test_resize_edge_preserving() {
        // Create a simple test image
        let width = 20;
        let height = 20;
        let mut img = ImageBuffer::new(width, height);

        // Create a pattern with edges
        for y in 0..height {
            for x in 0..width {
                let pixel_value = if x < width / 2 { 50 } else { 200 };
                img.put_pixel(x, y, Rgba([pixel_value, pixel_value, pixel_value, 255]));
            }
        }

        let src = DynamicImage::ImageRgba8(img);

        // Resize to smaller dimensions
        let result = resize_edge_preserving(&src, 10, 10).unwrap();

        // The edge should be preserved after resizing
        let edge_before = result.get_pixel(4, 5)[0];
        let edge_after = result.get_pixel(5, 5)[0];
        assert!(edge_after - edge_before > 50);

        // Also test upscaling
        let result_up = resize_edge_preserving(&src, 30, 30).unwrap();

        // Dimensions should be correct
        assert_eq!(result_up.width(), 30);
        assert_eq!(result_up.height(), 30);

        // The edge should be preserved after upscaling
        let edge_before_up = result_up.get_pixel(14, 15)[0];
        let edge_after_up = result_up.get_pixel(15, 15)[0];
        assert!(edge_after_up - edge_before_up > 50);
    }
}
