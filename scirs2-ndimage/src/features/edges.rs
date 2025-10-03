//! Edge detection algorithms
//!
//! This module provides functions for detecting edges in n-dimensional arrays,
//! including gradient-based methods, zero-crossing methods, and other edge detection techniques.

use crate::error::{NdimageError, NdimageResult};
use crate::filters::{
    convolve, gaussian_filter_f32, gradient_magnitude, prewitt, scharr, sobel, BorderMode,
};
use scirs2_core::ndarray::{Array, Array2, ArrayD, Ix2};
use std::f32::consts::PI;

/// Gradient calculation method for edge detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GradientMethod {
    /// Sobel operator (default)
    Sobel,
    /// Prewitt operator
    Prewitt,
    /// Scharr operator (better rotational symmetry)
    Scharr,
}

/// Edge detection algorithm options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDetectionAlgorithm {
    /// Canny edge detector (multi-stage algorithm)
    Canny,
    /// Laplacian of Gaussian (zero-crossing method)
    LoG,
    /// Simple gradient-based detection
    Gradient,
}

/// Configuration for edge detection algorithms
#[derive(Debug, Clone)]
pub struct EdgeDetectionConfig {
    /// Algorithm to use for edge detection
    pub algorithm: EdgeDetectionAlgorithm,
    /// Method to calculate gradients
    pub gradient_method: GradientMethod,
    /// Standard deviation for Gaussian blur applied before edge detection
    pub sigma: f32,
    /// Low threshold for hysteresis (for Canny edge detection)
    pub low_threshold: f32,
    /// High threshold for hysteresis (for Canny edge detection)
    pub high_threshold: f32,
    /// Border handling mode
    pub border_mode: BorderMode,
    /// Whether to return the edge magnitude (f32) or binary edges (bool)
    pub return_magnitude: bool,
}

impl Default for EdgeDetectionConfig {
    fn default() -> Self {
        Self {
            algorithm: EdgeDetectionAlgorithm::Canny,
            gradient_method: GradientMethod::Sobel,
            sigma: 1.0,
            low_threshold: 0.1,
            high_threshold: 0.2,
            border_mode: BorderMode::Reflect,
            return_magnitude: false,
        }
    }
}

/// Unified edge detector function that works with different algorithms
///
/// This function provides a common interface for various edge detection methods,
/// allowing users to choose the algorithm and configure parameters through a single function.
///
/// # Arguments
///
/// * `image` - Input image as a 2D array
/// * `config` - Configuration for the edge detection algorithm
///
/// # Returns
///
/// * If `config.return_magnitude` is `false` (default), returns a boolean array where `true` indicates edge pixels
/// * If `config.return_magnitude` is `true`, returns the original edge magnitude as floating-point values
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::features::{edge_detector, EdgeDetectionConfig, EdgeDetectionAlgorithm, GradientMethod};
/// use scirs2_ndimage::filters::BorderMode;
///
/// // Use a larger test image to avoid overflow issues in doctests
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// // Default settings - Canny edge detection with Sobel gradient
/// let edges = edge_detector(&image, EdgeDetectionConfig::default()).unwrap();
///
/// // Custom configuration - Gradient edge detection with Scharr operator and custom thresholds
/// let custom_config = EdgeDetectionConfig {
///     algorithm: EdgeDetectionAlgorithm::Gradient,
///     gradient_method: GradientMethod::Scharr,
///     sigma: 1.5,
///     low_threshold: 0.05,
///     high_threshold: 0.15,
///     border_mode: BorderMode::Reflect,
///     return_magnitude: true,
/// };
/// let edge_magnitudes = edge_detector(&image, custom_config).unwrap();
/// ```
#[allow(dead_code)]
pub fn edge_detector(
    image: &Array<f32, Ix2>,
    config: EdgeDetectionConfig,
) -> NdimageResult<Array<f32, Ix2>> {
    match config.algorithm {
        EdgeDetectionAlgorithm::Canny => {
            // Already returns f32 values, so we just return it as is
            Ok(canny_impl(
                image,
                config.sigma,
                config.low_threshold,
                config.high_threshold,
                config.gradient_method,
                config.border_mode,
            ))
        }
        EdgeDetectionAlgorithm::LoG => {
            let edges = laplacian_edges_impl(
                image,
                config.sigma,
                config.low_threshold,
                config.border_mode,
            )?;

            if !config.return_magnitude {
                // Threshold to binary edges
                Ok(edges
                    .mapv(|v| v.abs() > config.low_threshold)
                    .mapv(|v| if v { 1.0 } else { 0.0 }))
            } else {
                Ok(edges)
            }
        }
        EdgeDetectionAlgorithm::Gradient => {
            let edges = gradient_edges_impl(
                image,
                config.gradient_method,
                config.sigma,
                config.border_mode,
            )?;

            if !config.return_magnitude {
                // Threshold to binary edges
                Ok(edges
                    .mapv(|v| v > config.low_threshold)
                    .mapv(|v| if v { 1.0 } else { 0.0 }))
            } else {
                Ok(edges)
            }
        }
    }
}

/// Canny edge detector
///
/// The Canny edge detector is an edge detection operator that uses a multi-stage algorithm
/// to detect a wide range of edges in images. It was developed by John F. Canny in 1986.
///
/// # Arguments
///
/// * `image` - Input array, 2D array
/// * `sigma` - Standard deviation of the Gaussian filter
/// * `low_threshold` - Lower threshold for hysteresis
/// * `high_threshold` - Upper threshold for hysteresis
/// * `method` - Gradient calculation method (defaults to Sobel if None)
///
/// # Returns
///
/// * Array of edges, where `true` values indicate detected edges
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::features::{canny, GradientMethod};
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// // Using default Sobel method
/// let edges = canny(&image, 1.0, 0.1, 0.2, None);
///
/// // Using Scharr method for better edge detection
/// let edges_scharr = canny(&image, 1.0, 0.1, 0.2, Some(GradientMethod::Scharr));
/// ```
#[allow(dead_code)]
pub fn canny(
    image: &Array<f32, Ix2>,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
    method: Option<GradientMethod>,
) -> Array<f32, Ix2> {
    let method = method.unwrap_or(GradientMethod::Sobel);
    let border_mode = BorderMode::Reflect;

    canny_impl(
        image,
        sigma,
        low_threshold,
        high_threshold,
        method,
        border_mode,
    )
}

// Internal implementation of Canny edge detection with enhanced performance
#[allow(dead_code)]
fn canny_impl(
    image: &Array<f32, Ix2>,
    sigma: f32,
    low_threshold: f32,
    high_threshold: f32,
    method: GradientMethod,
    mode: BorderMode,
) -> Array<f32, Ix2> {
    let image_dim = image.raw_dim();

    // Step 1: Gaussian filter to reduce noise
    let image_d = image.clone().into_dyn();
    let smoothed =
        gaussian_filter_f32(&image_d, sigma, Some(mode), None).unwrap_or_else(|_| image_d.clone());

    // Step 2: Calculate gradients using the specified method
    let gradients = calculate_gradient(&smoothed, method, mode);
    let gradient_x = &gradients.0;
    let gradient_y = &gradients.1;

    // Step 3: Calculate gradient magnitude and direction
    let (magnitude, direction) =
        calculate_magnitude_and_direction(gradient_x, gradient_y, image_dim);

    // Step 4: Non-maximum suppression
    let suppressed = non_maximum_suppression(&magnitude, &direction);

    // Step 5: Double thresholding and edge tracking by hysteresis
    hysteresis_thresholding(&suppressed, low_threshold, high_threshold)
}

// Calculate gradients using the specified method
#[allow(dead_code)]
fn calculate_gradient(
    image: &ArrayD<f32>,
    method: GradientMethod,
    mode: BorderMode,
) -> (ArrayD<f32>, ArrayD<f32>) {
    match method {
        GradientMethod::Sobel => {
            let gy = sobel(image, 0, Some(mode)).unwrap_or_else(|_| ArrayD::zeros(image.raw_dim()));
            let gx = sobel(image, 1, Some(mode)).unwrap_or_else(|_| ArrayD::zeros(image.raw_dim()));
            (gx, gy)
        }
        GradientMethod::Prewitt => {
            let gy =
                prewitt(image, 0, Some(mode)).unwrap_or_else(|_| ArrayD::zeros(image.raw_dim()));
            let gx =
                prewitt(image, 1, Some(mode)).unwrap_or_else(|_| ArrayD::zeros(image.raw_dim()));
            (gx, gy)
        }
        GradientMethod::Scharr => {
            let gy =
                scharr(image, 0, Some(mode)).unwrap_or_else(|_| ArrayD::zeros(image.raw_dim()));
            let gx =
                scharr(image, 1, Some(mode)).unwrap_or_else(|_| ArrayD::zeros(image.raw_dim()));
            (gx, gy)
        }
    }
}

// Calculate magnitude and direction from gradient components
#[allow(dead_code)]
fn calculate_magnitude_and_direction(
    gradient_x: &ArrayD<f32>,
    gradient_y: &ArrayD<f32>,
    shape: Ix2,
) -> (Array<f32, Ix2>, Array<f32, Ix2>) {
    let magnitude = Array::<f32, Ix2>::zeros(shape);
    let mut direction = Array::<f32, Ix2>::zeros(shape);

    // Create a copy to avoid mutable borrow conflict
    let mut mag_copy = Array::<f32, Ix2>::zeros(shape);

    // Calculate gradient magnitude and direction
    for (pos, _) in magnitude.indexed_iter() {
        let idx_d = [pos.0, pos.1];
        let gx = gradient_x[idx_d.as_ref()];
        let gy = gradient_y[idx_d.as_ref()];

        // Calculate magnitude using Euclidean distance
        mag_copy[pos] = (gx * gx + gy * gy).sqrt();

        // Calculate direction in degrees and convert to one of four directions (0, 45, 90, 135)
        let angle = gy.atan2(gx) * 180.0 / PI;
        let angle = if angle < 0.0 { angle + 180.0 } else { angle };

        // Quantize the angle to 4 directions (0, 45, 90, 135 degrees)
        direction[pos] = if !(22.5..157.5).contains(&angle) {
            0.0 // 0 degrees (horizontal)
        } else if (22.5..67.5).contains(&angle) {
            45.0 // 45 degrees
        } else if (67.5..112.5).contains(&angle) {
            90.0 // 90 degrees (vertical)
        } else {
            135.0 // 135 degrees
        };
    }

    (mag_copy, direction)
}

// Non-maximum suppression to thin edges
#[allow(dead_code)]
fn non_maximum_suppression(
    magnitude: &Array<f32, Ix2>,
    direction: &Array<f32, Ix2>,
) -> Array<f32, Ix2> {
    let shape = magnitude.dim();
    let mut suppressed = Array::zeros(shape);

    // Skip the border pixels to avoid bounds checking
    for row in 1..(shape.0 - 1) {
        for col in 1..(shape.1 - 1) {
            let dir = direction[(row, col)];
            let mag = magnitude[(row, col)];

            // If the magnitude is zero, skip further processing
            if mag == 0.0 {
                continue;
            }

            let (neighbor1, neighbor2) = get_gradient_neighbors(row, col, dir, magnitude);

            // If the current pixel is a local maximum, keep it, otherwise suppress it
            if mag >= neighbor1 && mag >= neighbor2 {
                suppressed[(row, col)] = mag;
            }
        }
    }

    suppressed
}

// Hysteresis thresholding to connect edges
#[allow(dead_code)]
fn hysteresis_thresholding(
    suppressed: &Array<f32, Ix2>,
    low_threshold: f32,
    high_threshold: f32,
) -> Array<f32, Ix2> {
    let shape = suppressed.dim();
    let mut result = Array::from_elem(shape, 0.0);
    let mut candidates = Vec::new();

    // First pass: identify strong edges and potential candidates
    for ((row, col), &val) in suppressed.indexed_iter() {
        if val >= high_threshold {
            // Mark as strong edge
            result[(row, col)] = 1.0;

            // Add strong edges to the candidates list to check their neighbors
            candidates.push((row, col));
        } else if val >= low_threshold {
            // Potential edge candidate - will be processed in the next phase
            candidates.push((row, col));
        }
    }

    // Second pass: Process candidates using a queue-based approach for efficiency
    let mut processed = Array::from_elem(shape, false);
    for (row, col) in candidates.iter() {
        if result[(*row, *col)] > 0.0 || processed[(*row, *col)] {
            // Skip already processed pixels or confirmed edges
            continue;
        }

        processed[(*row, *col)] = true;

        // If this candidate is connected to a strong edge, mark it
        if is_connected_to_strong_edge(*row, *col, &result) {
            result[(*row, *col)] = 1.0;

            // Add neighbors to the queue for further processing
            let mut queue = Vec::with_capacity(8); // Pre-allocate for efficiency
            queue.push((*row, *col));

            while let Some((r, c)) = queue.pop() {
                // Check all 8 neighbors
                for nr in (r.saturating_sub(1))..=(r + 1).min(shape.0 - 1) {
                    for nc in (c.saturating_sub(1))..=(c + 1).min(shape.1 - 1) {
                        if processed[(nr, nc)] || result[(nr, nc)] > 0.0 {
                            continue;
                        }

                        // If this is a candidate edge, mark it and add to queue
                        if suppressed[(nr, nc)] >= low_threshold {
                            result[(nr, nc)] = 1.0;
                            processed[(nr, nc)] = true;
                            queue.push((nr, nc));
                        }
                    }
                }
            }
        }
    }

    result
}

/// Helper function to get the neighbors in the gradient direction
#[allow(dead_code)]
fn get_gradient_neighbors(
    row: usize,
    col: usize,
    direction: f32,
    magnitude: &Array<f32, Ix2>,
) -> (f32, f32) {
    // 0 degrees (horizontal)
    if direction == 0.0 {
        (magnitude[(row, col - 1)], magnitude[(row, col + 1)])
    }
    // 45 degrees
    else if direction == 45.0 {
        (magnitude[(row - 1, col + 1)], magnitude[(row + 1, col - 1)])
    }
    // 90 degrees (vertical)
    else if direction == 90.0 {
        (magnitude[(row - 1, col)], magnitude[(row + 1, col)])
    }
    // 135 degrees
    else {
        (magnitude[(row - 1, col - 1)], magnitude[(row + 1, col + 1)])
    }
}

/// Helper function to check if a pixel is connected to a strong edge
#[allow(dead_code)]
fn is_connected_to_strong_edge(row: usize, col: usize, edges: &Array<f32, Ix2>) -> bool {
    let shape = edges.dim();

    for i in (row.saturating_sub(1))..=(row + 1).min(shape.0 - 1) {
        for j in (col.saturating_sub(1))..=(col + 1).min(shape.1 - 1) {
            if !(i == row && j == col) && edges[(i, j)] > 0.0 {
                return true;
            }
        }
    }
    false
}

/// Laplacian of Gaussian (LoG) edge detector
///
/// The Laplacian of Gaussian operator computes the second derivative of an image.
/// It highlights regions of rapid intensity change and is often used for edge detection.
/// This implementation first applies Gaussian smoothing, then computes the Laplacian.
///
/// # Arguments
///
/// * `image` - Input array
/// * `sigma` - Standard deviation of the Gaussian filter
/// * `threshold` - Optional threshold for zero-crossing detection (if None, no thresholding is applied)
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * Result containing the LoG filtered image
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::features::laplacian_edges;
/// use scirs2_ndimage::filters::BorderMode;
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// // Apply LoG filter with default settings
/// let edges = laplacian_edges(&image, 1.0, None, None).unwrap();
///
/// // Apply LoG filter with thresholding
/// let edges_threshold = laplacian_edges(&image, 1.0, Some(0.1), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn laplacian_edges(
    image: &Array<f32, Ix2>,
    sigma: f32,
    threshold: Option<f32>,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<f32, Ix2>> {
    let mode = mode.unwrap_or(BorderMode::Reflect);
    Ok(laplacian_edges_impl(
        image,
        sigma,
        threshold.unwrap_or(0.0),
        mode,
    )?)
}

// Internal implementation of Laplacian of Gaussian edge detection
#[allow(dead_code)]
fn laplacian_edges_impl(
    image: &Array<f32, Ix2>,
    sigma: f32,
    threshold: f32,
    mode: BorderMode,
) -> NdimageResult<Array<f32, Ix2>> {
    // Convert to dynamic array for processing with our filter functions
    let image_d = image.clone().into_dyn();

    // First, apply Gaussian filter to reduce noise
    let smoothed =
        gaussian_filter_f32(&image_d, sigma, Some(mode), None).unwrap_or_else(|_| image_d.clone());

    // Create Laplace kernel size based on sigma (typically 2*ceil(3*sigma) + 1)
    let ksize = ((2.0 * (3.0 * sigma).ceil() + 1.0).max(3.0)) as usize;

    // Create Laplace kernel
    let mut kernel = ArrayD::zeros(vec![ksize; image.ndim()]);
    let center = ksize / 2;

    // Fill kernel with the discrete Laplacian operator
    // For 2D: [0 1 0; 1 -4 1; 0 1 0]
    let center_value = -2.0 * image.ndim() as f32;

    // Set center to the sum of all other values (negative)
    let center_idx = vec![center; image.ndim()];
    kernel[center_idx.as_slice()] = center_value;

    // Set direct neighbors to 1
    for dim in 0..image.ndim() {
        let mut idx = vec![center; image.ndim()];

        // Set previous neighbor
        if center > 0 {
            idx[dim] = center - 1;
            kernel[idx.as_slice()] = 1.0;
        }

        // Set next neighbor
        idx[dim] = center + 1;
        if idx[dim] < ksize {
            kernel[idx.as_slice()] = 1.0;
        }
    }

    // Apply convolution
    let laplacian = convolve(&smoothed, &kernel, Some(mode)).map_err(|e| {
        NdimageError::ComputationError(format!("Laplacian convolution failed: {}", e))
    })?;

    // Convert back to 2D array
    let mut result_copy = Array::zeros(image.dim());
    for i in 0..image.dim().0 {
        for j in 0..image.dim().1 {
            result_copy[(i, j)] = laplacian[[i, j]];
        }
    }

    // Apply thresholding if requested
    if threshold > 0.0 {
        return Ok(result_copy.mapv(|v| if v.abs() > threshold { v } else { 0.0 }));
    }

    Ok(result_copy)
}

/// Gradient-based edge detection
///
/// Detects edges using the gradient magnitude calculated with the specified method.
/// This is a simpler alternative to Canny edge detection when you don't need the
/// same level of edge quality or performance.
///
/// # Arguments
///
/// * `image` - Input image
/// * `method` - Gradient calculation method (defaults to Sobel if None)
/// * `sigma` - Standard deviation of Gaussian filter applied before edge detection (if None, no smoothing is applied)
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * Edge magnitude as an array of floats
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::features::{gradient_edges, GradientMethod};
/// use scirs2_ndimage::filters::BorderMode;
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// // Using default Sobel method without smoothing
/// let edges = gradient_edges(&image, None, None, None).unwrap();
///
/// // Using Scharr method with Gaussian smoothing
/// let edges_scharr = gradient_edges(&image, Some(GradientMethod::Scharr), Some(1.0), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn gradient_edges(
    image: &Array<f32, Ix2>,
    method: Option<GradientMethod>,
    sigma: Option<f32>,
    mode: Option<BorderMode>,
) -> NdimageResult<Array<f32, Ix2>> {
    let method = method.unwrap_or(GradientMethod::Sobel);
    let mode = mode.unwrap_or(BorderMode::Reflect);

    Ok(gradient_edges_impl(
        image,
        method,
        sigma.unwrap_or(0.0),
        mode,
    )?)
}

// Internal implementation of gradient-based edge detection
#[allow(dead_code)]
fn gradient_edges_impl(
    image: &Array<f32, Ix2>,
    method: GradientMethod,
    sigma: f32,
    mode: BorderMode,
) -> NdimageResult<Array<f32, Ix2>> {
    let image_d = image.clone().into_dyn();

    // Apply Gaussian smoothing if sigma > 0
    let processed = if sigma > 0.0 {
        gaussian_filter_f32(&image_d, sigma, Some(mode), None).map_err(|e| {
            NdimageError::ComputationError(format!("Gaussian smoothing failed: {}", e))
        })?
    } else {
        image_d
    };

    // Map the gradient method to the corresponding string parameter for gradient_magnitude
    let method_str = match method {
        GradientMethod::Sobel => "sobel",
        GradientMethod::Prewitt => "prewitt",
        GradientMethod::Scharr => "scharr",
    };

    // Calculate gradient magnitude
    let magnitude = gradient_magnitude(&processed, Some(mode), Some(method_str)).map_err(|e| {
        NdimageError::ComputationError(format!("Gradient magnitude calculation failed: {}", e))
    })?;

    // Convert back to 2D array
    let mut result_copy = Array::zeros(image.dim());

    for i in 0..image.dim().0 {
        for j in 0..image.dim().1 {
            result_copy[(i, j)] = magnitude[[i, j]];
        }
    }

    Ok(result_copy)
}

/// Sobel edge detector (for backward compatibility)
///
/// A simplified wrapper around the sobel function that returns the magnitude of edges.
///
/// # Arguments
///
/// * `image` - Input array
///
/// # Returns
///
/// * Result containing the magnitude of edges
#[allow(dead_code)]
pub fn sobel_edges(image: &ArrayD<f32>) -> NdimageResult<ArrayD<f32>> {
    edge_detector_simple(image, Some(GradientMethod::Sobel), None)
}

/// Enhanced edge detector (compatible with previous API)
///
/// A simplified wrapper around the gradient_magnitude function that returns the magnitude of edges.
/// This function allows selecting different gradient methods for more flexibility.
///
/// # Arguments
///
/// * `image` - Input array
/// * `method` - Gradient calculation method (defaults to Sobel if None)
/// * `mode` - Border handling mode (defaults to Reflect)
///
/// # Returns
///
/// * Result containing the magnitude of edges
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::features::{edge_detector_simple, GradientMethod};
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
/// ].into_dyn();
///
/// // Using default Sobel method
/// let edges = edge_detector_simple(&image, None, None).unwrap();
///
/// // Using Scharr method for better rotational invariance
/// let edges_scharr = edge_detector_simple(&image, Some(GradientMethod::Scharr), None).unwrap();
/// ```
#[allow(dead_code)]
pub fn edge_detector_simple(
    image: &ArrayD<f32>,
    method: Option<GradientMethod>,
    mode: Option<BorderMode>,
) -> NdimageResult<ArrayD<f32>> {
    let border_mode = mode.unwrap_or(BorderMode::Reflect);

    // Map the gradient method to the corresponding string parameter for gradient_magnitude
    let method_str = match method.unwrap_or(GradientMethod::Sobel) {
        GradientMethod::Sobel => "sobel",
        GradientMethod::Prewitt => "prewitt",
        GradientMethod::Scharr => "scharr",
    };

    gradient_magnitude(image, Some(border_mode), Some(method_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_canny_edge_detector() {
        // Create a simple test image with a clear edge
        let image = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ];

        // Apply Canny edge detection with default Sobel method
        let edges_sobel = canny(&image, 1.0, 0.1, 0.2, None);

        // Check that edges are detected
        assert!(
            edges_sobel.fold(false, |acc, &x| acc || x > 0.0),
            "No edges detected with Sobel"
        );

        // Apply Canny with Scharr method
        let edges_scharr = canny(&image, 1.0, 0.1, 0.2, Some(GradientMethod::Scharr));

        // Check that edges are detected
        assert!(
            edges_scharr.fold(false, |acc, &x| acc || x > 0.0),
            "No edges detected with Scharr"
        );
    }

    #[test]
    fn test_unified_edge_detector() {
        // Create a simple test image with a clear edge
        let image = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ];

        // Test with default config (Canny)
        let edges_default = edge_detector(&image, EdgeDetectionConfig::default())
            .expect("edge_detector should succeed for test with default config");
        assert!(
            edges_default.fold(false, |acc, &x| acc || (x > 0.0)),
            "Edges should be detected with default config"
        );

        // Test with gradient edge detection
        let gradient_config = EdgeDetectionConfig {
            algorithm: EdgeDetectionAlgorithm::Gradient,
            gradient_method: GradientMethod::Sobel,
            sigma: 1.0,
            low_threshold: 0.1,
            high_threshold: 0.2,
            border_mode: BorderMode::Reflect,
            return_magnitude: true,
        };

        let gradient_edges = edge_detector(&image, gradient_config)
            .expect("edge_detector should succeed for test with gradient config");

        // Check that magnitudes are returned
        let max_magnitude = gradient_edges.fold(0.0, |acc, &x| if x > acc { x } else { acc });
        assert!(
            max_magnitude > 0.1,
            "Gradient magnitudes should be above threshold"
        );

        // Test with Laplacian of Gaussian
        let log_config = EdgeDetectionConfig {
            algorithm: EdgeDetectionAlgorithm::LoG,
            sigma: 1.0,
            low_threshold: 0.05,
            ..EdgeDetectionConfig::default()
        };

        let log_edges = edge_detector(&image, log_config)
            .expect("edge_detector should succeed for test with LoG config");

        // Check that edges are detected
        assert!(
            log_edges.fold(false, |acc, &x| acc || (x > 0.0)),
            "LoG should detect edges"
        );
    }

    #[test]
    fn test_gradient_edges() {
        // Create a simple test image with a clear edge
        let image = array![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
        ];

        // Test different gradient methods
        let edges_sobel = gradient_edges(&image, Some(GradientMethod::Sobel), Some(1.0), None)
            .expect("gradient_edges with Sobel should succeed for test");
        let edges_prewitt = gradient_edges(&image, Some(GradientMethod::Prewitt), Some(1.0), None)
            .expect("gradient_edges with Prewitt should succeed for test");
        let edges_scharr = gradient_edges(&image, Some(GradientMethod::Scharr), Some(1.0), None)
            .expect("gradient_edges with Scharr should succeed for test");

        // Check that all methods detect edges
        assert!(
            edges_sobel.iter().any(|&x| x > 0.1),
            "Sobel should detect edges"
        );
        assert!(
            edges_prewitt.iter().any(|&x| x > 0.1),
            "Prewitt should detect edges"
        );
        assert!(
            edges_scharr.iter().any(|&x| x > 0.1),
            "Scharr should detect edges"
        );

        // Scharr should detect better gradient responses for diagonal edges
        let diagonalimage = array![
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        let diag_sobel = gradient_edges(&diagonalimage, Some(GradientMethod::Sobel), None, None)
            .expect("gradient_edges with Sobel should succeed for diagonal test");
        let diag_scharr = gradient_edges(&diagonalimage, Some(GradientMethod::Scharr), None, None)
            .expect("gradient_edges with Scharr should succeed for diagonal test");

        // Calculate the maximum magnitude for each
        let max_sobel = diag_sobel
            .iter()
            .fold(0.0, |acc, &x| if x > acc { x } else { acc });
        let max_scharr = diag_scharr
            .iter()
            .fold(0.0, |acc, &x| if x > acc { x } else { acc });

        // Scharr should have higher response for diagonal edges
        assert!(
            max_scharr > max_sobel,
            "Scharr should have stronger diagonal response"
        );
    }

    #[test]
    fn test_laplacian_edges() {
        // Create a simple test image with a point
        let mut image = Array2::<f32>::zeros((5, 5));
        image[[2, 2]] = 1.0;

        // Apply LoG filter
        let edges = laplacian_edges(&image, 1.0, None, None)
            .expect("laplacian_edges should succeed for test");

        // Check that edges are detected
        assert!(edges.iter().any(|&x| x != 0.0), "LoG should detect edges");

        // Center should have negative value (for a bright point)
        assert!(
            edges[[2, 2]] < 0.0,
            "Center of point should have negative LoG value"
        );

        // Apply with thresholding
        let edges_threshold = laplacian_edges(&image, 1.0, Some(0.1), None)
            .expect("laplacian_edges with threshold should succeed for test");

        // Count non-zero values before and after thresholding
        let count_before = edges.iter().filter(|&&x| x != 0.0).count();
        let count_after = edges_threshold.iter().filter(|&&x| x != 0.0).count();

        // Thresholding should reduce the number of non-zero values
        assert!(
            count_after <= count_before,
            "Thresholding should reduce edge points"
        );
    }
}
