use scirs2_core::ndarray::s;
// Median Filtering module
//
// This module implements median filtering techniques for signal and image processing.
// Median filtering is particularly effective at removing salt-and-pepper and
// impulse noise while preserving edges.
//
// The implementation includes:
// - 1D Median filtering for signals
// - 2D Median filtering for images
// - Weighted median filtering
// - Adaptive median filtering
// - Edge-preserving median filtering variants
//
// # Example
// ```
// use scirs2_core::ndarray::Array1;
// use scirs2signal::median::{median_filter_1d, MedianConfig};
//
// // Create a test signal with impulse noise
// let mut signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 5.0, 1.3, 1.2, 0.0, 1.1]);
//
// // Apply median filter with window size 3
// let config = MedianConfig::default();
// let filtered = median_filter_1d(&signal, 3, &config).unwrap();
// // The outliers (5.0 and 0.0) will be replaced with median values
// ```

use crate::error::{SignalError, SignalResult};
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};

#[allow(unused_imports)]
/// Edge handling mode for median filtering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeMode {
    /// Reflect the signal at boundaries
    Reflect,

    /// Pad with the nearest valid value
    Nearest,

    /// Pad with zeros
    Constant(f64),

    /// Wrap around (circular padding)
    Wrap,
}

/// Configuration for median filtering
#[derive(Debug, Clone)]
pub struct MedianConfig {
    /// Edge handling mode
    pub edge_mode: EdgeMode,

    /// Whether to use adaptive kernel size
    pub adaptive: bool,

    /// Maximum kernel size for adaptive filtering
    pub max_kernel_size: usize,

    /// Noise threshold for adaptive filtering
    pub noise_threshold: f64,

    /// Whether to apply center weighted median filtering
    pub center_weighted: bool,

    /// Center weight factor (higher values give more weight to the center pixel)
    pub center_weight: usize,
}

impl Default for MedianConfig {
    fn default() -> Self {
        Self {
            edge_mode: EdgeMode::Reflect,
            adaptive: false,
            max_kernel_size: 9,
            noise_threshold: 50.0,
            center_weighted: false,
            center_weight: 3,
        }
    }
}

/// Applies median filtering to a 1D signal.
///
/// Median filtering replaces each value with the median of neighboring values,
/// which is effective at removing outliers and impulse noise.
///
/// # Arguments
/// * `signal` - Input signal
/// * `kernel_size` - Size of the filtering window (must be odd)
/// * `config` - Filtering configuration
///
/// # Returns
/// * The filtered signal
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::median::{median_filter_1d, MedianConfig};
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 5.0, 1.1, 1.3, 0.0, 1.2]);
/// let config = MedianConfig::default();
/// let filtered = median_filter_1d(&signal, 3, &config).unwrap();
/// ```
#[allow(dead_code)]
pub fn median_filter_1d(
    signal: &Array1<f64>,
    kernel_size: usize,
    config: &MedianConfig,
) -> SignalResult<Array1<f64>> {
    // Validate kernel _size
    if kernel_size % 2 != 1 {
        return Err(SignalError::ValueError(
            "Kernel _size must be odd".to_string(),
        ));
    }

    let n = signal.len();

    // If signal is too short, return a copy
    if n <= 1 || kernel_size > n {
        return Ok(signal.clone());
    }

    let half_kernel = kernel_size / 2;

    // Create padded signal based on edge mode
    let paddedsignal = padsignal_1d(signal, half_kernel, config.edge_mode);

    // Apply either standard or adaptive median filtering
    if config.adaptive {
        adaptive_median_filter_1d(signal, &paddedsignal, half_kernel, config)
    } else if config.center_weighted {
        center_weighted_median_filter_1d(signal, &paddedsignal, half_kernel, config)
    } else {
        standard_median_filter_1d(signal, &paddedsignal, half_kernel)
    }
}

/// Applies standard median filtering to a 1D signal
#[allow(dead_code)]
fn standard_median_filter_1d(
    signal: &Array1<f64>,
    paddedsignal: &Array1<f64>,
    half_kernel: usize,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let mut filtered = Array1::zeros(n);

    // Process each point in the signal
    for i in 0..n {
        // Extract window around current point
        let window_start = i;
        let window_end = i + 2 * half_kernel + 1;

        // Ensure window is within bounds
        if window_start >= paddedsignal.len() || window_end > paddedsignal.len() {
            return Err(SignalError::DimensionMismatch(
                "Window extends beyond padded signal bounds".to_string(),
            ));
        }

        // Extract and sort window values
        let mut window: Vec<f64> = paddedsignal.slice(s![window_start..window_end]).to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Set output to median value
        filtered[i] = window[half_kernel];
    }

    Ok(filtered)
}

/// Applies center-weighted median filtering to a 1D signal
#[allow(dead_code)]
fn center_weighted_median_filter_1d(
    signal: &Array1<f64>,
    paddedsignal: &Array1<f64>,
    half_kernel: usize,
    config: &MedianConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let mut filtered = Array1::zeros(n);

    // Process each point in the signal
    for i in 0..n {
        // Extract window around current point
        let window_start = i;
        let window_end = i + 2 * half_kernel + 1;

        // Ensure window is within bounds
        if window_start >= paddedsignal.len() || window_end > paddedsignal.len() {
            return Err(SignalError::DimensionMismatch(
                "Window extends beyond padded signal bounds".to_string(),
            ));
        }

        // Create weighted window by repeating the center value
        let mut weighted_window = Vec::new();

        for j in window_start..window_end {
            let value = paddedsignal[j];

            // Add center value with higher weight
            if j == window_start + half_kernel {
                for _ in 0..config.center_weight {
                    weighted_window.push(value);
                }
            } else {
                weighted_window.push(value);
            }
        }

        // Sort the weighted window
        weighted_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate the median position (considering the weighted values)
        let median_idx = weighted_window.len() / 2;

        // Set output to weighted median value
        filtered[i] = weighted_window[median_idx];
    }

    Ok(filtered)
}

/// Applies adaptive median filtering to a 1D signal
#[allow(dead_code)]
fn adaptive_median_filter_1d(
    signal: &Array1<f64>,
    paddedsignal: &Array1<f64>,
    initial_half_kernel: usize,
    config: &MedianConfig,
) -> SignalResult<Array1<f64>> {
    let n = signal.len();
    let mut filtered = Array1::zeros(n);

    // Maximum half _kernel size
    let max_half_kernel = config.max_kernel_size / 2;

    // Process each point in the signal
    for i in 0..n {
        // Start with the initial _kernel size
        let mut half_kernel = initial_half_kernel;
        let mut window_size = 2 * half_kernel + 1;

        // Extract the current pixel value
        let curr_val = paddedsignal[i + half_kernel];

        // Adaptive window size adjustment
        while half_kernel <= max_half_kernel {
            // Extract window around current point
            let window_start = i + (initial_half_kernel - half_kernel);
            let window_end = window_start + window_size;

            // Ensure window is within bounds
            if window_end > paddedsignal.len() {
                break;
            }

            // Extract and sort window values
            let window: Vec<f64> = paddedsignal.slice(s![window_start..window_end]).to_vec();
            let mut sorted_window = window.clone();
            sorted_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate window statistics
            let median = sorted_window[half_kernel];
            let min_val = sorted_window[0];
            let max_val = sorted_window[sorted_window.len() - 1];

            // Level A: Test if median is impulse
            let level_a = min_val < median && median < max_val;

            if level_a {
                // Level B: Test if current pixel is impulse
                let level_b = min_val < curr_val && curr_val < max_val;

                if level_b {
                    // Not an impulse, keep original value
                    filtered[i] = curr_val;
                } else {
                    // Impulse detected, use median
                    filtered[i] = median;
                }

                // Exit the window size adjustment loop
                break;
            } else {
                // Median might be impulse, increase window size
                half_kernel += 1;
                window_size = 2 * half_kernel + 1;

                // If we've reached the maximum window size, use the median
                if half_kernel > max_half_kernel {
                    filtered[i] = median;
                }
            }
        }
    }

    Ok(filtered)
}

/// Applies median filtering to a 2D image.
///
/// Median filtering is particularly effective at removing salt-and-pepper noise
/// from images while preserving edges.
///
/// # Arguments
/// * `image` - Input image (2D array)
/// * `kernel_size` - Size of the filtering window (must be odd)
/// * `config` - Filtering configuration
///
/// # Returns
/// * The filtered image
///
/// # Example
/// ```
/// use scirs2_core::ndarray::Array2;
/// use scirs2_signal::median::{median_filter_2d, MedianConfig};
///
/// let image = Array2::from_shape_fn((5, 5), |(i, j)| {
///     if i == 2 && j == 2 { 100.0 } else { 1.0 }  // Center pixel is an outlier
/// });
/// let config = MedianConfig::default();
/// let filtered = median_filter_2d(&image, 3, &config).unwrap();
/// ```
#[allow(dead_code)]
pub fn median_filter_2d(
    image: &Array2<f64>,
    kernel_size: usize,
    config: &MedianConfig,
) -> SignalResult<Array2<f64>> {
    // Validate kernel _size
    if kernel_size % 2 != 1 {
        return Err(SignalError::ValueError(
            "Kernel _size must be odd".to_string(),
        ));
    }

    let (height, width) = image.dim();

    // If image is too small, return a copy
    if height <= 1 || width <= 1 || kernel_size > height || kernel_size > width {
        return Ok(image.clone());
    }

    let half_kernel = kernel_size / 2;

    // Create padded image based on edge mode
    let paddedimage = padimage_2d(image, half_kernel, config.edge_mode);

    // Apply either standard or adaptive median filtering
    if config.adaptive {
        adaptive_median_filter_2d(image, &paddedimage, half_kernel, config)
    } else if config.center_weighted {
        center_weighted_median_filter_2d(image, &paddedimage, half_kernel, config)
    } else {
        standard_median_filter_2d(image, &paddedimage, half_kernel)
    }
}

/// Applies standard median filtering to a 2D image
#[allow(dead_code)]
fn standard_median_filter_2d(
    image: &Array2<f64>,
    paddedimage: &Array2<f64>,
    half_kernel: usize,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();
    let mut filtered = Array2::zeros((height, width));

    // Process each pixel in the image
    for i in 0..height {
        for j in 0..width {
            // Extract window around current pixel
            let window_i_start = i;
            let window_i_end = i + 2 * half_kernel + 1;
            let window_j_start = j;
            let window_j_end = j + 2 * half_kernel + 1;

            // Ensure window is within bounds
            if window_i_end > paddedimage.dim().0 || window_j_end > paddedimage.dim().1 {
                return Err(SignalError::DimensionMismatch(
                    "Window extends beyond padded image bounds".to_string(),
                ));
            }

            // Extract window values
            let window = paddedimage.slice(s![
                window_i_start..window_i_end,
                window_j_start..window_j_end
            ]);

            // Flatten and sort window values
            let mut flat_window: Vec<f64> = window.iter().copied().collect();
            flat_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Set output to median value
            let median_idx = flat_window.len() / 2;
            filtered[[i, j]] = flat_window[median_idx];
        }
    }

    Ok(filtered)
}

/// Applies center-weighted median filtering to a 2D image
#[allow(dead_code)]
fn center_weighted_median_filter_2d(
    image: &Array2<f64>,
    paddedimage: &Array2<f64>,
    half_kernel: usize,
    config: &MedianConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();
    let mut filtered = Array2::zeros((height, width));

    // Calculate the center position in the _kernel
    let center_i = half_kernel;
    let center_j = half_kernel;

    // Process each pixel in the image
    for i in 0..height {
        for j in 0..width {
            // Extract window around current pixel
            let window_i_start = i;
            let window_i_end = i + 2 * half_kernel + 1;
            let window_j_start = j;
            let window_j_end = j + 2 * half_kernel + 1;

            // Ensure window is within bounds
            if window_i_end > paddedimage.dim().0 || window_j_end > paddedimage.dim().1 {
                return Err(SignalError::DimensionMismatch(
                    "Window extends beyond padded image bounds".to_string(),
                ));
            }

            // Create weighted window with repeated center value
            let mut weighted_window = Vec::new();

            for wi in 0..(2 * half_kernel + 1) {
                for wj in 0..(2 * half_kernel + 1) {
                    let value = paddedimage[[window_i_start + wi, window_j_start + wj]];

                    // Add center value with higher weight
                    if wi == center_i && wj == center_j {
                        for _ in 0..config.center_weight {
                            weighted_window.push(value);
                        }
                    } else {
                        weighted_window.push(value);
                    }
                }
            }

            // Sort the weighted window values
            weighted_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate the median position
            let median_idx = weighted_window.len() / 2;

            // Set output to weighted median value
            filtered[[i, j]] = weighted_window[median_idx];
        }
    }

    Ok(filtered)
}

/// Applies adaptive median filtering to a 2D image
#[allow(dead_code)]
fn adaptive_median_filter_2d(
    image: &Array2<f64>,
    paddedimage: &Array2<f64>,
    initial_half_kernel: usize,
    config: &MedianConfig,
) -> SignalResult<Array2<f64>> {
    let (height, width) = image.dim();
    let mut filtered = Array2::zeros((height, width));

    // Maximum half _kernel size
    let max_half_kernel = config.max_kernel_size / 2;

    // Process each pixel in the image
    for i in 0..height {
        for j in 0..width {
            // Start with the initial _kernel size
            let mut half_kernel = initial_half_kernel;

            // Get current pixel value
            let curr_val = paddedimage[[i + half_kernel, j + half_kernel]];

            // Adaptive window size adjustment
            while half_kernel <= max_half_kernel {
                let kernel_size = 2 * half_kernel + 1;

                // Calculate offset from initial to current window
                let offset = half_kernel - initial_half_kernel;

                // Extract window around current pixel
                let window_i_start = i + offset;
                let window_i_end = window_i_start + kernel_size;
                let window_j_start = j + offset;
                let window_j_end = window_j_start + kernel_size;

                // Ensure window is within bounds
                if window_i_end > paddedimage.dim().0 || window_j_end > paddedimage.dim().1 {
                    break;
                }

                // Extract window
                let window = paddedimage.slice(s![
                    window_i_start..window_i_end,
                    window_j_start..window_j_end
                ]);

                // Flatten and sort window values
                let mut flat_window: Vec<f64> = window.iter().copied().collect();
                flat_window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                // Calculate window statistics
                let min_val = flat_window[0];
                let max_val = flat_window[flat_window.len() - 1];
                let median_idx = flat_window.len() / 2;
                let median = flat_window[median_idx];

                // Level A: Test if median is impulse
                let level_a = min_val < median && median < max_val;

                if level_a {
                    // Level B: Test if current pixel is impulse
                    let level_b = min_val < curr_val && curr_val < max_val;

                    if level_b {
                        // Not an impulse, keep original value
                        filtered[[i, j]] = curr_val;
                    } else {
                        // Impulse detected, use median
                        filtered[[i, j]] = median;
                    }

                    // Exit the window size adjustment loop
                    break;
                } else {
                    // Median might be impulse, increase window size
                    half_kernel += 1;

                    // If we've reached the maximum window size, use the median
                    if half_kernel > max_half_kernel {
                        filtered[[i, j]] = median;
                    }
                }
            }
        }
    }

    Ok(filtered)
}

/// Applies median filtering to a color image.
///
/// This function processes each color channel independently or jointly
/// depending on the specified method.
///
/// # Arguments
/// * `image` - Input color image (3D array with last axis being color channels)
/// * `kernel_size` - Size of the filtering window (must be odd)
/// * `config` - Filtering configuration
/// * `vector_median` - Whether to use vector median filtering (preserves color relationships)
///
/// # Returns
/// * The filtered color image
#[allow(dead_code)]
pub fn median_filter_color(
    image: &Array3<f64>,
    kernel_size: usize,
    config: &MedianConfig,
    vector_median: bool,
) -> SignalResult<Array3<f64>> {
    let (height, width, channels) = image.dim();

    if vector_median {
        // Vector _median filtering (preserves color relationships)
        vector_median_filter(image, kernel_size, config)
    } else {
        // Channel-by-channel _median filtering
        let mut filtered = Array3::zeros((height, width, channels));

        for c in 0..channels {
            // Extract channel
            let channel = image.index_axis(Axis(2), c).to_owned();

            // Apply _median filtering to the channel
            let filtered_channel = median_filter_2d(&channel, kernel_size, config)?;

            // Store result
            for i in 0..height {
                for j in 0..width {
                    filtered[[i, j, c]] = filtered_channel[[i, j]];
                }
            }
        }

        Ok(filtered)
    }
}

/// Applies vector median filtering to a color image.
///
/// Vector median filtering preserves color relationships by treating
/// RGB pixels as vectors and finding the pixel with minimum sum of
/// distances to other pixels in the window.
///
/// # Arguments
/// * `image` - Input color image
/// * `kernel_size` - Size of the filtering window
/// * `config` - Filtering configuration
///
/// # Returns
/// * The filtered color image
#[allow(dead_code)]
fn vector_median_filter(
    image: &Array3<f64>,
    kernel_size: usize,
    config: &MedianConfig,
) -> SignalResult<Array3<f64>> {
    // Validate kernel _size
    if kernel_size % 2 != 1 {
        return Err(SignalError::ValueError(
            "Kernel _size must be odd".to_string(),
        ));
    }

    let (height, width, channels) = image.dim();

    // If image is too small, return a copy
    if height <= 1 || width <= 1 || kernel_size > height || kernel_size > width {
        return Ok(image.clone());
    }

    let half_kernel = kernel_size / 2;

    // Create padded image for each channel based on edge mode
    let mut padded_channels = Vec::with_capacity(channels);
    for c in 0..channels {
        let channel = image.index_axis(Axis(2), c).to_owned();
        padded_channels.push(padimage_2d(&channel, half_kernel, config.edge_mode));
    }

    // Allocate output image
    let mut filtered = Array3::zeros((height, width, channels));

    // Process each pixel in the image
    for i in 0..height {
        for j in 0..width {
            // Extract windows around current pixel for each channel
            let window_i_start = i;
            let window_i_end = i + 2 * half_kernel + 1;
            let window_j_start = j;
            let window_j_end = j + 2 * half_kernel + 1;

            // Ensure window is within bounds
            if window_i_end > padded_channels[0].dim().0
                || window_j_end > padded_channels[0].dim().1
            {
                return Err(SignalError::DimensionMismatch(
                    "Window extends beyond padded image bounds".to_string(),
                ));
            }

            // Extract all pixels in the window as vectors
            let kernel_size = 2 * half_kernel + 1;
            let window_size = kernel_size * kernel_size;
            let mut window_vectors = Vec::with_capacity(window_size);

            for wi in 0..kernel_size {
                for wj in 0..kernel_size {
                    let pi = window_i_start + wi;
                    let pj = window_j_start + wj;

                    // Extract color vector for this pixel
                    let mut color_vector = Vec::with_capacity(channels);
                    for (_c, padded_channel) in padded_channels.iter().enumerate().take(channels) {
                        color_vector.push(padded_channel[[pi, pj]]);
                    }

                    window_vectors.push(color_vector);
                }
            }

            // Find the vector median
            let vector_median = find_vector_median(&window_vectors);

            // Store the result
            for (c, value) in vector_median.iter().enumerate().take(channels) {
                filtered[[i, j, c]] = *value;
            }
        }
    }

    Ok(filtered)
}

/// Finds the vector median in a collection of vectors
///
/// The vector median is the vector that minimizes the sum of
/// distances to all other vectors in the collection.
#[allow(dead_code)]
fn find_vector_median(vectors: &[Vec<f64>]) -> Vec<f64> {
    if vectors.is_empty() {
        return Vec::new();
    }

    if vectors.len() == 1 {
        return vectors[0].clone();
    }

    // Calculate sum of distances for each vector
    let mut min_distance_sum = f64::INFINITY;
    let mut median_idx = 0;

    for i in 0..vectors.len() {
        let mut distance_sum = 0.0;

        for j in 0..vectors.len() {
            if i != j {
                distance_sum += euclidean_distance(&vectors[i], &vectors[j]);
            }
        }

        if distance_sum < min_distance_sum {
            min_distance_sum = distance_sum;
            median_idx = i;
        }
    }

    vectors[median_idx].clone()
}

/// Computes the Euclidean distance between two vectors
#[allow(dead_code)]
fn euclidean_distance(v1: &[f64], v2: &[f64]) -> f64 {
    if v1.len() != v2.len() {
        return f64::INFINITY;
    }

    let mut sum_squared = 0.0;
    for i in 0..v1.len() {
        let diff = v1[i] - v2[i];
        sum_squared += diff * diff;
    }

    sum_squared.sqrt()
}

/// Applies rank-order filtering to a 1D signal.
///
/// Rank-order filtering is a generalization of median filtering where any
/// rank (percentile) can be selected instead of just the median (50th percentile).
///
/// # Arguments
/// * `signal` - Input signal
/// * `kernel_size` - Size of the filtering window
/// * `rank` - Rank to select (0.0 = minimum, 0.5 = median, 1.0 = maximum)
/// * `edge_mode` - Edge handling mode
///
/// # Returns
/// * The filtered signal
#[allow(dead_code)]
pub fn rank_filter_1d(
    signal: &Array1<f64>,
    kernel_size: usize,
    rank: f64,
    edge_mode: EdgeMode,
) -> SignalResult<Array1<f64>> {
    // Validate parameters
    if kernel_size % 2 != 1 {
        return Err(SignalError::ValueError(
            "Kernel _size must be odd".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&rank) {
        return Err(SignalError::ValueError(
            "Rank must be between 0.0 and 1.0".to_string(),
        ));
    }

    let n = signal.len();

    // If signal is too short, return a copy
    if n <= 1 || kernel_size > n {
        return Ok(signal.clone());
    }

    let half_kernel = kernel_size / 2;

    // Create padded signal based on edge _mode
    let paddedsignal = padsignal_1d(signal, half_kernel, edge_mode);

    // Apply rank filter
    let mut filtered = Array1::zeros(n);

    // Process each point in the signal
    for i in 0..n {
        // Extract window around current point
        let window_start = i;
        let window_end = i + 2 * half_kernel + 1;

        // Ensure window is within bounds
        if window_start >= paddedsignal.len() || window_end > paddedsignal.len() {
            return Err(SignalError::DimensionMismatch(
                "Window extends beyond padded signal bounds".to_string(),
            ));
        }

        // Extract and sort window values
        let mut window: Vec<f64> = paddedsignal.slice(s![window_start..window_end]).to_vec();
        window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate the index for the requested rank
        let rank_idx = ((window.len() - 1) as f64 * rank).round() as usize;

        // Set output to the value at the specified rank
        filtered[i] = window[rank_idx];
    }

    Ok(filtered)
}

/// Applies hybrid median filtering to a 2D image.
///
/// Hybrid median filtering uses multiple structural elements (like crosses and Xs)
/// to better preserve edges in different orientations.
///
/// # Arguments
/// * `image` - Input image
/// * `kernel_size` - Size of the filtering window
/// * `config` - Filtering configuration
///
/// # Returns
/// * The filtered image
#[allow(dead_code)]
pub fn hybrid_median_filter_2d(
    image: &Array2<f64>,
    kernel_size: usize,
    config: &MedianConfig,
) -> SignalResult<Array2<f64>> {
    // Validate kernel _size
    if kernel_size % 2 != 1 {
        return Err(SignalError::ValueError(
            "Kernel _size must be odd".to_string(),
        ));
    }

    let (height, width) = image.dim();

    // If image is too small, return a copy
    if height <= 1 || width <= 1 || kernel_size > height || kernel_size > width {
        return Ok(image.clone());
    }

    let half_kernel = kernel_size / 2;

    // Create padded image based on edge mode
    let paddedimage = padimage_2d(image, half_kernel, config.edge_mode);

    // Allocate output image
    let mut filtered = Array2::zeros((height, width));

    // Process each pixel in the image
    for i in 0..height {
        for j in 0..width {
            // Extract window around current pixel
            let window_i_start = i;
            let window_i_end = i + 2 * half_kernel + 1;
            let window_j_start = j;
            let window_j_end = j + 2 * half_kernel + 1;

            // Ensure window is within bounds
            if window_i_end > paddedimage.dim().0 || window_j_end > paddedimage.dim().1 {
                return Err(SignalError::DimensionMismatch(
                    "Window extends beyond padded image bounds".to_string(),
                ));
            }

            // Extract pixels from different structural elements
            let mut plusshape = Vec::new(); // + shape
            let mut crossshape = Vec::new(); // X shape

            for k in 0..(2 * half_kernel + 1) {
                // Horizontal line (part of + shape)
                plusshape.push(paddedimage[[window_i_start + half_kernel, window_j_start + k]]);

                // Vertical line (part of + shape)
                plusshape.push(paddedimage[[window_i_start + k, window_j_start + half_kernel]]);

                // Diagonal 1 (part of X shape)
                if k < kernel_size {
                    let diag_i = window_i_start + k;
                    let diag_j = window_j_start + k;
                    crossshape.push(paddedimage[[diag_i, diag_j]]);
                }

                // Diagonal 2 (part of X shape)
                if k < kernel_size {
                    let diag_i = window_i_start + k;
                    let diag_j = window_j_start + kernel_size - 1 - k;
                    crossshape.push(paddedimage[[diag_i, diag_j]]);
                }
            }

            // Remove duplicate center pixel
            if !plusshape.is_empty() {
                plusshape.pop();
            }

            // Sort the values from each shape
            plusshape.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            crossshape.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Get median for each shape
            let plus_median = plusshape[plusshape.len() / 2];
            let cross_median = crossshape[crossshape.len() / 2];

            // Get the original pixel value
            let orig_value =
                paddedimage[[window_i_start + half_kernel, window_j_start + half_kernel]];

            // Find the median of the three values: plus_median, cross_median, original
            let mut final_values = [plus_median, cross_median, orig_value];
            final_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Set output to the median of the three values
            filtered[[i, j]] = final_values[1];
        }
    }

    Ok(filtered)
}

/// Helper function to pad a 1D signal for edge handling
#[allow(dead_code)]
fn padsignal_1d(signal: &Array1<f64>, pad_size: usize, edgemode: EdgeMode) -> Array1<f64> {
    let n = signal.len();
    let mut padded = Array1::zeros(n + 2 * pad_size);

    // Copy original signal
    for i in 0..n {
        padded[i + pad_size] = signal[i];
    }

    // Apply padding based on edge _mode
    match edgemode {
        EdgeMode::Reflect => {
            // Reflect the signal at boundaries
            for i in 0..pad_size {
                // Left boundary: reflect
                padded[pad_size - 1 - i] = signal[i.min(n - 1)];

                // Right boundary: reflect
                padded[n + pad_size + i] = signal[n - 1 - i.min(n - 1)];
            }
        }
        EdgeMode::Nearest => {
            // Pad with the nearest valid value
            let first_val = signal[0];
            let last_val = signal[n - 1];

            for i in 0..pad_size {
                padded[i] = first_val;
                padded[n + pad_size + i] = last_val;
            }
        }
        EdgeMode::Constant(value) => {
            // Pad with a constant value
            for i in 0..pad_size {
                padded[i] = value;
                padded[n + pad_size + i] = value;
            }
        }
        EdgeMode::Wrap => {
            // Wrap around (circular padding)
            for i in 0..pad_size {
                padded[i] = signal[(n - pad_size + i) % n];
                padded[n + pad_size + i] = signal[i % n];
            }
        }
    }

    padded
}

/// Helper function to pad a 2D image for edge handling
#[allow(dead_code)]
fn padimage_2d(image: &Array2<f64>, pad_size: usize, edgemode: EdgeMode) -> Array2<f64> {
    let (height, width) = image.dim();
    let mut padded = Array2::zeros((height + 2 * pad_size, width + 2 * pad_size));

    // Copy original image
    for i in 0..height {
        for j in 0..width {
            padded[[i + pad_size, j + pad_size]] = image[[i, j]];
        }
    }

    // Apply padding based on edge _mode
    match edgemode {
        EdgeMode::Reflect => {
            // Reflect the image at boundaries

            // Top and bottom edges
            for i in 0..pad_size {
                for j in 0..width {
                    // Top edge
                    padded[[pad_size - 1 - i, j + pad_size]] = image[[i.min(height - 1), j]];

                    // Bottom edge
                    padded[[height + pad_size + i, j + pad_size]] =
                        image[[height - 1 - i.min(height - 1), j]];
                }
            }

            // Left and right edges
            for i in 0..height + 2 * pad_size {
                for j in 0..pad_size {
                    // Map to valid row in the padded image
                    let src_i = if i < pad_size {
                        2 * pad_size - i - 1
                    } else if i >= height + pad_size {
                        2 * (height + pad_size) - i - 1
                    } else {
                        i
                    };

                    // Left edge
                    padded[[i, pad_size - 1 - j]] = padded[[src_i, pad_size + j.min(width - 1)]];

                    // Right edge
                    padded[[i, width + pad_size + j]] =
                        padded[[src_i, width + pad_size - 1 - j.min(width - 1)]];
                }
            }
        }
        EdgeMode::Nearest => {
            // Pad with the nearest valid value

            // Top and bottom edges
            for i in 0..pad_size {
                for j in 0..width {
                    // Top edge
                    padded[[i, j + pad_size]] = image[[0, j]];

                    // Bottom edge
                    padded[[height + pad_size + i, j + pad_size]] = image[[height - 1, j]];
                }
            }

            // Left and right edges
            for i in 0..height + 2 * pad_size {
                for j in 0..pad_size {
                    // Get the nearest valid column
                    let col_left = 0;
                    let col_right = width - 1;

                    // Map to valid row
                    let row = if i < pad_size {
                        0
                    } else if i >= height + pad_size {
                        height - 1
                    } else {
                        i - pad_size
                    };

                    // Left edge
                    padded[[i, j]] = image[[row, col_left]];

                    // Right edge
                    padded[[i, width + pad_size + j]] = image[[row, col_right]];
                }
            }
        }
        EdgeMode::Constant(value) => {
            // Pad with a constant value

            // Top and bottom edges
            for i in 0..pad_size {
                for j in 0..width + 2 * pad_size {
                    padded[[i, j]] = value;
                    padded[[height + pad_size + i, j]] = value;
                }
            }

            // Left and right edges
            for i in pad_size..height + pad_size {
                for j in 0..pad_size {
                    padded[[i, j]] = value;
                    padded[[i, width + pad_size + j]] = value;
                }
            }
        }
        EdgeMode::Wrap => {
            // Wrap around (circular padding)

            // Top and bottom edges
            for i in 0..pad_size {
                for j in 0..width {
                    // Top edge
                    padded[[i, j + pad_size]] = image[[(height - pad_size + i) % height, j]];

                    // Bottom edge
                    padded[[height + pad_size + i, j + pad_size]] = image[[i % height, j]];
                }
            }

            // Left and right edges
            for i in 0..height + 2 * pad_size {
                for j in 0..pad_size {
                    // Map to valid row in the padded image
                    let src_i = if i < pad_size {
                        (height - pad_size + i) % height + pad_size
                    } else if i >= height + pad_size {
                        (i - pad_size) % height + pad_size
                    } else {
                        i
                    };

                    // Left edge
                    padded[[i, j]] = padded[[src_i, width + j]];

                    // Right edge
                    padded[[i, width + pad_size + j]] = padded[[src_i, pad_size + j]];
                }
            }
        }
    }

    padded
}
