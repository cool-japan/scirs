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
// It also includes other outlier-robust filters (merged in from the former
// standalone `robust` module), which share this module's [`EdgeMode`]:
// - Alpha-trimmed mean filtering
// - Hampel filter for outlier detection and replacement
// - Winsorized filtering
// - Huber loss-based robust filtering
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
// let filtered = median_filter_1d(&signal, 3, &config).expect("operation should succeed");
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
/// let filtered = median_filter_1d(&signal, 3, &config).expect("operation should succeed");
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
/// let filtered = median_filter_2d(&image, 3, &config).expect("operation should succeed");
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

// ---------------------------------------------------------------------------
// Robust filtering (alpha-trimmed mean, Hampel, Winsorize, Huber)
//
// Merged in from the former standalone `robust` module: these filters share
// this module's [`EdgeMode`] and complement the median-based filters above
// with other outlier-robust estimators for signal and image processing.
// ---------------------------------------------------------------------------

/// Configuration for robust filtering algorithms
#[derive(Debug, Clone)]
pub struct RobustConfig {
    /// Edge handling mode
    pub edge_mode: EdgeMode,

    /// Whether to return outlier positions
    pub return_outliers: bool,

    /// Parallelization enabled
    pub parallel: bool,
}

impl Default for RobustConfig {
    fn default() -> Self {
        Self {
            edge_mode: EdgeMode::Reflect,
            return_outliers: false,
            parallel: false,
        }
    }
}

/// Alpha-trimmed mean filter for robust signal processing
///
/// This filter removes the α% largest and smallest values from a local window
/// and computes the mean of the remaining values. It provides robustness
/// against outliers while maintaining computational efficiency.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `alpha` - Trimming fraction (0.0 to 0.5). Higher values remove more outliers.
///
/// # Returns
///
/// * Filtered signal with the same length as input
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::median::alpha_trimmed_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let filtered = alpha_trimmed_filter(&signal, 3, 0.2).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn alpha_trimmed_filter(
    signal: &Array1<f64>,
    window_size: usize,
    alpha: f64,
) -> SignalResult<Array1<f64>> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window _size must be odd and >= 3".to_string(),
        ));
    }

    if !(0.0..=0.5).contains(&alpha) {
        return Err(SignalError::ValueError(
            "Alpha must be between 0.0 and 0.5".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(n);

    // Number of samples to trim from each end
    let trim_count = (window_size as f64 * alpha).floor() as usize;
    let keep_count = window_size - 2 * trim_count;

    if keep_count == 0 {
        return Err(SignalError::ValueError(
            "Alpha value too large for given window _size".to_string(),
        ));
    }

    for i in 0..n {
        // Determine window bounds with boundary handling
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases by padding if necessary
        while window_values.len() < window_size {
            if start == 0 {
                // Pad at beginning by reflecting
                window_values.insert(0, window_values[0]);
            } else {
                // Pad at end by reflecting
                window_values.push(*window_values.last().expect("Operation should succeed"));
            }
        }

        // Sort window values
        window_values.sort_by(|a, b| a.partial_cmp(b).expect("Operation should succeed"));

        // Trim alpha portion from both ends and compute mean
        let trimmed_values = &window_values[trim_count..window_values.len() - trim_count];
        let trimmed_mean = trimmed_values.iter().sum::<f64>() / trimmed_values.len() as f64;

        result[i] = trimmed_mean;
    }

    Ok(result)
}

/// Hampel filter for outlier detection and replacement
///
/// The Hampel filter detects outliers by comparing each point to the median
/// of its local neighborhood. Outliers are identified when they deviate more
/// than k times the median absolute deviation (MAD) from the local median.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `k` - Threshold factor (typically 2.0 to 3.0)
///
/// # Returns
///
/// * Tuple of (filtered_signal, outlier_indices)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::median::hampel_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let (filtered, outliers) = hampel_filter(&signal, 3, 3.0).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn hampel_filter(
    signal: &Array1<f64>,
    window_size: usize,
    k: f64,
) -> SignalResult<(Array1<f64>, Vec<usize>)> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window _size must be odd and >= 3".to_string(),
        ));
    }

    if k <= 0.0 {
        return Err(SignalError::ValueError(
            "Threshold factor k must be positive".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok((Array1::zeros(0), Vec::new()));
    }

    let half_window = window_size / 2;
    let mut result = signal.clone();
    let mut outlier_indices = Vec::new();

    for i in 0..n {
        // Determine window bounds
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases
        while window_values.len() < window_size {
            if start == 0 {
                window_values.insert(0, window_values[0]);
            } else {
                window_values.push(*window_values.last().expect("Operation should succeed"));
            }
        }

        // Calculate median
        let mut sorted_values = window_values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).expect("Operation should succeed"));
        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        // Calculate MAD (Median Absolute Deviation)
        let mut abs_deviations: Vec<f64> =
            window_values.iter().map(|&x| (x - median).abs()).collect();
        abs_deviations.sort_by(|a, b| a.partial_cmp(b).expect("Operation should succeed"));

        let mad = if abs_deviations.len() % 2 == 0 {
            let mid = abs_deviations.len() / 2;
            (abs_deviations[mid - 1] + abs_deviations[mid]) / 2.0
        } else {
            abs_deviations[abs_deviations.len() / 2]
        };

        // Check if current point is an outlier
        let current_value = signal[i];
        let deviation = (current_value - median).abs();

        if mad > 0.0 && deviation > k * mad {
            // Point is an outlier - replace with median
            result[i] = median;
            outlier_indices.push(i);
        }
    }

    Ok((result, outlier_indices))
}

/// Winsorized filter for robust signal processing
///
/// This filter replaces extreme values with the nearest non-extreme values.
/// Values below the p-th percentile are replaced with the p-th percentile value,
/// and values above the (100-p)-th percentile are replaced with the (100-p)-th percentile value.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `percentile` - Percentile for winsorization (0.0 to 50.0)
///
/// # Returns
///
/// * Filtered signal
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::median::winsorize_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let filtered = winsorize_filter(&signal, 3, 10.0).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn winsorize_filter(
    signal: &Array1<f64>,
    window_size: usize,
    percentile: f64,
) -> SignalResult<Array1<f64>> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window _size must be odd and >= 3".to_string(),
        ));
    }

    if !(0.0..=50.0).contains(&percentile) {
        return Err(SignalError::ValueError(
            "Percentile must be between 0.0 and 50.0".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        // Determine window bounds
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract and sort window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases
        while window_values.len() < window_size {
            if start == 0 {
                window_values.insert(0, window_values[0]);
            } else {
                window_values.push(*window_values.last().expect("Operation should succeed"));
            }
        }

        window_values.sort_by(|a, b| a.partial_cmp(b).expect("Operation should succeed"));

        // Calculate percentile indices
        let lower_idx = ((percentile / 100.0) * (window_values.len() - 1) as f64) as usize;
        let upper_idx =
            (((100.0 - percentile) / 100.0) * (window_values.len() - 1) as f64) as usize;

        let lower_threshold = window_values[lower_idx];
        let upper_threshold = window_values[upper_idx];

        // Winsorize the center value
        let current_value = signal[i];
        result[i] = if current_value < lower_threshold {
            lower_threshold
        } else if current_value > upper_threshold {
            upper_threshold
        } else {
            current_value
        };
    }

    Ok(result)
}

/// Huber loss-based robust filter
///
/// This filter uses the Huber loss function to provide robustness against outliers
/// while maintaining efficiency for inliers. The Huber loss is quadratic for small
/// residuals and linear for large residuals.
///
/// # Arguments
///
/// * `signal` - Input signal
/// * `window_size` - Size of the filtering window (must be odd and >= 3)
/// * `delta` - Threshold parameter for Huber loss (transition point between quadratic and linear)
///
/// # Returns
///
/// * Filtered signal
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::Array1;
/// use scirs2_signal::median::huber_filter;
///
/// let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3]);
/// let filtered = huber_filter(&signal, 3, 1.35).expect("Operation failed");
/// ```
#[allow(dead_code)]
pub fn huber_filter(
    signal: &Array1<f64>,
    window_size: usize,
    delta: f64,
) -> SignalResult<Array1<f64>> {
    if window_size % 2 == 0 || window_size < 3 {
        return Err(SignalError::ValueError(
            "Window _size must be odd and >= 3".to_string(),
        ));
    }

    if delta <= 0.0 {
        return Err(SignalError::ValueError(
            "Delta parameter must be positive".to_string(),
        ));
    }

    let n = signal.len();
    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    let half_window = window_size / 2;
    let mut result = Array1::zeros(n);

    for i in 0..n {
        // Determine window bounds
        let start = i.saturating_sub(half_window);
        let end = if i + half_window < n {
            i + half_window + 1
        } else {
            n
        };

        // Extract window values
        let mut window_values: Vec<f64> = signal.slice(s![start..end]).to_vec();

        // Handle edge cases
        while window_values.len() < window_size {
            if start == 0 {
                window_values.insert(0, window_values[0]);
            } else {
                window_values.push(*window_values.last().expect("Operation should succeed"));
            }
        }

        // Calculate robust estimate using Huber loss iteratively
        let mut estimate = window_values.iter().sum::<f64>() / window_values.len() as f64; // Start with mean

        // Iterative reweighting for Huber loss
        for _iter in 0..10 {
            // Maximum 10 iterations
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for &value in &window_values {
                let residual = value - estimate;
                let abs_residual = residual.abs();

                let weight = if abs_residual <= delta {
                    1.0 // Quadratic regime
                } else {
                    delta / abs_residual // Linear regime
                };

                weighted_sum += weight * value;
                weight_sum += weight;
            }

            let new_estimate = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                estimate
            };

            // Check for convergence
            if (new_estimate - estimate).abs() < 1e-6 {
                break;
            }
            estimate = new_estimate;
        }

        result[i] = estimate;
    }

    Ok(result)
}

/// Apply robust filter to 2D data (images)
///
/// This function applies any of the 1D robust filters to each row and then each column
/// of a 2D array for robust image filtering.
///
/// # Arguments
///
/// * `image` - Input 2D array
/// * `filter_fn` - 1D robust filter function to apply
/// * `window_size` - Size of the filtering window
/// * `param` - Additional parameter for the filter function
///
/// # Returns
///
/// * Filtered 2D array
#[allow(dead_code)]
pub fn robust_filter_2d<F>(
    image: &Array2<f64>,
    filter_fn: F,
    window_size: usize,
    param: f64,
) -> SignalResult<Array2<f64>>
where
    F: Fn(&Array1<f64>, usize, f64) -> SignalResult<Array1<f64>>,
{
    let (rows, cols) = image.dim();
    if rows == 0 || cols == 0 {
        return Ok(Array2::zeros((0, 0)));
    }

    let mut result = image.clone();

    // Filter rows
    for i in 0..rows {
        let row = image.row(i).to_owned();
        let filtered_row = filter_fn(&row, window_size, param)?;
        result.row_mut(i).assign(&filtered_row);
    }

    // Filter columns
    for j in 0..cols {
        let col = result.column(j).to_owned();
        let filtered_col = filter_fn(&col, window_size, param)?;
        result.column_mut(j).assign(&filtered_col);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_trimmed_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 10.0, 1.1, 1.3, 1.15, -5.0, 1.25]);
        let filtered = alpha_trimmed_filter(&signal, 3, 0.3).expect("Operation should succeed");

        assert_eq!(filtered.len(), signal.len());

        // The outliers (10.0 and -5.0) should be reduced
        assert!(filtered[2] < signal[2]); // 10.0 should be reduced
        assert!(filtered[6] > signal[6]); // -5.0 should be increased
    }

    #[test]
    fn test_hampel_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 10.0, 1.3, 1.2, 1.1]);
        let (filtered, outliers) =
            hampel_filter(&signal, 3, 3.0).expect("Operation should succeed");

        assert_eq!(filtered.len(), signal.len());
        assert!(!outliers.is_empty()); // Should detect the outlier at index 3
        assert!(outliers.contains(&3)); // Index 3 has the outlier (10.0)
    }

    #[test]
    fn test_winsorize_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 10.0, 1.3, 1.2, 1.1]);
        let filtered = winsorize_filter(&signal, 5, 20.0).expect("Operation should succeed");

        assert_eq!(filtered.len(), signal.len());
        // Extreme values should be winsorized
        assert!(filtered[3] <= signal[3]); // 10.0 should be reduced
    }

    #[test]
    fn test_huber_filter() {
        let signal = Array1::from_vec(vec![1.0, 1.2, 1.1, 10.0, 1.3, 1.2, 1.1]);
        let filtered = huber_filter(&signal, 3, 1.0).expect("Operation should succeed");

        assert_eq!(filtered.len(), signal.len());
        // All values should be finite
        for &val in filtered.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_robust_filter_2d() {
        let image =
            Array2::from_shape_vec((3, 3), vec![1.0, 1.2, 1.1, 1.1, 10.0, 1.2, 1.3, 1.2, 1.1])
                .expect("Operation should succeed");

        let filtered = robust_filter_2d(&image, alpha_trimmed_filter, 3, 0.2)
            .expect("Operation should succeed");

        assert_eq!(filtered.dim(), image.dim());
        // The outlier (10.0) should be reduced
        assert!(filtered[[1, 1]] < image[[1, 1]]);
    }

    #[test]
    fn test_robust_edge_cases() {
        // Empty signal
        let empty_signal = Array1::zeros(0);
        let result = alpha_trimmed_filter(&empty_signal, 3, 0.2).expect("Operation should succeed");
        assert_eq!(result.len(), 0);

        // Small signal
        let small_signal = Array1::from_vec(vec![1.0, 2.0]);
        let result = alpha_trimmed_filter(&small_signal, 3, 0.2).expect("Operation should succeed");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_robust_parameter_validation() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Invalid window size (even)
        assert!(alpha_trimmed_filter(&signal, 4, 0.2).is_err());

        // Invalid alpha (too large)
        assert!(alpha_trimmed_filter(&signal, 3, 0.6).is_err());

        // Invalid k parameter for Hampel filter
        assert!(hampel_filter(&signal, 3, -1.0).is_err());

        // Invalid percentile for Winsorize filter
        assert!(winsorize_filter(&signal, 3, 60.0).is_err());

        // Invalid delta for Huber filter
        assert!(huber_filter(&signal, 3, -1.0).is_err());
    }
}
