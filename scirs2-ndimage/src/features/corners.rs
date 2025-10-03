//! Corner detection algorithms
//!
//! This module provides functions for detecting corners in images,
//! including Harris corner detection and other corner detection methods.

use crate::filters::{sobel, BorderMode};
use scirs2_core::ndarray::{Array, ArrayD, Dimension, Ix2};
use scirs2_core::numeric::{Float, NumAssign};

/// Harris corner detector
///
/// The Harris corner detector is a corner detection operator that uses the autocorrelation
/// matrix of gradients to determine if a pixel is a corner.
///
/// # Arguments
///
/// * `image` - Input 2D array
/// * `block_size` - Size of the window to consider for each point
/// * `k` - Harris detector free parameter (typically 0.04 to 0.06)
/// * `threshold` - Response threshold for corner detection
///
/// # Returns
///
/// * Array with corner locations marked as true
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::features::harris_corners;
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// let corners = harris_corners(&image, 3, 0.04, 0.01);
/// ```
#[allow(dead_code)]
pub fn harris_corners(
    image: &Array<f32, Ix2>,
    block_size: usize,
    k: f32,
    threshold: f32,
) -> Array<bool, Ix2> {
    // Convert 2D array to dynamic array for processing
    let image_dim = image.raw_dim();
    let mut image_d = ArrayD::from_elem(image_dim.into_dyn(), 0.0f32);
    for (idx, &val) in image.indexed_iter() {
        let idx_d = [idx.0, idx.1];
        image_d[idx_d] = val;
    }

    // Step 1: Calculate gradients using Sobel
    let gradient_y = sobel(&image_d, 0, Some(BorderMode::Reflect)).unwrap();
    let gradient_x = sobel(&image_d, 1, Some(BorderMode::Reflect)).unwrap();

    // Step 2: Calculate products of derivatives at each pixel
    let mut ix2: ArrayD<f32> = gradient_x.clone().mapv(|x| x * x);
    let mut iy2: ArrayD<f32> = gradient_y.clone().mapv(|y| y * y);
    let mut ixy: ArrayD<f32> = gradient_x.clone();

    for (idx, &y) in gradient_y.indexed_iter() {
        ixy[idx.slice()] *= y;
    }

    // Step 3: Gaussian filtering to smooth the derivatives
    let sigma = 0.5 * (block_size as f32 - 1.0) / 3.0; // scale sigmas based on block _size
                                                       // Use specialized f32 version that doesn't require Send/Sync bounds
    ix2 =
        crate::filters::gaussian_filter_f32(&ix2, sigma, Some(BorderMode::Reflect), None).unwrap();
    iy2 =
        crate::filters::gaussian_filter_f32(&iy2, sigma, Some(BorderMode::Reflect), None).unwrap();
    ixy =
        crate::filters::gaussian_filter_f32(&ixy, sigma, Some(BorderMode::Reflect), None).unwrap();

    // Convert back to 2D arrays for the response calculation
    let shape = image.raw_dim();
    let mut ix2_2d = Array::zeros(shape);
    let mut iy2_2d = Array::zeros(shape);
    let mut ixy_2d = Array::zeros(shape);

    for idx in image.indexed_iter() {
        let pos = idx.0;
        let idx_d = [pos.0, pos.1];
        ix2_2d[pos] = ix2[idx_d.as_ref()];
        iy2_2d[pos] = iy2[idx_d.as_ref()];
        ixy_2d[pos] = ixy[idx_d.as_ref()];
    }

    // Step 4: Calculate the response function
    // R = det(M) - k * trace(M)^2
    // det(M) = ix2 * iy2 - ixy^2
    // trace(M) = ix2 + iy2
    let mut response = Array::zeros(shape);
    for idx in response.indexed_iter_mut() {
        let pos = idx.0;
        let det = ix2_2d[pos] * iy2_2d[pos] - ixy_2d[pos] * ixy_2d[pos];
        let trace = ix2_2d[pos] + iy2_2d[pos];
        *idx.1 = det - k * trace * trace;
    }

    // Step 5: Find local maximum using non-maximum suppression
    let mut corners = Array::from_elem(shape, false);
    let (rows, cols) = (shape[0], shape[1]);
    let window_radius = block_size / 2;

    for row in window_radius..(rows - window_radius) {
        for col in window_radius..(cols - window_radius) {
            let r = response[(row, col)];

            // Only consider points above threshold
            if r > threshold {
                // Check if it's a local maximum in the neighborhood
                let mut is_local_max = true;

                // Compare with all points in the window
                'window: for i in (row - window_radius)..=(row + window_radius) {
                    for j in (col - window_radius)..=(col + window_radius) {
                        if i < rows && j < cols && !(i == row && j == col) && response[(i, j)] > r {
                            is_local_max = false;
                            break 'window;
                        }
                    }
                }

                if is_local_max {
                    corners[(row, col)] = true;
                }
            }
        }
    }

    corners
}

/// Fast corner detector (FAST: Features from Accelerated Segment Test)
///
/// A simplified implementation of the FAST corner detection algorithm that
/// looks for a ring of pixels that are all brighter or darker than a central pixel
/// by more than a threshold.
///
/// # Arguments
///
/// * `image` - Input 2D array
/// * `threshold` - Intensity threshold between central pixel and surrounding pixels
/// * `n` - Number of contiguous pixels that must be brighter or darker (8 to 12 typical)
///
/// # Returns
///
/// * Array with corner locations marked as true
///
/// # Example
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_ndimage::features::fast_corners;
///
/// let image = array![
///     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///     [0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
/// ];
///
/// let corners = fast_corners(&image, 0.3, 9);
/// ```
#[allow(dead_code)]
pub fn fast_corners<T>(image: &Array<T, Ix2>, threshold: T, n: usize) -> Array<bool, Ix2>
where
    T: Float + NumAssign + scirs2_core::numeric::FromPrimitive,
{
    let shape = image.raw_dim();
    let (rows, cols) = (shape[0], shape[1]);
    let mut corners = Array::from_elem(shape, false);

    // FAST requires at least a 3x3 neighborhood around each pixel
    if rows < 7 || cols < 7 {
        return corners;
    }

    // Bresenham's circle with radius 3
    // These are the coordinates of 16 points on a circle of radius 3
    let circle_points = [
        (0, 3),
        (1, 3),
        (2, 2),
        (3, 1),
        (3, 0),
        (3, -1),
        (2, -2),
        (1, -3),
        (0, -3),
        (-1, -3),
        (-2, -2),
        (-3, -1),
        (-3, 0),
        (-3, 1),
        (-2, 2),
        (-1, 3),
    ]
    .map(|(y, x)| (y as isize, x as isize));

    // Examine each pixel (except the border)
    for row in 3..(rows - 3) {
        for col in 3..(cols - 3) {
            let center_value = image[(row, col)];
            let high_threshold = center_value + threshold;
            let low_threshold = center_value - threshold;

            // Count consecutive points that are brighter or darker than the center
            let mut is_corner = false;
            let num_points = circle_points.len();

            for start in 0..num_points {
                let mut _count_brighter = 0;
                let mut _count_darker = 0;
                let mut consecutive_brighter = 0;
                let mut consecutive_darker = 0;

                // Check each point on the circle
                for offset in 0..num_points {
                    let idx = (start + offset) % num_points;
                    let (dy, dx) = circle_points[idx];
                    let r = (row as isize + dy) as usize;
                    let c = (col as isize + dx) as usize;
                    let pixel_value = image[(r, c)];

                    if pixel_value > high_threshold {
                        consecutive_brighter += 1;
                        consecutive_darker = 0;
                    } else if pixel_value < low_threshold {
                        consecutive_darker += 1;
                        consecutive_brighter = 0;
                    } else {
                        consecutive_brighter = 0;
                        consecutive_darker = 0;
                    }

                    // Check if we have enough contiguous points
                    if consecutive_brighter >= n || consecutive_darker >= n {
                        is_corner = true;
                        break;
                    }
                }

                // Also check if we wrap around the circle
                if !is_corner {
                    for offset in 0..n - 1 {
                        let idx = (start + offset) % num_points;
                        let (dy, dx) = circle_points[idx];
                        let r = (row as isize + dy) as usize;
                        let c = (col as isize + dx) as usize;
                        let pixel_value = image[(r, c)];

                        if pixel_value > high_threshold {
                            consecutive_brighter += 1;
                        } else {
                            consecutive_brighter = 0;
                        }

                        if pixel_value < low_threshold {
                            consecutive_darker += 1;
                        } else {
                            consecutive_darker = 0;
                        }
                    }

                    if consecutive_brighter >= n || consecutive_darker >= n {
                        is_corner = true;
                    }
                }

                if is_corner {
                    break;
                }
            }

            if is_corner {
                corners[(row, col)] = true;
            }
        }
    }

    corners
}
