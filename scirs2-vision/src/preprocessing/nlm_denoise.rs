//! Non-local means denoising
//!
//! Non-local means (NLM) is a powerful denoising algorithm that exploits
//! the self-similarity of natural images. Unlike local filters that only
//! consider nearby pixels, NLM searches for similar patches throughout the
//! entire image and uses their weighted average for denoising.

use crate::error::Result;
use ndarray::{s, Array2, Array3};
use rayon::prelude::*;
use std::sync::Mutex;

/// Apply non-local means denoising to a grayscale image
///
/// # Arguments
///
/// * `input` - Noisy input image (values in [0, 1])
/// * `h` - Filtering parameter controlling the decay of weights (higher h removes more noise)
/// * `template_window_size` - Size of patches to compare (typically 7)
/// * `search_window_size` - Size of search window (typically 21)
///
/// # Returns
///
/// * Result containing the denoised image
///
/// # Example
///
/// ```rust
/// use ndarray::Array2;
/// use scirs2_vision::preprocessing::nlm_denoise;
///
/// # fn main() -> scirs2_vision::error::Result<()> {
/// let noisy_image = Array2::zeros((100, 100));
/// let denoised = nlm_denoise(&noisy_image, 0.1, 7, 21)?;
/// # Ok(())
/// # }
/// ```
pub fn nlm_denoise(
    input: &Array2<f32>,
    h: f32,
    template_window_size: usize,
    search_window_size: usize,
) -> Result<Array2<f32>> {
    let (height, width) = input.dim();

    // Ensure odd window sizes
    let template_size = if template_window_size % 2 == 0 {
        template_window_size + 1
    } else {
        template_window_size
    };

    let search_size = if search_window_size % 2 == 0 {
        search_window_size + 1
    } else {
        search_window_size
    };

    let template_radius = template_size / 2;
    let search_radius = search_size / 2;

    // Pad the input image
    let pad_size = search_radius + template_radius;
    let padded = pad_reflect(input, pad_size);

    // Create output array
    let mut output = Array2::zeros((height, width));

    // Pre-compute squared h
    let h_squared = h * h;

    // Process each pixel
    for y in 0..height {
        for x in 0..width {
            let py = y + pad_size;
            let px = x + pad_size;

            // Get the template patch around current pixel
            let template_patch = padded.slice(s![
                (py - template_radius)..=(py + template_radius),
                (px - template_radius)..=(px + template_radius)
            ]);

            let mut weight_sum = 0.0;
            let mut weighted_value_sum = 0.0;

            // Search in the window around current pixel
            for sy in (py.saturating_sub(search_radius))
                ..=(py + search_radius).min(padded.nrows() - template_radius - 1)
            {
                for sx in (px.saturating_sub(search_radius))
                    ..=(px + search_radius).min(padded.ncols() - template_radius - 1)
                {
                    // Skip if we can't extract a full template
                    if sy < template_radius || sx < template_radius {
                        continue;
                    }

                    // Get the comparison patch
                    let compare_patch = padded.slice(s![
                        (sy - template_radius)..=(sy + template_radius),
                        (sx - template_radius)..=(sx + template_radius)
                    ]);

                    // Compute squared distance between patches
                    let mut distance = 0.0;
                    for i in 0..template_size {
                        for j in 0..template_size {
                            let diff = template_patch[[i, j]] - compare_patch[[i, j]];
                            distance += diff * diff;
                        }
                    }
                    distance /= (template_size * template_size) as f32;

                    // Compute weight
                    let weight = (-distance / h_squared).exp();

                    // Accumulate weighted value
                    weight_sum += weight;
                    weighted_value_sum += weight * padded[[sy, sx]];
                }
            }

            // Normalize and store result
            output[[y, x]] = if weight_sum > 0.0 {
                weighted_value_sum / weight_sum
            } else {
                input[[y, x]]
            };
        }
    }

    Ok(output)
}

/// Apply non-local means denoising to a color image
///
/// # Arguments
///
/// * `input` - Noisy input image (HxWx3, values in [0, 1])
/// * `h` - Filtering parameter controlling the decay of weights
/// * `template_window_size` - Size of patches to compare
/// * `search_window_size` - Size of search window
///
/// # Returns
///
/// * Result containing the denoised color image
pub fn nlm_denoise_color(
    input: &Array3<f32>,
    h: f32,
    template_window_size: usize,
    search_window_size: usize,
) -> Result<Array3<f32>> {
    let (height, width, channels) = input.dim();

    if channels != 3 {
        return Err(crate::error::VisionError::InvalidParameter(
            "Input must be an HxWx3 color image".to_string(),
        ));
    }

    let mut output = Array3::zeros((height, width, 3));

    // Process each channel independently
    for c in 0..3 {
        let channel = input.slice(s![.., .., c]).to_owned();
        let denoised = nlm_denoise(&channel, h, template_window_size, search_window_size)?;
        output.slice_mut(s![.., .., c]).assign(&denoised);
    }

    Ok(output)
}

/// Fast non-local means denoising using parallel processing
///
/// This version uses Rayon for parallel processing to speed up computation.
///
/// # Arguments
///
/// * `input` - Noisy input image
/// * `h` - Filtering parameter
/// * `template_window_size` - Size of patches to compare
/// * `search_window_size` - Size of search window
///
/// # Returns
///
/// * Result containing the denoised image
pub fn nlm_denoise_parallel(
    input: &Array2<f32>,
    h: f32,
    template_window_size: usize,
    search_window_size: usize,
) -> Result<Array2<f32>> {
    let (height, width) = input.dim();

    // Ensure odd window sizes
    let template_size = if template_window_size % 2 == 0 {
        template_window_size + 1
    } else {
        template_window_size
    };

    let search_size = if search_window_size % 2 == 0 {
        search_window_size + 1
    } else {
        search_window_size
    };

    let template_radius = template_size / 2;
    let search_radius = search_size / 2;

    // Pad the input image
    let pad_size = search_radius + template_radius;
    let padded = pad_reflect(input, pad_size);

    // Create output array wrapped in Mutex for thread-safe access
    let output = Mutex::new(Array2::zeros((height, width)));

    // Pre-compute squared h
    let h_squared = h * h;

    // Create a vector of all pixel coordinates
    let pixels: Vec<(usize, usize)> = (0..height)
        .flat_map(|y| (0..width).map(move |x| (y, x)))
        .collect();

    // Process pixels in parallel
    pixels.par_iter().for_each(|&(y, x)| {
        let py = y + pad_size;
        let px = x + pad_size;

        // Get the template patch around current pixel
        let template_patch = padded.slice(s![
            (py - template_radius)..=(py + template_radius),
            (px - template_radius)..=(px + template_radius)
        ]);

        let mut weight_sum = 0.0;
        let mut weighted_value_sum = 0.0;

        // Search in the window around current pixel
        for sy in (py.saturating_sub(search_radius))
            ..=(py + search_radius).min(padded.nrows() - template_radius - 1)
        {
            for sx in (px.saturating_sub(search_radius))
                ..=(px + search_radius).min(padded.ncols() - template_radius - 1)
            {
                // Skip if we can't extract a full template
                if sy < template_radius || sx < template_radius {
                    continue;
                }

                // Get the comparison patch
                let compare_patch = padded.slice(s![
                    (sy - template_radius)..=(sy + template_radius),
                    (sx - template_radius)..=(sx + template_radius)
                ]);

                // Compute squared distance between patches
                let mut distance = 0.0;
                for i in 0..template_size {
                    for j in 0..template_size {
                        let diff = template_patch[[i, j]] - compare_patch[[i, j]];
                        distance += diff * diff;
                    }
                }
                distance /= (template_size * template_size) as f32;

                // Compute weight
                let weight = (-distance / h_squared).exp();

                // Accumulate weighted value
                weight_sum += weight;
                weighted_value_sum += weight * padded[[sy, sx]];
            }
        }

        // Normalize and store result
        let value = if weight_sum > 0.0 {
            weighted_value_sum / weight_sum
        } else {
            input[[y, x]]
        };

        // Thread-safe write to output
        output.lock().unwrap()[[y, x]] = value;
    });

    Ok(output.into_inner().unwrap())
}

/// Pad an array with reflected boundary conditions
fn pad_reflect(array: &Array2<f32>, pad_size: usize) -> Array2<f32> {
    let (height, width) = array.dim();
    let new_height = height + 2 * pad_size;
    let new_width = width + 2 * pad_size;

    let mut padded = Array2::zeros((new_height, new_width));

    // Copy original data
    padded
        .slice_mut(s![pad_size..pad_size + height, pad_size..pad_size + width])
        .assign(array);

    // Pad top and bottom
    for i in 0..pad_size {
        // Top
        let src_row = pad_size + i;
        let dst_row = pad_size - i - 1;
        for col in pad_size..pad_size + width {
            padded[[dst_row, col]] = padded[[src_row, col]];
        }

        // Bottom
        let src_row = pad_size + height - i - 1;
        let dst_row = pad_size + height + i;
        for col in pad_size..pad_size + width {
            padded[[dst_row, col]] = padded[[src_row, col]];
        }
    }

    // Pad left and right (including corners)
    for j in 0..pad_size {
        // Left
        let src_col = pad_size + j;
        let dst_col = pad_size - j - 1;
        for row in 0..new_height {
            padded[[row, dst_col]] = padded[[row, src_col]];
        }

        // Right
        let src_col = pad_size + width - j - 1;
        let dst_col = pad_size + width + j;
        for row in 0..new_height {
            padded[[row, dst_col]] = padded[[row, src_col]];
        }
    }

    padded
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_pad_reflect() {
        let array =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let padded = pad_reflect(&array, 1);

        assert_eq!(padded.dim(), (5, 5));

        // Check that the center 3x3 matches the original
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(padded[[i + 1, j + 1]], array[[i, j]]);
            }
        }

        // The current implementation duplicates edge values
        // This is a valid padding strategy for NLM denoising
        assert_eq!(padded[[0, 0]], padded[[1, 1]]); // Corner reflects diagonally
        assert_eq!(padded[[0, 1]], padded[[1, 1]]); // Top edge
        assert_eq!(padded[[1, 0]], padded[[1, 1]]); // Left edge
    }

    #[test]
    fn test_nlm_denoise_constant() {
        // Test on constant image (should remain unchanged)
        let input = Array2::ones((10, 10));
        let result = nlm_denoise(&input, 0.1, 3, 5).unwrap();

        // Check that constant regions remain constant
        for val in result.iter() {
            assert!((val - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_nlm_denoise_dimensions() {
        let input = Array2::zeros((20, 30));
        let result = nlm_denoise(&input, 0.1, 7, 21).unwrap();

        assert_eq!(result.dim(), (20, 30));
    }

    #[test]
    fn test_nlm_denoise_color() {
        let input = Array3::ones((10, 10, 3));
        let result = nlm_denoise_color(&input, 0.1, 3, 5).unwrap();

        assert_eq!(result.dim(), (10, 10, 3));
    }

    #[test]
    fn test_nlm_denoise_parallel() {
        let input = Array2::from_shape_fn((20, 20), |(i, j)| ((i + j) % 2) as f32);

        let serial = nlm_denoise(&input, 0.1, 3, 7).unwrap();
        let parallel = nlm_denoise_parallel(&input, 0.1, 3, 7).unwrap();

        // Results should be very similar
        for (s, p) in serial.iter().zip(parallel.iter()) {
            assert!((s - p).abs() < 1e-5);
        }
    }

    #[test]
    fn test_invalid_color_channels() {
        let input = Array3::zeros((10, 10, 4)); // 4 channels instead of 3
        let result = nlm_denoise_color(&input, 0.1, 3, 5);

        assert!(result.is_err());
    }
}
