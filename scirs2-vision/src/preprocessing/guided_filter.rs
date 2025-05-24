//! Guided filtering for edge-preserving smoothing
//!
//! The guided filter is a linear time edge-preserving smoothing filter.
//! It has many applications including noise reduction, detail smoothing,
//! HDR compression, image matting/feathering, haze removal, and joint upsampling.

use crate::error::Result;
use ndarray::{s, Array2, Array3};

/// Apply guided filter to a grayscale image
///
/// The guided filter uses a guidance image to perform edge-preserving smoothing.
/// When the guidance image is the same as the input, it performs edge-preserving smoothing.
///
/// # Arguments
///
/// * `input` - Input image to be filtered
/// * `guide` - Guidance image (can be the same as input)
/// * `radius` - Radius of the filter window
/// * `epsilon` - Regularization parameter controlling the degree of smoothing
///
/// # Returns
///
/// * Result containing the filtered image
pub fn guided_filter(
    input: &Array2<f32>,
    guide: &Array2<f32>,
    radius: usize,
    epsilon: f32,
) -> Result<Array2<f32>> {
    let (height, width) = input.dim();
    let (g_height, g_width) = guide.dim();

    // Check dimensions match
    if height != g_height || width != g_width {
        return Err(crate::error::VisionError::InvalidParameter(
            "Input and guide images must have the same dimensions".to_string(),
        ));
    }

    // Compute mean values using box filter
    let mean_i = box_filter(input, radius);
    let mean_g = box_filter(guide, radius);

    // Compute correlation and variance
    let corr_ig = box_filter(&(input * guide), radius);
    let corr_gg = box_filter(&(guide * guide), radius);

    // Variance of guide: var_g = corr_gg - mean_g * mean_g
    let var_g = &corr_gg - &mean_g * &mean_g;

    // Covariance of (guide, input): cov_ig = corr_ig - mean_g * mean_i
    let cov_ig = &corr_ig - &mean_g * &mean_i;

    // Coefficients: a = cov_ig / (var_g + epsilon), b = mean_i - a * mean_g
    let mut a = Array2::zeros((height, width));
    for i in 0..height {
        for j in 0..width {
            a[[i, j]] = cov_ig[[i, j]] / (var_g[[i, j]] + epsilon);
        }
    }

    let b = &mean_i - &a * &mean_g;

    // Compute mean coefficients
    let mean_a = box_filter(&a, radius);
    let mean_b = box_filter(&b, radius);

    // Output: q = mean_a * guide + mean_b
    Ok(&mean_a * guide + &mean_b)
}

/// Apply guided filter to a color image
///
/// # Arguments
///
/// * `input` - Input color image to be filtered (HxWx3)
/// * `guide` - Guidance color image (can be the same as input) (HxWx3)
/// * `radius` - Radius of the filter window
/// * `epsilon` - Regularization parameter controlling the degree of smoothing
///
/// # Returns
///
/// * Result containing the filtered color image
pub fn guided_filter_color(
    input: &Array3<f32>,
    guide: &Array3<f32>,
    radius: usize,
    epsilon: f32,
) -> Result<Array3<f32>> {
    let (height, width, channels) = input.dim();
    let (g_height, g_width, g_channels) = guide.dim();

    // Check dimensions
    if height != g_height || width != g_width || channels != 3 || g_channels != 3 {
        return Err(crate::error::VisionError::InvalidParameter(
            "Input and guide must be HxWx3 color images with matching dimensions".to_string(),
        ));
    }

    let mut output = Array3::zeros((height, width, 3));

    // Process each channel independently using the guide
    for c in 0..3 {
        let input_channel = input.slice(s![.., .., c]);
        let guide_channel = guide.slice(s![.., .., c]);
        let filtered = guided_filter(
            &input_channel.to_owned(),
            &guide_channel.to_owned(),
            radius,
            epsilon,
        )?;
        output.slice_mut(s![.., .., c]).assign(&filtered);
    }

    Ok(output)
}

/// Box filter (mean filter) implementation
fn box_filter(input: &Array2<f32>, radius: usize) -> Array2<f32> {
    let (height, width) = input.dim();
    let mut output = Array2::zeros((height, width));

    // Simple box filter implementation
    for y in 0..height {
        for x in 0..width {
            let mut sum = 0.0;
            let mut count = 0;

            // Define window boundaries
            let y_start = y.saturating_sub(radius);
            let y_end = (y + radius + 1).min(height);
            let x_start = x.saturating_sub(radius);
            let x_end = (x + radius + 1).min(width);

            // Sum values in window
            for wy in y_start..y_end {
                for wx in x_start..x_end {
                    sum += input[[wy, wx]];
                    count += 1;
                }
            }

            output[[y, x]] = sum / count as f32;
        }
    }

    output
}

/// Fast guided filter using subsampling
///
/// This is a faster version that subsamples the image before filtering
/// and then upsamples the result.
///
/// # Arguments
///
/// * `input` - Input image to be filtered
/// * `guide` - Guidance image (can be the same as input)
/// * `radius` - Radius of the filter window
/// * `epsilon` - Regularization parameter
/// * `subsample` - Subsampling factor (e.g., 2 or 4)
///
/// # Returns
///
/// * Result containing the filtered image
pub fn fast_guided_filter(
    input: &Array2<f32>,
    guide: &Array2<f32>,
    radius: usize,
    epsilon: f32,
    subsample: usize,
) -> Result<Array2<f32>> {
    if subsample == 1 {
        return guided_filter(input, guide, radius, epsilon);
    }

    let (height, width) = input.dim();

    // Subsample input and guide
    let sub_height = height.div_ceil(subsample);
    let sub_width = width.div_ceil(subsample);

    let mut sub_input = Array2::zeros((sub_height, sub_width));
    let mut sub_guide = Array2::zeros((sub_height, sub_width));

    for y in 0..sub_height {
        for x in 0..sub_width {
            let orig_y = (y * subsample).min(height - 1);
            let orig_x = (x * subsample).min(width - 1);
            sub_input[[y, x]] = input[[orig_y, orig_x]];
            sub_guide[[y, x]] = guide[[orig_y, orig_x]];
        }
    }

    // Apply guided filter at lower resolution
    let sub_radius = radius.div_ceil(subsample);
    let sub_filtered = guided_filter(&sub_input, &sub_guide, sub_radius, epsilon)?;

    // Upsample result with bilinear interpolation
    let mut output = Array2::zeros((height, width));
    for y in 0..height {
        for x in 0..width {
            let sy = y as f32 / subsample as f32;
            let sx = x as f32 / subsample as f32;

            let y0 = sy.floor() as usize;
            let x0 = sx.floor() as usize;
            let y1 = (y0 + 1).min(sub_height - 1);
            let x1 = (x0 + 1).min(sub_width - 1);

            let fy = sy - y0 as f32;
            let fx = sx - x0 as f32;

            // Bilinear interpolation
            let v00 = sub_filtered[[y0, x0]];
            let v01 = sub_filtered[[y0, x1]];
            let v10 = sub_filtered[[y1, x0]];
            let v11 = sub_filtered[[y1, x1]];

            output[[y, x]] =
                (1.0 - fy) * ((1.0 - fx) * v00 + fx * v01) + fy * ((1.0 - fx) * v10 + fx * v11);
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_guided_filter_identity() {
        // When epsilon is very small and guide=input, output should be close to input
        let input = Array2::from_shape_vec(
            (5, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0, 4.0,
                5.0, 6.0, 7.0, 8.0, 5.0, 6.0, 7.0, 8.0, 9.0,
            ],
        )
        .unwrap();

        let result = guided_filter(&input, &input, 1, 0.001).unwrap();

        // With small epsilon and guide=input, the filter should preserve most structure
        // Check that the result is similar to input (allowing for boundary effects)
        for y in 1..4 {
            for x in 1..4 {
                let diff = (result[[y, x]] - input[[y, x]]).abs();
                assert!(diff < 2.0, "Difference at ({}, {}): {}", x, y, diff);
            }
        }
    }

    #[test]
    fn test_guided_filter_edge_preserving() {
        // Create image with sharp edge
        let mut input = Array2::zeros((10, 10));
        input.slice_mut(s![.., 0..5]).fill(0.0);
        input.slice_mut(s![.., 5..]).fill(1.0);

        let result = guided_filter(&input, &input, 2, 0.1).unwrap();

        // Check that edge is somewhat preserved (middle values should show transition)
        assert!(result[[5, 4]] < 0.5);
        assert!(result[[5, 5]] > 0.5);
    }

    #[test]
    fn test_box_filter() {
        let input = Array2::ones((5, 5));
        let result = box_filter(&input, 1);

        // All values should remain 1.0 for constant input
        for val in result.iter() {
            assert!((val - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_guided_filter_color() {
        let input = Array3::ones((5, 5, 3));
        let result = guided_filter_color(&input, &input, 1, 0.1).unwrap();

        assert_eq!(result.dim(), (5, 5, 3));
    }

    #[test]
    fn test_fast_guided_filter() {
        let input =
            Array2::from_shape_vec((8, 8), (0..64).map(|i| i as f32 / 64.0).collect()).unwrap();

        let normal = guided_filter(&input, &input, 2, 0.1).unwrap();
        let fast = fast_guided_filter(&input, &input, 2, 0.1, 2).unwrap();

        // Fast version should produce similar results
        assert_eq!(normal.dim(), fast.dim());

        // Check that results are reasonably close (not exact due to subsampling)
        let diff = (&normal - &fast).mapv(|x| x.abs());
        let max_diff = diff.iter().fold(0.0f32, |a, &b| a.max(b));
        assert!(max_diff < 0.2); // Allow some difference due to subsampling
    }
}
