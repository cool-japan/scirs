//! Bio-Inspired Color Constancy
//!
//! This module implements color constancy mechanisms inspired by human color perception
//! for robust color processing under varying illumination.

use scirs2_core::ndarray::{Array2, Array3};
use scirs2_core::numeric::{Float, FromPrimitive};

use super::config::{BiologicalVisionConfig, ColorConstancySystem};
use crate::error::{NdimageError, NdimageResult};

/// Bio-Inspired Color Constancy
///
/// Implements color constancy mechanisms inspired by human color perception
/// for robust color processing under varying illumination.
pub fn bio_inspired_color_constancy<T>(
    colorimage_sequence: &[Array3<T>],
    config: &BiologicalVisionConfig,
) -> NdimageResult<ColorConstancySystem>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    if colorimage_sequence.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Empty color sequence".to_string(),
        ));
    }

    let (height, width, channels) = colorimage_sequence[0].dim();
    if channels != 3 {
        return Err(NdimageError::InvalidInput(
            "Expected RGB images".to_string(),
        ));
    }

    let mut color_system = ColorConstancySystem {
        illumination_estimates: Array2::from_elem((height, width), (1.0, 1.0, 1.0)),
        surface_reflectance: Array2::from_elem((height, width), (0.5, 0.5, 0.5)),
        adaptationstate: (1.0, 1.0, 1.0),
        color_memory: Vec::new(),
    };

    // Process color sequence
    for colorimage in colorimage_sequence {
        // Estimate illumination using biological algorithms
        estimate_illumination(&mut color_system, colorimage, config)?;

        // Adapt to illumination changes
        adapt_to_illumination(&mut color_system, config)?;

        // Compute surface reflectance
        compute_surface_reflectance(&mut color_system, colorimage)?;

        // Update color memory
        update_color_memory(&mut color_system, colorimage, config)?;
    }

    Ok(color_system)
}

/// Estimate illumination using biological algorithms
pub fn estimate_illumination<T>(
    color_system: &mut ColorConstancySystem,
    image: &Array3<T>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width, _) = image.dim();

    // Gray-world assumption with local adaptation
    for y in 0..height {
        for x in 0..width {
            // Extract RGB values
            let r = image[(y, x, 0)].to_f64().unwrap_or(0.0);
            let g = image[(y, x, 1)].to_f64().unwrap_or(0.0);
            let b = image[(y, x, 2)].to_f64().unwrap_or(0.0);

            // Local neighborhood analysis for illumination estimation
            let local_illumination = estimate_local_illumination(image, (y, x), 5)?;

            // Combine global and local estimates
            let current_estimate = color_system.illumination_estimates[(y, x)];
            let new_estimate = (
                current_estimate.0 * 0.9 + local_illumination.0 * 0.1,
                current_estimate.1 * 0.9 + local_illumination.1 * 0.1,
                current_estimate.2 * 0.9 + local_illumination.2 * 0.1,
            );

            color_system.illumination_estimates[(y, x)] = new_estimate;
        }
    }

    Ok(())
}

/// Estimate local illumination for a specific pixel location
fn estimate_local_illumination<T>(
    image: &Array3<T>,
    position: (usize, usize),
    window_size: usize,
) -> NdimageResult<(f64, f64, f64)>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width, _) = image.dim();
    let (center_y, center_x) = position;
    let half_window = window_size / 2;

    let mut sum_r = 0.0;
    let mut sum_g = 0.0;
    let mut sum_b = 0.0;
    let mut count = 0;

    for dy in 0..window_size {
        for dx in 0..window_size {
            let y_offset = dy as isize - half_window as isize;
            let x_offset = dx as isize - half_window as isize;
            let y = (center_y as isize + y_offset) as usize;
            let x = (center_x as isize + x_offset) as usize;

            if center_y as isize + y_offset >= 0
                && center_x as isize + x_offset >= 0
                && y < height
                && x < width
            {
                let r = image[(y, x, 0)].to_f64().unwrap_or(0.0);
                let g = image[(y, x, 1)].to_f64().unwrap_or(0.0);
                let b = image[(y, x, 2)].to_f64().unwrap_or(0.0);

                sum_r += r;
                sum_g += g;
                sum_b += b;
                count += 1;
            }
        }
    }

    if count > 0 {
        Ok((
            sum_r / count as f64,
            sum_g / count as f64,
            sum_b / count as f64,
        ))
    } else {
        Ok((0.5, 0.5, 0.5)) // Default neutral illumination
    }
}

/// Adapt to illumination changes using biological mechanisms
pub fn adapt_to_illumination(
    color_system: &mut ColorConstancySystem,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let (height, width) = color_system.illumination_estimates.dim();

    // Compute global adaptation state from local estimates
    let mut global_r = 0.0;
    let mut global_g = 0.0;
    let mut global_b = 0.0;
    let mut count = 0;

    for y in 0..height {
        for x in 0..width {
            let estimate = color_system.illumination_estimates[(y, x)];
            global_r += estimate.0;
            global_g += estimate.1;
            global_b += estimate.2;
            count += 1;
        }
    }

    if count > 0 {
        let avg_illumination = (
            global_r / count as f64,
            global_g / count as f64,
            global_b / count as f64,
        );

        // Adaptive update of adaptation state
        let adaptation_rate = config.color_adaptation_rate;
        let current_state = color_system.adaptationstate;

        color_system.adaptationstate = (
            current_state.0 * (1.0 - adaptation_rate) + avg_illumination.0 * adaptation_rate,
            current_state.1 * (1.0 - adaptation_rate) + avg_illumination.1 * adaptation_rate,
            current_state.2 * (1.0 - adaptation_rate) + avg_illumination.2 * adaptation_rate,
        );
    }

    Ok(())
}

/// Compute surface reflectance from image and illumination estimates
pub fn compute_surface_reflectance<T>(
    color_system: &mut ColorConstancySystem,
    image: &Array3<T>,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width, _) = image.dim();

    for y in 0..height {
        for x in 0..width {
            // Extract RGB values
            let r = image[(y, x, 0)].to_f64().unwrap_or(0.0);
            let g = image[(y, x, 1)].to_f64().unwrap_or(0.0);
            let b = image[(y, x, 2)].to_f64().unwrap_or(0.0);

            // Get illumination estimate
            let illumination = color_system.illumination_estimates[(y, x)];

            // Compute surface reflectance by dividing observed color by illumination
            let reflectance = (
                if illumination.0 > 1e-6 {
                    r / illumination.0
                } else {
                    r
                },
                if illumination.1 > 1e-6 {
                    g / illumination.1
                } else {
                    g
                },
                if illumination.2 > 1e-6 {
                    b / illumination.2
                } else {
                    b
                },
            );

            // Clamp reflectance to reasonable bounds
            let clamped_reflectance = (
                reflectance.0.max(0.0).min(1.0),
                reflectance.1.max(0.0).min(1.0),
                reflectance.2.max(0.0).min(1.0),
            );

            color_system.surface_reflectance[(y, x)] = clamped_reflectance;
        }
    }

    Ok(())
}

/// Update color memory with significant color observations
pub fn update_color_memory<T>(
    color_system: &mut ColorConstancySystem,
    image: &Array3<T>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width, _) = image.dim();

    // Sample representative colors from the image
    let sample_stride = (height * width / 100).max(1); // Sample ~100 colors
    let mut sampled_colors = Vec::new();

    for idx in (0..height * width).step_by(sample_stride) {
        let y = idx / width;
        let x = idx % width;

        if y < height && x < width {
            let color = (
                image[(y, x, 0)].to_f64().unwrap_or(0.0),
                image[(y, x, 1)].to_f64().unwrap_or(0.0),
                image[(y, x, 2)].to_f64().unwrap_or(0.0),
            );

            // Only add colors that are sufficiently different from existing memory
            let is_novel = color_system.color_memory.iter().all(|&existing_color| {
                let color_distance = ((color.0 - existing_color.0).powi(2)
                    + (color.1 - existing_color.1).powi(2)
                    + (color.2 - existing_color.2).powi(2))
                .sqrt();
                color_distance > 0.1 // Threshold for color novelty
            });

            if is_novel {
                sampled_colors.push(color);
            }
        }
    }

    // Add novel colors to memory
    for color in sampled_colors {
        color_system.color_memory.push(color);

        // Limit memory size
        if color_system.color_memory.len() > 1000 {
            color_system.color_memory.remove(0); // Remove oldest color
        }
    }

    Ok(())
}
