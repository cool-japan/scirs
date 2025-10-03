//! Retinal Processing with Center-Surround
//!
//! This module implements biological retinal processing including center-surround
//! receptive fields, temporal dynamics, and edge enhancement.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};

use super::config::{BiologicalVisionConfig, RetinaModel};
use crate::error::{NdimageError, NdimageResult};

/// Retinal Processing with Center-Surround
///
/// Implements biological retinal processing including center-surround
/// receptive fields, temporal dynamics, and edge enhancement.
pub fn retinal_processing<T>(
    image_sequence: &[ArrayView2<T>],
    config: &BiologicalVisionConfig,
) -> NdimageResult<RetinaModel>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    if image_sequence.is_empty() {
        return Err(NdimageError::InvalidInput(
            "Empty image sequence".to_string(),
        ));
    }

    let (height, width) = image_sequence[0].dim();
    let mut retina = RetinaModel {
        photoreceptors: Array2::zeros((height, width)),
        bipolar_cells: Array2::zeros((height, width)),
        horizontal_cells: Array2::zeros((height, width)),
        ganglion_cells: Array2::zeros((height, width)),
        center_surround_filters: create_center_surround_filters()?,
    };

    // Process temporal sequence
    for (t, image) in image_sequence.iter().enumerate() {
        // Photoreceptor adaptation
        update_photoreceptors(&mut retina.photoreceptors, image, t, config)?;

        // Horizontal cell lateral inhibition
        update_horizontal_cells(&mut retina.horizontal_cells, &retina.photoreceptors, config)?;

        // Bipolar cell center-surround processing
        update_bipolar_cells(
            &mut retina.bipolar_cells,
            &retina.photoreceptors,
            &retina.horizontal_cells,
            &retina.center_surround_filters,
        )?;

        // Ganglion cell edge detection
        update_ganglion_cells(&mut retina.ganglion_cells, &retina.bipolar_cells, config)?;
    }

    Ok(retina)
}

/// Create center-surround filters for retinal processing
pub fn create_center_surround_filters() -> NdimageResult<Vec<Array2<f64>>> {
    let mut filters = Vec::new();

    // Create ON-center filter
    let on_center = Array2::from_shape_fn((5, 5), |(y, x)| {
        let dy = y as f64 - 2.0;
        let dx = x as f64 - 2.0;
        let distance = (dy * dy + dx * dx).sqrt();

        if distance <= 1.0 {
            1.0
        } else if distance <= 2.0 {
            -0.5
        } else {
            0.0
        }
    });

    // Create OFF-center filter
    let off_center = Array2::from_shape_fn((5, 5), |(y, x)| {
        let dy = y as f64 - 2.0;
        let dx = x as f64 - 2.0;
        let distance = (dy * dy + dx * dx).sqrt();

        if distance <= 1.0 {
            -1.0
        } else if distance <= 2.0 {
            0.5
        } else {
            0.0
        }
    });

    filters.push(on_center);
    filters.push(off_center);

    Ok(filters)
}

/// Update photoreceptor responses with adaptation
pub fn update_photoreceptors<T>(
    photoreceptors: &mut Array2<f64>,
    image: &ArrayView2<T>,
    time: usize,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = photoreceptors.dim();
    let adaptation_rate = 0.1;

    for y in 0..height {
        for x in 0..width {
            if y < image.nrows() && x < image.ncols() {
                let current_light = image[(y, x)].to_f64().unwrap_or(0.0);
                let previous_response = photoreceptors[(y, x)];

                // Adaptive response with temporal dynamics
                photoreceptors[(y, x)] =
                    previous_response * (1.0 - adaptation_rate) + current_light * adaptation_rate;
            }
        }
    }

    Ok(())
}

/// Update horizontal cell responses (lateral inhibition)
pub fn update_horizontal_cells(
    horizontal_cells: &mut Array2<f64>,
    photoreceptors: &Array2<f64>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let (height, width) = horizontal_cells.dim();

    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let mut lateral_sum = 0.0;
            let mut count = 0;

            // Average neighboring photoreceptor responses
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    lateral_sum += photoreceptors[(ny, nx)];
                    count += 1;
                }
            }

            horizontal_cells[(y, x)] = lateral_sum / count as f64;
        }
    }

    Ok(())
}

/// Update bipolar cell responses (center-surround processing)
pub fn update_bipolar_cells(
    bipolar_cells: &mut Array2<f64>,
    photoreceptors: &Array2<f64>,
    horizontal_cells: &Array2<f64>,
    center_surround_filters: &[Array2<f64>],
) -> NdimageResult<()> {
    let (height, width) = bipolar_cells.dim();

    for y in 0..height {
        for x in 0..width {
            // Center-surround processing
            let center_response = photoreceptors[(y, x)];
            let surround_response = horizontal_cells[(y, x)];

            // ON-center response
            bipolar_cells[(y, x)] = center_response - surround_response;
        }
    }

    Ok(())
}

/// Update ganglion cell responses (edge detection)
pub fn update_ganglion_cells(
    ganglion_cells: &mut Array2<f64>,
    bipolar_cells: &Array2<f64>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let (height, width) = ganglion_cells.dim();

    // Simple edge detection for ganglion cells
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let horizontal_gradient = bipolar_cells[(y, x + 1)] - bipolar_cells[(y, x - 1)];
            let vertical_gradient = bipolar_cells[(y + 1, x)] - bipolar_cells[(y - 1, x)];

            ganglion_cells[(y, x)] = (horizontal_gradient * horizontal_gradient
                + vertical_gradient * vertical_gradient)
                .sqrt();
        }
    }

    Ok(())
}
