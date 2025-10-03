//! Compound Eye Motion Detection
//!
//! This module implements insect-inspired compound eye vision for ultra-wide field
//! motion detection and looming object detection.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::VecDeque;
use std::f64::consts::PI;

use super::config::{BiologicalVisionConfig, CompoundEyeModel, Ommatidium};
use crate::error::{NdimageError, NdimageResult};

/// Compound Eye Motion Detection
///
/// Implements insect-inspired compound eye vision for ultra-wide field
/// motion detection and looming object detection.
pub fn compound_eye_motion_detection<T>(
    image_sequence: &[ArrayView2<T>],
    config: &BiologicalVisionConfig,
) -> NdimageResult<CompoundEyeModel>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    if image_sequence.len() < 2 {
        return Err(NdimageError::InvalidInput(
            "Need at least 2 frames for motion detection".to_string(),
        ));
    }

    let (height, width) = image_sequence[0].dim();

    // Initialize compound eye structure
    let mut compound_eye = initialize_compound_eye(height, width, config)?;

    // Process temporal sequence for motion detection
    for window in image_sequence.windows(2) {
        let current_frame = window[0];
        let previous_frame = window[1];

        // Update ommatidial responses
        update_ommatidia_responses(&mut compound_eye, &current_frame, &previous_frame, config)?;

        // Compute motion detection
        compute_motion_detection(&mut compound_eye, config)?;

        // Detect looming objects
        detect_looming_objects(&mut compound_eye, config)?;

        // Wide-field integration
        update_wide_field_neurons(&mut compound_eye, config)?;
    }

    Ok(compound_eye)
}

/// Initialize compound eye structure with hexagonal ommatidial arrangement
pub fn initialize_compound_eye(
    height: usize,
    width: usize,
    config: &BiologicalVisionConfig,
) -> NdimageResult<CompoundEyeModel> {
    let mut ommatidia = Vec::new();

    // Create ommatidia in hexagonal pattern
    for i in 0..config.ommatidial_count {
        let angle = 2.0 * PI * i as f64 / config.ommatidial_count as f64;
        let radius = 0.3; // Normalized radius

        let ommatidium = Ommatidium {
            position: (radius * angle.cos(), radius * angle.sin()),
            optical_axis: (angle.cos(), angle.sin(), 0.0),
            response: 0.0,
            responsehistory: VecDeque::new(),
        };

        ommatidia.push(ommatidium);
    }

    Ok(CompoundEyeModel {
        ommatidia,
        motion_detectors: Array2::zeros((height / 10, width / 10)),
        wide_field_neurons: Array1::zeros(8), // 8 directional channels
        looming_detectors: Array1::zeros(config.ommatidial_count),
    })
}

/// Update ommatidial responses based on current and previous frames
pub fn update_ommatidia_responses<T>(
    compound_eye: &mut CompoundEyeModel,
    current_frame: &ArrayView2<T>,
    previous_frame: &ArrayView2<T>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = current_frame.dim();

    for ommatidium in &mut compound_eye.ommatidia {
        // Map ommatidium position to image coordinates
        let x = ((ommatidium.position.0 + 1.0) / 2.0 * width as f64) as usize;
        let y = ((ommatidium.position.1 + 1.0) / 2.0 * height as f64) as usize;

        if x < width && y < height {
            let current_value = current_frame[(y, x)].to_f64().unwrap_or(0.0);
            let previous_value = previous_frame[(y, x)].to_f64().unwrap_or(0.0);

            // Compute temporal difference (motion signal)
            let motion_signal = (current_value - previous_value).abs();
            ommatidium.response = motion_signal;

            // Update response history
            ommatidium.responsehistory.push_back(motion_signal);
            if ommatidium.responsehistory.len() > config.temporal_window {
                ommatidium.responsehistory.pop_front();
            }
        }
    }

    Ok(())
}

/// Compute motion detection using elementary motion detectors
pub fn compute_motion_detection(
    compound_eye: &mut CompoundEyeModel,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let (height, width) = compound_eye.motion_detectors.dim();

    // Simple motion detection based on ommatidial responses
    for y in 0..height {
        for x in 0..width {
            // Find nearby ommatidia
            let mut local_motion = 0.0;
            let mut count = 0;

            let center_x = x as f64 / width as f64 * 2.0 - 1.0;
            let center_y = y as f64 / height as f64 * 2.0 - 1.0;

            for ommatidium in &compound_eye.ommatidia {
                let dx = ommatidium.position.0 - center_x;
                let dy = ommatidium.position.1 - center_y;
                let distance = (dx * dx + dy * dy).sqrt();

                if distance < 0.2 {
                    // Within local neighborhood
                    local_motion += ommatidium.response;
                    count += 1;
                }
            }

            compound_eye.motion_detectors[(y, x)] = if count > 0 {
                local_motion / count as f64
            } else {
                0.0
            };
        }
    }

    Ok(())
}

/// Detect looming objects based on expanding motion patterns
pub fn detect_looming_objects(
    compound_eye: &mut CompoundEyeModel,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    for (i, ommatidium) in compound_eye.ommatidia.iter().enumerate() {
        // Detect expanding patterns around this ommatidium
        let mut expansion_signal = 0.0;

        if ommatidium.responsehistory.len() >= 2 {
            let recent_responses: Vec<f64> = ommatidium.responsehistory.iter().cloned().collect();

            // Simple expansion detection: increasing response over time
            for j in 1..recent_responses.len() {
                if recent_responses[j] > recent_responses[j - 1] {
                    expansion_signal += recent_responses[j] - recent_responses[j - 1];
                }
            }
        }

        compound_eye.looming_detectors[i] = expansion_signal;
    }

    Ok(())
}

/// Update wide-field integration neurons
pub fn update_wide_field_neurons(
    compound_eye: &mut CompoundEyeModel,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let num_directions = compound_eye.wide_field_neurons.len();

    // Integrate motion in different directions
    for dir_idx in 0..num_directions {
        let preferred_direction = dir_idx as f64 * 2.0 * PI / num_directions as f64;
        let mut directional_motion = 0.0;
        let mut count = 0;

        for ommatidium in &compound_eye.ommatidia {
            // Calculate direction from center to ommatidium
            let ommatidium_direction = ommatidium.position.1.atan2(ommatidium.position.0);

            // Check if aligned with preferred direction (within tolerance)
            let angle_diff = (ommatidium_direction - preferred_direction).abs();
            let normalized_diff =
                (angle_diff % (2.0 * PI)).min(2.0 * PI - (angle_diff % (2.0 * PI)));

            if normalized_diff < PI / 4.0 {
                // 45-degree tolerance
                directional_motion += ommatidium.response;
                count += 1;
            }
        }

        compound_eye.wide_field_neurons[dir_idx] = if count > 0 {
            directional_motion / count as f64
        } else {
            0.0
        };
    }

    Ok(())
}
