//! Hierarchical Cortical Processing
//!
//! This module implements hierarchical processing inspired by the mammalian visual cortex.
//! Features predictive coding, lateral inhibition, and multi-scale analysis.

use scirs2_core::ndarray::{Array3, ArrayView2, Axis};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::f64::consts::PI;

use super::config::{BiologicalVisionConfig, CorticalLayer};
use crate::error::{NdimageError, NdimageResult};

/// Hierarchical Cortical Processing
///
/// Implements hierarchical processing inspired by the mammalian visual cortex.
/// Features predictive coding, lateral inhibition, and multi-scale analysis.
pub fn hierarchical_cortical_processing<T>(
    image: ArrayView2<T>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<Vec<CorticalLayer>>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    let (height, width) = image.dim();
    let mut cortical_layers = Vec::new();

    // Initialize cortical hierarchy
    for level in 0..config.cortical_layers {
        let rf_size = config.receptive_field_sizes.get(level).unwrap_or(&7);
        let num_features = 2_usize.pow(level as u32 + 4); // Increasing feature complexity

        let layer = CorticalLayer {
            level,
            feature_maps: Array3::zeros((num_features, height / (level + 1), width / (level + 1))),
            receptive_field_size: *rf_size,
            lateral_connections: scirs2_core::ndarray::Array2::zeros((num_features, num_features)),
            top_down_predictions: Array3::zeros((
                num_features,
                height / (level + 1),
                width / (level + 1),
            )),
            bottom_upfeatures: Array3::zeros((
                num_features,
                height / (level + 1),
                width / (level + 1),
            )),
            prediction_errors: Array3::zeros((
                num_features,
                height / (level + 1),
                width / (level + 1),
            )),
        };

        cortical_layers.push(layer);
    }

    // Initialize with V1-like processing (first layer)
    initialize_v1_processing(&mut cortical_layers[0], &image, config)?;

    // Forward pass through hierarchy
    for level in 1..config.cortical_layers {
        let (lower, upper) = cortical_layers.split_at_mut(level);
        forward_pass_cortical_layer(&mut upper[0], &lower[level - 1], config)?;
    }

    // Backward pass with predictions
    for level in (0..config.cortical_layers - 1).rev() {
        let (lower, upper) = cortical_layers.split_at_mut(level + 1);
        backward_pass_cortical_layer(&mut lower[level], &upper[0], config)?;
    }

    // Apply lateral inhibition
    for layer in &mut cortical_layers {
        apply_lateral_inhibition(layer, config)?;
    }

    Ok(cortical_layers)
}

/// Initialize V1-like processing for the first cortical layer
pub fn initialize_v1_processing<T>(
    layer: &mut CorticalLayer,
    image: &ArrayView2<T>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = image.dim();

    // Simplified V1-like edge detection filters
    for feature_idx in 0..layer.feature_maps.len_of(Axis(0)) {
        for y in 0..layer.feature_maps.len_of(Axis(1)) {
            for x in 0..layer.feature_maps.len_of(Axis(2)) {
                // Scale coordinates to original image
                let orig_y = y * height / layer.feature_maps.len_of(Axis(1));
                let orig_x = x * width / layer.feature_maps.len_of(Axis(2));

                if orig_y < height && orig_x < width {
                    let pixel_value = image[(orig_y, orig_x)].to_f64().unwrap_or(0.0);

                    // Simple orientation-selective response
                    let orientation =
                        feature_idx as f64 * PI / layer.feature_maps.len_of(Axis(0)) as f64;
                    let response = pixel_value * orientation.cos();

                    layer.bottom_upfeatures[(feature_idx, y, x)] = response;
                    layer.feature_maps[(feature_idx, y, x)] = response;
                }
            }
        }
    }

    Ok(())
}

/// Forward pass through cortical layer
pub fn forward_pass_cortical_layer(
    current_layer: &mut CorticalLayer,
    previous_layer: &CorticalLayer,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    // Simplified forward pass - pool and transform features from previous layer
    let scale_factor =
        previous_layer.feature_maps.len_of(Axis(1)) / current_layer.feature_maps.len_of(Axis(1));

    for feature_idx in 0..current_layer.feature_maps.len_of(Axis(0)) {
        for y in 0..current_layer.feature_maps.len_of(Axis(1)) {
            for x in 0..current_layer.feature_maps.len_of(Axis(2)) {
                let mut pooled_response = 0.0;
                let mut count = 0;

                // Pool from previous layer
                for dy in 0..scale_factor {
                    for dx in 0..scale_factor {
                        let prev_y = y * scale_factor + dy;
                        let prev_x = x * scale_factor + dx;

                        if prev_y < previous_layer.feature_maps.len_of(Axis(1))
                            && prev_x < previous_layer.feature_maps.len_of(Axis(2))
                        {
                            // Combine features from previous layer
                            for prev_feature_idx in 0..previous_layer.feature_maps.len_of(Axis(0)) {
                                pooled_response +=
                                    previous_layer.feature_maps[(prev_feature_idx, prev_y, prev_x)];
                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    current_layer.bottom_upfeatures[(feature_idx, y, x)] =
                        pooled_response / count as f64;
                    current_layer.feature_maps[(feature_idx, y, x)] =
                        pooled_response / count as f64;
                }
            }
        }
    }

    Ok(())
}

/// Backward pass for cortical layer (predictive coding)
pub fn backward_pass_cortical_layer(
    current_layer: &mut CorticalLayer,
    next_layer: &CorticalLayer,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    // Simplified backward pass - generate predictions from higher layer
    let scale_factor =
        current_layer.feature_maps.len_of(Axis(1)) / next_layer.feature_maps.len_of(Axis(1));

    for feature_idx in 0..current_layer.feature_maps.len_of(Axis(0)) {
        for y in 0..current_layer.feature_maps.len_of(Axis(1)) {
            for x in 0..current_layer.feature_maps.len_of(Axis(2)) {
                let next_y = y / scale_factor;
                let next_x = x / scale_factor;

                if next_y < next_layer.feature_maps.len_of(Axis(1))
                    && next_x < next_layer.feature_maps.len_of(Axis(2))
                {
                    // Generate prediction from higher layer
                    let mut prediction = 0.0;
                    for next_feature_idx in 0..next_layer.feature_maps.len_of(Axis(0)) {
                        prediction += next_layer.feature_maps[(next_feature_idx, next_y, next_x)];
                    }

                    current_layer.top_down_predictions[(feature_idx, y, x)] =
                        prediction / next_layer.feature_maps.len_of(Axis(0)) as f64;

                    // Compute prediction error
                    let error = current_layer.bottom_upfeatures[(feature_idx, y, x)]
                        - current_layer.top_down_predictions[(feature_idx, y, x)];
                    current_layer.prediction_errors[(feature_idx, y, x)] = error;
                }
            }
        }
    }

    Ok(())
}

/// Apply lateral inhibition to cortical layer
pub fn apply_lateral_inhibition(
    layer: &mut CorticalLayer,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let num_features = layer.feature_maps.len_of(Axis(0));
    let height = layer.feature_maps.len_of(Axis(1));
    let width = layer.feature_maps.len_of(Axis(2));

    let mut inhibitedfeatures = layer.feature_maps.clone();

    for feature_idx in 0..num_features {
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_response = layer.feature_maps[(feature_idx, y, x)];

                // Compute lateral inhibition from neighbors
                let mut inhibition = 0.0;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dy != 0 || dx != 0 {
                            let ny = (y as i32 + dy) as usize;
                            let nx = (x as i32 + dx) as usize;
                            inhibition += layer.feature_maps[(feature_idx, ny, nx)];
                        }
                    }
                }

                // Apply inhibition
                let inhibited_response =
                    center_response - config.lateral_inhibition_strength * inhibition / 8.0;
                inhibitedfeatures[(feature_idx, y, x)] = inhibited_response.max(0.0);
            }
        }
    }

    layer.feature_maps = inhibitedfeatures;
    Ok(())
}
