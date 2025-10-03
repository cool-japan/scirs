//! Bio-Inspired Attention and Saccade Planning
//!
//! This module implements attention mechanisms and saccade planning inspired by
//! primate visual systems for efficient scene exploration.

use scirs2_core::ndarray::{Array2, Array3, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::collections::HashMap;

use super::config::{AttentionSystem, BiologicalVisionConfig};
use crate::error::{NdimageError, NdimageResult};

/// Bio-Inspired Attention and Saccade Planning
///
/// Implements attention mechanisms and saccade planning inspired by
/// primate visual systems for efficient scene exploration.
pub fn bio_inspired_attention_saccades<T>(
    image: ArrayView2<T>,
    feature_maps: &[Array3<f64>],
    config: &BiologicalVisionConfig,
) -> NdimageResult<AttentionSystem>
where
    T: Float + FromPrimitive + Copy + Send + Sync,
{
    let (height, width) = image.dim();

    // Initialize attention system
    let mut attention_system = AttentionSystem {
        attention_center: (height / 2, width / 2),
        attention_map: Array2::zeros((height, width)),
        saccade_targets: Vec::new(),
        inhibition_of_return: Array2::zeros((height, width)),
        feature_attention_weights: HashMap::new(),
    };

    // Compute bottom-up attention (salience)
    compute_bottom_up_attention(&mut attention_system.attention_map, &image, config)?;

    // Incorporate feature-based attention
    for (feature_idx, feature_map) in feature_maps.iter().enumerate() {
        let feature_name = format!("feature_{}", feature_idx);
        let feature_weight = 1.0 / (feature_idx + 1) as f64; // Decreasing weight with complexity

        attention_system
            .feature_attention_weights
            .insert(feature_name, feature_weight);
        add_feature_based_attention(
            &mut attention_system.attention_map,
            feature_map,
            feature_weight,
        )?;
    }

    // Apply inhibition of return
    apply_inhibition_of_return(&mut attention_system, config)?;

    // Plan saccade sequence
    plan_saccade_sequence(&mut attention_system, config)?;

    Ok(attention_system)
}

/// Compute bottom-up attention based on local contrast and saliency
pub fn compute_bottom_up_attention<T>(
    attention_map: &mut Array2<f64>,
    image: &ArrayView2<T>,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()>
where
    T: Float + FromPrimitive + Copy,
{
    let (height, width) = attention_map.dim();

    // Simple saliency based on local contrast
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            if y < image.nrows() && x < image.ncols() {
                let center = image[(y, x)].to_f64().unwrap_or(0.0);
                let mut contrast = 0.0;
                let mut count = 0;

                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dy != 0 || dx != 0 {
                            let ny = (y as i32 + dy) as usize;
                            let nx = (x as i32 + dx) as usize;
                            if ny < image.nrows() && nx < image.ncols() {
                                let neighbor = image[(ny, nx)].to_f64().unwrap_or(0.0);
                                contrast += (center - neighbor).abs();
                                count += 1;
                            }
                        }
                    }
                }

                attention_map[(y, x)] = if count > 0 {
                    contrast / count as f64
                } else {
                    0.0
                };
            }
        }
    }

    Ok(())
}

/// Add feature-based attention to the attention map
pub fn add_feature_based_attention(
    attention_map: &mut Array2<f64>,
    feature_map: &Array3<f64>,
    weight: f64,
) -> NdimageResult<()> {
    let (height, width) = attention_map.dim();
    let (num_features, feat_height, feat_width) = feature_map.dim();

    // Combine all feature channels
    for y in 0..height {
        for x in 0..width {
            let feat_y = y * feat_height / height;
            let feat_x = x * feat_width / width;

            if feat_y < feat_height && feat_x < feat_width {
                let mut feature_response = 0.0;

                // Aggregate across all feature channels
                for feat_idx in 0..num_features {
                    feature_response += feature_map[(feat_idx, feat_y, feat_x)].abs();
                }

                attention_map[(y, x)] += weight * feature_response / num_features as f64;
            }
        }
    }

    Ok(())
}

/// Apply inhibition of return to prevent immediate return to previously attended locations
pub fn apply_inhibition_of_return(
    attention_system: &mut AttentionSystem,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let (height, width) = attention_system.attention_map.dim();
    let (center_y, center_x) = attention_system.attention_center;

    // Create inhibition around current attention center
    let inhibition_radius = config.attention_radius;

    for y in 0..height {
        for x in 0..width {
            let distance =
                ((y as i32 - center_y as i32).pow(2) + (x as i32 - center_x as i32).pow(2)) as f64;
            let distance = distance.sqrt();

            if distance < inhibition_radius as f64 {
                let inhibition_strength = (1.0 - distance / inhibition_radius as f64) * 0.8;
                attention_system.inhibition_of_return[(y, x)] = inhibition_strength;
                attention_system.attention_map[(y, x)] *= (1.0 - inhibition_strength);
            }
        }
    }

    Ok(())
}

/// Plan sequence of saccade targets based on attention map
pub fn plan_saccade_sequence(
    attention_system: &mut AttentionSystem,
    config: &BiologicalVisionConfig,
) -> NdimageResult<()> {
    let (height, width) = attention_system.attention_map.dim();
    attention_system.saccade_targets.clear();

    // Find peaks in attention map for saccade planning
    for _ in 0..config.saccade_horizon {
        let mut max_attention = 0.0;
        let mut best_target = (height / 2, width / 2);

        // Find maximum attention location
        for y in 0..height {
            for x in 0..width {
                if attention_system.attention_map[(y, x)] > max_attention {
                    max_attention = attention_system.attention_map[(y, x)];
                    best_target = (y, x);
                }
            }
        }

        if max_attention > 0.1 {
            // Threshold for saccade generation
            attention_system.saccade_targets.push(best_target);

            // Suppress attention around this target for next iteration
            let suppress_radius = config.attention_radius / 2;
            for y in 0..height {
                for x in 0..width {
                    let distance = ((y as i32 - best_target.0 as i32).pow(2)
                        + (x as i32 - best_target.1 as i32).pow(2))
                        as f64;
                    let distance = distance.sqrt();

                    if distance < suppress_radius as f64 {
                        let suppression = 1.0 - distance / suppress_radius as f64;
                        attention_system.attention_map[(y, x)] *= (1.0 - suppression * 0.9);
                    }
                }
            }
        } else {
            break; // No more significant targets
        }
    }

    Ok(())
}
