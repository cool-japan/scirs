//! Biological Vision System Inspired Algorithms
//!
//! This module has been refactored into focused components for better maintainability.
//! See the submodules for specific functionality.
//!
//! # Revolutionary Features
//!
//! - **Hierarchical Feature Processing**: Multi-scale cortical-like processing
//! - **Compound Eye Vision**: Ultra-wide field motion detection
//! - **Retinal Processing**: Biological-accurate retinal transformations
//! - **Attention and Saccades**: Bio-inspired attention mechanisms
//! - **Predictive Coding**: Brain-like prediction and error processing
//! - **Lateral Inhibition**: Contrast enhancement through biological mechanisms
//! - **Color Constancy**: Advanced color perception under varying illumination
//! - **Motion Prediction**: Biological motion prediction and tracking

// Re-export all module components for backward compatibility
pub use self::{
    advanced_processing::*, attention_saccades::*, color_constancy::*, compound_eye::*, config::*,
    cortical_processing::*, motion_tracking::*, predictive_coding::*, retinal_processing::*,
};

// Module declarations
pub mod advanced_processing;
pub mod attention_saccades;
pub mod color_constancy;
pub mod compound_eye;
pub mod config;
pub mod cortical_processing;
pub mod motion_tracking;
pub mod predictive_coding;
pub mod retinal_processing;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array2, Array3, Axis};

    #[test]
    fn test_biological_vision_config_default() {
        let config = BiologicalVisionConfig::default();

        assert_eq!(config.cortical_layers, 6);
        assert_eq!(config.receptive_field_sizes.len(), 6);
        assert_eq!(config.lateral_inhibition_strength, 0.5);
        assert_eq!(config.temporal_window, 10);
    }

    #[test]
    fn test_cortical_layer_creation() {
        let layer = CorticalLayer {
            level: 1,
            feature_maps: Array3::zeros((16, 64, 64)),
            receptive_field_size: 5,
            lateral_connections: Array2::zeros((16, 16)),
            top_down_predictions: Array3::zeros((16, 64, 64)),
            bottom_upfeatures: Array3::zeros((16, 64, 64)),
            prediction_errors: Array3::zeros((16, 64, 64)),
        };

        assert_eq!(layer.level, 1);
        assert_eq!(layer.feature_maps.dim(), (16, 64, 64));
        assert_eq!(layer.receptive_field_size, 5);
    }

    #[test]
    fn test_hierarchical_cortical_processing() {
        let image =
            Array2::from_shape_vec((32, 32), (0..1024).map(|x| x as f64 / 1024.0).collect())
                .unwrap();
        let config = BiologicalVisionConfig::default();

        let cortical_layers = hierarchical_cortical_processing(image.view(), &config).unwrap();

        assert_eq!(cortical_layers.len(), 6);
        assert!(cortical_layers[0].feature_maps.len_of(Axis(0)) > 0);
    }

    #[test]
    fn test_retinal_processing() {
        let image1 =
            Array2::from_shape_vec((16, 16), (0..256).map(|x| x as f64 / 256.0).collect()).unwrap();
        let image2 = Array2::from_shape_vec(
            (16, 16),
            (0..256).map(|x| (x + 50) as f64 / 256.0).collect(),
        )
        .unwrap();
        let images = vec![image1.view(), image2.view()];
        let config = BiologicalVisionConfig::default();

        let retina = retinal_processing(&images, &config).unwrap();

        assert_eq!(retina.photoreceptors.dim(), (16, 16));
        assert_eq!(retina.ganglion_cells.dim(), (16, 16));
        assert!(!retina.center_surround_filters.is_empty());
    }

    #[test]
    fn test_compound_eye_motion_detection() {
        let image1 =
            Array2::from_shape_vec((20, 20), (0..400).map(|x| x as f64 / 400.0).collect()).unwrap();
        let image2 = Array2::from_shape_vec(
            (20, 20),
            (0..400).map(|x| (x + 100) as f64 / 400.0).collect(),
        )
        .unwrap();
        let images = vec![image1.view(), image2.view()];
        let config = BiologicalVisionConfig::default();

        let compound_eye = compound_eye_motion_detection(&images, &config).unwrap();

        assert_eq!(compound_eye.ommatidia.len(), config.ommatidial_count);
        assert_eq!(compound_eye.wide_field_neurons.len(), 8);
        assert_eq!(
            compound_eye.looming_detectors.len(),
            config.ommatidial_count
        );
    }

    #[test]
    fn test_bio_inspired_attention_saccades() {
        let image =
            Array2::from_shape_vec((24, 24), (0..576).map(|x| x as f64 / 576.0).collect()).unwrap();
        let feature_maps = vec![Array3::zeros((8, 12, 12)), Array3::zeros((16, 6, 6))];
        let config = BiologicalVisionConfig::default();

        let attention_system =
            bio_inspired_attention_saccades(image.view(), &feature_maps, &config).unwrap();

        assert_eq!(attention_system.attention_map.dim(), (24, 24));
        assert_eq!(attention_system.inhibition_of_return.dim(), (24, 24));
        assert!(attention_system.saccade_targets.len() <= config.saccade_horizon);
    }

    #[test]
    fn test_predictive_coding_visual_processing() {
        let image1 =
            Array2::from_shape_vec((16, 16), (0..256).map(|x| x as f64 / 256.0).collect()).unwrap();
        let image2 = Array2::from_shape_vec(
            (16, 16),
            (0..256).map(|x| (x + 30) as f64 / 256.0).collect(),
        )
        .unwrap();
        let images = vec![image1.view(), image2.view()];
        let config = BiologicalVisionConfig::default();

        let predictive_system = predictive_coding_visual_processing(&images, &config).unwrap();

        assert_eq!(
            predictive_system.prediction_models.len(),
            config.cortical_layers
        );
        assert_eq!(
            predictive_system.prediction_errors.len(),
            config.cortical_layers
        );
        assert_eq!(
            predictive_system.confidence_estimates.len(),
            config.cortical_layers
        );
    }

    #[test]
    fn test_bio_inspired_color_constancy() {
        let color_image1 =
            Array3::from_shape_fn((12, 12, 3), |(y, x, c)| (y + x + c) as f64 / 50.0);
        let color_image2 =
            Array3::from_shape_fn((12, 12, 3), |(y, x, c)| (y + x + c + 10) as f64 / 50.0);
        let images = vec![color_image1, color_image2];
        let config = BiologicalVisionConfig::default();

        let color_system = bio_inspired_color_constancy(&images, &config).unwrap();

        assert_eq!(color_system.illumination_estimates.dim(), (12, 12));
        assert_eq!(color_system.surface_reflectance.dim(), (12, 12));
        assert!(!color_system.color_memory.is_empty());
    }

    #[test]
    fn test_bio_motion_prediction_tracking() {
        let images: Vec<Array2<f64>> = (0..10)
            .map(|i| {
                Array2::from_shape_vec(
                    (16, 16),
                    (0..256).map(|x| (x + i * 10) as f64 / 256.0).collect(),
                )
                .unwrap()
            })
            .collect();
        let image_views: Vec<_> = images.iter().map(|img| img.view()).collect();
        let initial_targets = vec![(8, 8), (4, 12)];
        let config = BiologicalVisionConfig::default();

        let motion_tracks =
            bio_motion_prediction_tracking(&image_views, &initial_targets, &config).unwrap();

        assert!(!motion_tracks.is_empty()); // At least one track should remain
        for track in &motion_tracks {
            assert!(!track.positionhistory.is_empty());
            assert!(!track.predicted_positions.is_empty());
            assert!(track.confidence >= 0.0 && track.confidence <= 1.0);
        }
    }

    #[test]
    fn test_advanced_retinal_circuits() {
        let image =
            Array2::from_shape_vec((20, 20), (0..400).map(|x| x as f64 / 400.0).collect()).unwrap();
        let config = BiologicalVisionConfig::default();

        let advanced_retina = advanced_retinal_circuits(image.view(), &config).unwrap();

        assert_eq!(advanced_retina.on_center_ganglion.dim(), (20, 20));
        assert_eq!(advanced_retina.off_center_ganglion.dim(), (20, 20));
        assert_eq!(advanced_retina.direction_selective_ganglion.len(), 8);
        assert_eq!(advanced_retina.iprgc_responses.dim(), (20, 20));
        assert_eq!(advanced_retina.local_edge_detectors.dim(), (20, 20));
    }

    #[test]
    fn test_binocular_stereo_processing() {
        let left_image =
            Array2::from_shape_vec((16, 16), (0..256).map(|x| x as f64 / 256.0).collect()).unwrap();
        let right_image =
            Array2::from_shape_vec((16, 16), (0..256).map(|x| (x + 5) as f64 / 256.0).collect())
                .unwrap();
        let config = BiologicalVisionConfig::default();

        let stereo_result =
            binocular_stereo_processing(left_image.view(), right_image.view(), &config).unwrap();

        assert_eq!(stereo_result.disparity_map.dim(), (16, 16));
        assert_eq!(stereo_result.depth_map.dim(), (16, 16));
        assert_eq!(stereo_result.ocular_dominance_map.dim(), (16, 16));
        assert_eq!(stereo_result.stereo_confidence.dim(), (16, 16));
        assert!(!stereo_result.binocular_neurons.is_empty());
    }

    #[test]
    fn test_visual_working_memory_processing() {
        let images: Vec<Array2<f64>> = (0..3)
            .map(|i| {
                Array2::from_shape_vec(
                    (8, 8),
                    (0..64).map(|x| (x + i * 10) as f64 / 64.0).collect(),
                )
                .unwrap()
            })
            .collect();
        let image_views: Vec<_> = images.iter().map(|img| img.view()).collect();
        let config = BiologicalVisionConfig::default();

        let vwm_result = visual_working_memory_processing(&image_views, &config).unwrap();

        assert!(!vwm_result.memory_slots.is_empty());
        assert!(!vwm_result.attention_weights.is_empty());
        assert!(!vwm_result.maintenance_activity.is_empty());
    }

    #[test]
    fn test_circadian_vision_processing() {
        let image =
            Array2::from_shape_vec((12, 12), (0..144).map(|x| x as f64 / 144.0).collect()).unwrap();
        let illumination_estimate = 0.7;
        let circadian_phase = 0.3;
        let config = BiologicalVisionConfig::default();

        let processed_image = circadian_vision_processing(
            image.view(),
            illumination_estimate,
            circadian_phase,
            &config,
        )
        .unwrap();

        assert_eq!(processed_image.dim(), (12, 12));
    }

    #[test]
    fn test_neural_plasticity_adaptation() {
        let images: Vec<Array2<f64>> = (0..10)
            .map(|i| {
                Array2::from_shape_vec((8, 8), (0..64).map(|x| (x + i * 5) as f64 / 64.0).collect())
                    .unwrap()
            })
            .collect();
        let image_views: Vec<_> = images.iter().map(|img| img.view()).collect();
        let config = BiologicalVisionConfig::default();

        let adaptation_maps = neural_plasticity_adaptation(&image_views, &config).unwrap();

        assert_eq!(adaptation_maps.dim(), (4, 8, 8)); // 4 adaptation types
    }
}
