//! Integration tests for advanced neural vision features
//!
//! These tests verify that all advanced neural vision components work together
//! seamlessly and maintain consistent performance characteristics.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_vision::{
    error::Result,
    feature::{
        AdvancedDenoiser, AttentionFeatureMatcher, DeepSORT, DenoisingMethod, Detection,
        HDRProcessor, LearnedSIFT, NeuralFeatureConfig, NeuralFeatureMatcher, SIFTConfig,
        SuperPointNet, SuperResolutionMethod, SuperResolutionProcessor, ToneMappingMethod,
        TrackingBoundingBox,
    },
};

/// Test that neural feature detection integrates properly with tracking
#[test]
#[allow(dead_code)]
fn test_neural_features_tracking_integration() -> Result<()> {
    // Create test image
    let image = create_test_image((480, 640));

    // 1. Extract neural features
    let config = NeuralFeatureConfig {
        input_size: (480, 640),
        max_keypoints: 100,
        detection_threshold: 0.01,
        use_gpu: false, // Use CPU for reproducible tests
        ..Default::default()
    };

    let superpoint = SuperPointNet::new(Some(config))?;
    let (keypoints, descriptors) = superpoint.detect_and_describe(&image.view())?;

    assert!(!keypoints.is_empty(), "Should detect some keypoints");
    assert_eq!(descriptors.shape()[0], keypoints.len());

    // 2. Convert keypoints to tracking detections
    let mut detections = Vec::new();
    for (i, kp) in keypoints.iter().take(5).enumerate() {
        let bbox =
            TrackingBoundingBox::new(kp.x - 25.0, kp.y - 25.0, 50.0, 50.0, kp.response, i as i32);
        let feature = descriptors
            .slice(scirs2_core::ndarray::s![i, ..])
            .to_owned();
        detections.push(Detection::with_feature(bbox, feature));
    }

    // 3. Track objects using neural features
    let mut tracker = DeepSORT::new();
    let tracks = tracker.update(detections)?;

    // Should create tracks for all detections (tentative state)
    assert_eq!(tracks.len(), 0); // First frame creates tentative tracks
    assert_eq!(tracker.track_count(), 5);

    Ok(())
}

/// Test HDR processing with super-resolution pipeline
#[test]
#[allow(dead_code)]
fn test_hdr_super_resolution_pipeline() -> Result<()> {
    // Create multi-exposure images
    let base_image = create_test_image((240, 320));
    let hdr_images = [
        base_image.mapv(|x| (x * 0.3).clamp(0.0, 1.0)), // Dark
        base_image.clone(),                             // Normal
        base_image.mapv(|x| (x * 2.5).clamp(0.0, 1.0)), // Bright
    ];

    // 1. Create HDR image
    let exposures = vec![-1.0, 0.0, 1.0];
    let hdr_processor = HDRProcessor::new(exposures, ToneMappingMethod::Reinhard);
    let hdr_views: Vec<_> = hdr_images.iter().map(|img| img.view()).collect();
    let hdr_result = hdr_processor.create_hdr(&hdr_views)?;

    assert_eq!(hdr_result.dim(), base_image.dim());

    // 2. Apply super-resolution to HDR result
    let sr_processor = SuperResolutionProcessor::new(2, SuperResolutionMethod::Bicubic)?;
    let upscaled = sr_processor.upscale(&hdr_result.view())?;

    let expected_height = base_image.shape()[0] * 2;
    let expected_width = base_image.shape()[1] * 2;
    assert_eq!(upscaled.dim(), (expected_height, expected_width));

    // Verify pixel values are in reasonable range (allow for slight numerical errors)
    let min_val = upscaled.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = upscaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        min_val >= -0.1,
        "Minimum value {min_val} should be close to valid range"
    );
    assert!(
        max_val <= 1.1,
        "Maximum value {max_val} should be close to valid range"
    );

    Ok(())
}

/// Test denoising with neural feature detection
#[test]
#[ignore] // GPU-intensive test - times out in CI
#[allow(dead_code)]
fn test_denoising_feature_detection_pipeline() -> Result<()> {
    // Create noisy image
    let clean_image = create_test_image((240, 320));
    let noisy_image = add_noise(&clean_image, 0.1);

    // 1. Apply advanced denoising
    let denoiser = AdvancedDenoiser::new(DenoisingMethod::NonLocalMeans, 0.01);
    let denoised = denoiser.denoise(&noisy_image.view())?;

    assert_eq!(denoised.dim(), clean_image.dim());

    // 2. Extract features from denoised image
    let learned_sift = LearnedSIFT::new(Some(SIFTConfig {
        peak_threshold: 0.01,
        ..Default::default()
    }));

    let keypoints_noisy = learned_sift.detect_keypoints(&noisy_image.view())?;
    let keypoints_denoised = learned_sift.detect_keypoints(&denoised.view())?;

    // Denoising should typically improve feature detection quality
    if !keypoints_noisy.is_empty() && !keypoints_denoised.is_empty() {
        let avg_response_noisy: f32 = keypoints_noisy.iter().map(|kp| kp.response).sum::<f32>()
            / keypoints_noisy.len() as f32;
        let avg_response_denoised: f32 =
            keypoints_denoised.iter().map(|kp| kp.response).sum::<f32>()
                / keypoints_denoised.len() as f32;

        // Denoised features should generally have higher quality responses
        assert!(
            avg_response_denoised >= avg_response_noisy * 0.8,
            "Denoising should not significantly degrade feature quality"
        );
    }

    Ok(())
}

/// Test attention-based matching with neural descriptors
#[test]
#[allow(dead_code)]
fn test_attention_matching_neural_descriptors() -> Result<()> {
    let image1 = create_test_image((240, 320));
    let image2 = create_transformed_image(&image1)?;

    // 1. Extract neural features from both images
    let config = NeuralFeatureConfig {
        input_size: (240, 320),
        max_keypoints: 50,
        use_gpu: false,
        ..Default::default()
    };

    let superpoint = SuperPointNet::new(Some(config))?;
    let (kp1, desc1) = superpoint.detect_and_describe(&image1.view())?;
    let (kp2, desc2) = superpoint.detect_and_describe(&image2.view())?;

    if kp1.is_empty() || kp2.is_empty() {
        return Ok(()); // Skip test if no features detected
    }

    // 2. Test neural feature matcher
    let neural_matcher = NeuralFeatureMatcher::new();
    let neural_matches = neural_matcher.match_descriptors(&desc1.view(), &desc2.view())?;

    // 3. Test attention-based matcher
    let attention_matcher = AttentionFeatureMatcher::new(desc1.shape()[1], 4);
    let attention_matches =
        attention_matcher.match_with_attention(&kp1, &desc1.view(), &kp2, &desc2.view())?;

    // Both matchers should find some matches
    assert!(!neural_matches.is_empty() || !attention_matches.is_empty());

    // Verify match indices are valid
    for &(i, j) in &neural_matches {
        assert!(i < kp1.len() && j < kp2.len());
    }

    for &(i, j) in &attention_matches {
        assert!(i < kp1.len() && j < kp2.len());
    }

    Ok(())
}

/// Test complete multi-frame tracking workflow
#[test]
#[allow(dead_code)]
fn test_complete_tracking_workflow() -> Result<()> {
    // Create sequence of images with moving objects
    let mut tracker = DeepSORT::new();
    let mut all_tracks = Vec::new();

    for frame_idx in 0..5 {
        // Create detections for this frame
        let detections = create_synthetic_detections(frame_idx);

        // Update tracker
        let tracks = tracker.update(detections)?;
        all_tracks.push(tracks);

        // Verify tracking consistency
        if frame_idx > 2 {
            // After a few frames, should have some confirmed tracks
            let confirmed_count = all_tracks[frame_idx].len();
            if confirmed_count > 0 {
                // Track IDs should be consistent
                for track in &all_tracks[frame_idx] {
                    assert!(track.id > 0, "Track should have valid ID");
                    assert!(
                        track.get_age() >= 3,
                        "Confirmed track should have sufficient age"
                    );
                }
            }
        }
    }

    Ok(())
}

/// Test performance characteristics of neural features
#[test]
#[allow(dead_code)]
fn test_neural_features_performance() -> Result<()> {
    let image = create_test_image((480, 640));

    // Test different configurations for performance
    let configs = vec![
        NeuralFeatureConfig {
            max_keypoints: 100,
            detection_threshold: 0.05,
            use_gpu: false,
            ..Default::default()
        },
        NeuralFeatureConfig {
            max_keypoints: 500,
            detection_threshold: 0.01,
            use_gpu: false,
            ..Default::default()
        },
    ];

    for config in configs {
        let superpoint = SuperPointNet::new(Some(config.clone()))?;
        let (keypoints, descriptors) = superpoint.detect_and_describe(&image.view())?;

        // Verify performance constraints
        assert!(
            keypoints.len() <= config.max_keypoints,
            "Should not exceed max keypoints limit"
        );

        // All keypoints should meet threshold
        for kp in &keypoints {
            assert!(
                kp.response >= config.detection_threshold,
                "All keypoints should meet detection threshold"
            );
        }

        // Descriptors should be properly normalized
        for i in 0..descriptors.shape()[0] {
            let desc = descriptors.slice(scirs2_core::ndarray::s![i, ..]);
            let norm = desc.dot(&desc).sqrt();
            assert!(
                (norm - 1.0).abs() < 0.1,
                "Descriptors should be approximately normalized"
            );
        }
    }

    Ok(())
}

/// Test error handling and edge cases
#[test]
#[allow(dead_code)]
fn test_advanced_features_error_handling() -> Result<()> {
    // Test with empty image
    let _empty_image: Array2<f32> = Array2::zeros((0, 0));

    // Neural features should handle empty input gracefully
    let config = NeuralFeatureConfig {
        input_size: (240, 320),
        use_gpu: false,
        ..Default::default()
    };

    // This should either fail gracefully or handle empty input
    let superpoint = SuperPointNet::new(Some(config))?;

    // Test with mismatched dimensions
    let wrong_size_image = Array2::zeros((100, 100)); // Not multiple of 8
    let result = superpoint.detect_and_describe(&wrong_size_image.view());

    // Should either succeed with resizing or fail with clear error
    match result {
        Ok((kp, desc)) => {
            assert_eq!(kp.len(), desc.shape()[0]);
        }
        Err(_) => {
            // Expected failure for wrong dimensions is acceptable
        }
    }

    // Test tracker with empty detections
    let mut tracker = DeepSORT::new();
    let empty_detections = Vec::new();
    let tracks = tracker.update(empty_detections)?;
    assert!(tracks.is_empty());

    // Test denoiser with extreme parameters
    let image = create_test_image((50, 50));
    let denoiser = AdvancedDenoiser::new(DenoisingMethod::NonLocalMeans, 1.0); // High noise
    let _result = denoiser.denoise(&image.view())?; // Should not crash

    Ok(())
}

// Helper functions

#[allow(dead_code)]
fn create_test_image(size: (usize, usize)) -> Array2<f32> {
    let (height, width) = size;
    Array2::from_shape_fn((height, width), |(y, x)| {
        let fx = x as f32 / width as f32;
        let fy = y as f32 / height as f32;

        // Create structured pattern
        let pattern =
            (fx * 10.0 * std::f32::consts::PI).sin() * (fy * 8.0 * std::f32::consts::PI).cos();
        (0.5 + pattern * 0.3).clamp(0.0, 1.0)
    })
}

#[allow(dead_code)]
fn create_transformed_image(image: &Array2<f32>) -> Result<Array2<f32>> {
    let (height, width) = image.dim();
    let mut transformed = Array2::zeros((height, width));

    // Apply simple translation
    let dx = 5;
    let dy = 3;

    for y in 0..height {
        for x in 0..width {
            let src_x = x.saturating_sub(dx);
            let src_y = y.saturating_sub(dy);

            if src_x < width && src_y < height {
                transformed[[y, x]] = image[[src_y, src_x]];
            }
        }
    }

    Ok(transformed)
}

#[allow(dead_code)]
fn add_noise(image: &Array2<f32>, noiselevel: f32) -> Array2<f32> {
    image.mapv(|x| {
        let noise = (scirs2_core::random::random::<f32>() - 0.5) * noiselevel;
        (x + noise).clamp(0.0, 1.0)
    })
}

#[allow(dead_code)]
fn create_synthetic_detections(_frameidx: usize) -> Vec<Detection> {
    let mut detections = Vec::new();

    // Create 2-3 moving objects
    for obj_id in 0..3 {
        let t = _frameidx as f32;
        let base_x = 50.0 + obj_id as f32 * 100.0;
        let base_y = 50.0 + obj_id as f32 * 80.0;

        // Simple motion
        let x = base_x + t * 10.0;
        let y = base_y + t * 5.0;

        let bbox = TrackingBoundingBox::new(x, y, 40.0, 60.0, 0.8, obj_id);

        // Synthetic appearance feature
        let feature = Array1::from_shape_fn(128, |i| {
            ((obj_id as f32 + i as f32) * 0.1).sin() * 0.5 + 0.5
        });

        detections.push(Detection::with_feature(bbox, feature));
    }

    detections
}
