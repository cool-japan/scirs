//! Enhanced wavelet denoising with advanced thresholding methods
//!
//! This module provides state-of-the-art wavelet denoising techniques including:
//! - Translation-invariant denoising
//! - Block thresholding
//! - Stein's unbiased risk estimate (SURE)
//! - BayesShrink and VisuShrink
//! - Adaptive thresholding
//! - Multiple alternative denoising algorithms (non-local means, total variation, etc.)
//! - SIMD optimizations and morphological operations

// Sub-modules
pub mod algorithms;
pub mod morphological;
pub mod simd;
pub mod thresholding;
pub mod types;
pub mod utils;
pub mod wavelet;

// Re-export main types for convenience
pub use types::{
    AdaptiveLMSConfig, BilateralConfig, Denoise2dResult, DenoiseConfig, DenoiseResult,
    GuidedFilterConfig, NonLocalMeansConfig, QualityMetrics, SubbandRetention, SubbandThresholds,
    ThresholdMethod, ThresholdRule, TotalVariationConfig, WienerConfig, WienerNoiseEstimation,
};

// Re-export main wavelet denoising functions
pub use wavelet::{
    apply_thresholding, compute_effective_df, compute_effective_df_ti, compute_quality_metrics,
    compute_sure_risk, denoise_wavelet_1d, denoise_wavelet_2d, estimate_noise_2d,
    estimate_noise_mad, threshold_subband,
};

// Re-export alternative denoising algorithms
pub use algorithms::{
    denoise_adaptive_lms, denoise_bilateral_1d, denoise_guided_filter_1d, denoise_median_1d,
    denoise_non_local_means_1d, denoise_total_variation_1d, denoise_wiener_1d,
};

// Re-export thresholding functions when module is available
pub use thresholding::{
    block_threshold, compute_bayes_threshold, compute_cv_threshold, compute_fdr_threshold,
    compute_minimax_threshold, compute_sure_threshold, firm_threshold, garotte_threshold,
    hard_threshold, hyperbolic_threshold, scad_threshold, soft_threshold,
};

// Re-export morphological operations when module is available
pub use morphological::{
    denoise_morphological_closing, denoise_morphological_opening, morphological_dilation,
    morphological_erosion,
};

// Re-export SIMD functions when module is available
pub use simd::{
    simd_average_unshifted_results, simd_circular_shift, simd_hard_threshold, simd_soft_threshold,
    simd_weighted_sum,
};

// Re-export utility functions
pub use utils::{
    adaptive_noise_estimation, compute_energy, compute_numerical_stability,
    compute_patch_similarity_1d, compute_snr_improvement, compute_tree_energy,
    estimate_memory_usage, memory_optimized_denoise_1d, next_power_of_two, validate_denoise_config,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_module_integration() {
        // Test that main functions are accessible through the module interface
        // Use power of 2 length (32) for better wavelet processing
        let signal = Array1::from_vec(vec![
            1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5, 1.5, 2.5, 1.2,
            2.2, 3.2, 2.2, 1.2, 0.2, 1.2, 2.2, 1.8, 2.8, 3.8, 2.8, 1.8, 0.8, 1.8, 2.8,
        ]);

        // Test wavelet denoising
        let config = DenoiseConfig::default();
        let result = denoise_wavelet_1d(&signal, &config);
        assert!(result.is_ok());
        let denoised_result = result.unwrap();
        // Check that the denoised signal length is reasonable (within a few elements of original)
        assert!(
            denoised_result.signal.len() >= signal.len()
                && denoised_result.signal.len() <= signal.len() + 4,
            "Denoised signal length {} should be close to original length {}",
            denoised_result.signal.len(),
            signal.len()
        );

        // Test noise estimation
        let noise_sigma = estimate_noise_mad(&signal);
        assert!(noise_sigma > 0.0);

        // Test configuration
        let config = DenoiseConfig::default();
        assert_eq!(config.method, ThresholdMethod::Soft);
        assert_eq!(config.threshold_rule, ThresholdRule::SURE);
        assert!(config.use_simd);
        assert!(config.parallel);
    }

    #[test]
    fn test_alternative_algorithms() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);

        // Test non-local means
        let nlm_config = NonLocalMeansConfig::default();
        let result = denoise_non_local_means_1d(&signal, &nlm_config);
        assert!(result.is_ok());

        // Test total variation
        let tv_config = TotalVariationConfig::default();
        let result = denoise_total_variation_1d(&signal, &tv_config);
        assert!(result.is_ok());

        // Test bilateral filtering
        let bilateral_config = BilateralConfig::default();
        let result = denoise_bilateral_1d(&signal, &bilateral_config);
        assert!(result.is_ok());

        // Test median filtering
        let result = denoise_median_1d(&signal, 3);
        assert!(result.is_ok());
    }

    #[test]
    fn test_configuration_types() {
        // Test all configuration types have sensible defaults
        let denoise_config = DenoiseConfig::default();
        assert!(denoise_config.parallel);
        assert!(denoise_config.use_simd);

        let nlm_config = NonLocalMeansConfig::default();
        assert_eq!(nlm_config.patch_size, 7);

        let tv_config = TotalVariationConfig::default();
        assert_eq!(tv_config.max_iterations, 100);

        let bilateral_config = BilateralConfig::default();
        assert_eq!(bilateral_config.window_size, 15);

        let wiener_config = WienerConfig::default();
        assert_eq!(
            wiener_config.noise_estimation,
            WienerNoiseEstimation::HighFrequency
        );

        let lms_config = AdaptiveLMSConfig::default();
        assert_eq!(lms_config.filter_length, 32);

        let guided_config = GuidedFilterConfig::default();
        assert_eq!(guided_config.radius, 8);
    }

    #[test]
    fn test_result_types() {
        // Test that result types can be created and accessed
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = DenoiseResult {
            signal,
            noise_sigma: 0.1,
            thresholds: vec![0.05, 0.03],
            retention_rate: 0.8,
            effective_df: 10.0,
            risk_estimate: Some(0.02),
            processing_time_ms: 5.0,
            memory_usage_bytes: 1024,
            stability_score: 0.95,
            snr_improvement_db: 15.0,
        };

        assert_eq!(result.signal.len(), 3);
        assert_eq!(result.noise_sigma, 0.1);
        assert_eq!(result.thresholds.len(), 2);
        assert!(result.risk_estimate.is_some());
    }
}
