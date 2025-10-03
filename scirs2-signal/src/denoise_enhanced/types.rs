//! Configuration types and data structures for enhanced denoising
//!
//! This module contains all the configuration structures, enums, and result types
//! used across the enhanced denoising algorithms.

use crate::dwt::Wavelet;
use scirs2_core::ndarray::{Array1, Array2};

/// Enhanced denoising configuration
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// Wavelet to use
    pub wavelet: Wavelet,
    /// Decomposition levels (None for automatic)
    pub levels: Option<usize>,
    /// Thresholding method
    pub method: ThresholdMethod,
    /// Threshold selection rule
    pub threshold_rule: ThresholdRule,
    /// Use translation-invariant denoising
    pub translation_invariant: bool,
    /// Number of shifts for TI denoising
    pub n_shifts: usize,
    /// Use parallel processing
    pub parallel: bool,
    /// Preserve approximation coefficients
    pub preserve_approx: bool,
    /// Level-dependent thresholding
    pub level_dependent: bool,
    /// Use SIMD optimization
    pub use_simd: bool,
    /// Memory optimization for large signals
    pub memory_optimized: bool,
    /// Block size for memory-optimized processing
    pub block_size: Option<usize>,
    /// Enhanced numerical stability
    pub numerical_stability: bool,
    /// Adaptive noise estimation
    pub adaptive_noise: bool,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            wavelet: Wavelet::DB(4),
            levels: None,
            method: ThresholdMethod::Soft,
            threshold_rule: ThresholdRule::SURE,
            translation_invariant: false,
            n_shifts: 8,
            parallel: true,
            preserve_approx: true,
            level_dependent: true,
            use_simd: true,
            memory_optimized: true,
            block_size: None,
            numerical_stability: true,
            adaptive_noise: true,
        }
    }
}

/// Thresholding methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Soft thresholding (shrinkage)
    Soft,
    /// Hard thresholding
    Hard,
    /// Garotte thresholding
    Garotte,
    /// SCAD (Smoothly Clipped Absolute Deviation)
    SCAD { a: f64 },
    /// Firm thresholding
    Firm { alpha: f64 },
    /// Hyperbolic thresholding
    Hyperbolic,
    /// Block thresholding
    Block { block_size: usize },
}

/// Threshold selection rules
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdRule {
    /// Universal threshold (VisuShrink)
    Universal,
    /// Stein's Unbiased Risk Estimate
    SURE,
    /// BayesShrink
    Bayes,
    /// Minimax threshold
    Minimax,
    /// Cross-validation
    CrossValidation,
    /// False Discovery Rate
    FDR { q: f64 },
    /// Custom threshold value
    Custom(f64),
}

/// Denoising result with diagnostics
#[derive(Debug, Clone)]
pub struct DenoiseResult {
    /// Denoised signal
    pub signal: Array1<f64>,
    /// Estimated noise level
    pub noise_sigma: f64,
    /// Thresholds used at each level
    pub thresholds: Vec<f64>,
    /// Percentage of coefficients retained
    pub retention_rate: f64,
    /// Effective degrees of freedom
    pub effective_df: f64,
    /// Risk estimate (if available)
    pub risk_estimate: Option<f64>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Numerical stability score (0-1)
    pub stability_score: f64,
    /// Signal-to-noise ratio improvement
    pub snr_improvement_db: f64,
}

/// 2D denoising result
#[derive(Debug, Clone)]
pub struct Denoise2dResult {
    /// Denoised image
    pub image: Array2<f64>,
    /// Estimated noise level
    pub noise_sigma: f64,
    /// Thresholds per subband
    pub thresholds: SubbandThresholds,
    /// Retention rates per subband
    pub retention_rates: SubbandRetention,
    /// Quality metrics
    pub quality: QualityMetrics,
}

/// Thresholds for each subband
#[derive(Debug, Clone)]
pub struct SubbandThresholds {
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    pub diagonal: Vec<f64>,
}

/// Retention rates for each subband
#[derive(Debug, Clone)]
pub struct SubbandRetention {
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    pub diagonal: Vec<f64>,
}

/// Quality metrics for denoising
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Estimated SNR improvement (dB)
    pub snr_improvement: f64,
    /// Edge preservation index
    pub edge_preservation: f64,
    /// Texture preservation index
    pub texture_preservation: f64,
}

/// Configuration for non-local means denoising
#[derive(Debug, Clone)]
pub struct NonLocalMeansConfig {
    /// Size of patches to compare
    pub patch_size: usize,
    /// Size of search window
    pub search_window: usize,
    /// Filtering parameter (controls decay of weights)
    pub filtering_parameter: f64,
    /// Use parallel processing
    pub parallel: bool,
}

impl Default for NonLocalMeansConfig {
    fn default() -> Self {
        Self {
            patch_size: 7,
            search_window: 21,
            filtering_parameter: 0.1,
            parallel: true,
        }
    }
}

/// Configuration for total variation denoising
#[derive(Debug, Clone)]
pub struct TotalVariationConfig {
    /// Regularization parameter (controls smoothness vs. fidelity trade-off)
    pub lambda: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Step size for gradient descent
    pub step_size: f64,
    /// Convergence tolerance
    pub tolerance: f64,
}

impl Default for TotalVariationConfig {
    fn default() -> Self {
        Self {
            lambda: 0.1,
            max_iterations: 100,
            step_size: 0.01,
            tolerance: 1e-6,
        }
    }
}

/// Configuration for bilateral filtering
#[derive(Debug, Clone)]
pub struct BilateralConfig {
    /// Spatial standard deviation
    pub spatial_sigma: f64,
    /// Intensity standard deviation
    pub intensity_sigma: f64,
    /// Window size for filtering
    pub window_size: usize,
    /// Use parallel processing
    pub parallel: bool,
}

impl Default for BilateralConfig {
    fn default() -> Self {
        Self {
            spatial_sigma: 2.0,
            intensity_sigma: 0.1,
            window_size: 15,
            parallel: true,
        }
    }
}

/// Configuration for Wiener filtering
#[derive(Debug, Clone, PartialEq)]
pub struct WienerConfig {
    /// Method for noise estimation
    pub noise_estimation: WienerNoiseEstimation,
}

impl Default for WienerConfig {
    fn default() -> Self {
        Self {
            noise_estimation: WienerNoiseEstimation::HighFrequency,
        }
    }
}

/// Noise estimation methods for Wiener filtering
#[derive(Debug, Clone, PartialEq)]
pub enum WienerNoiseEstimation {
    /// Constant noise power
    Constant(f64),
    /// Estimate from high-frequency components
    HighFrequency,
    /// Minimum statistics approach
    MinimumStatistics,
}

/// Configuration for adaptive LMS filtering
#[derive(Debug, Clone)]
pub struct AdaptiveLMSConfig {
    /// Length of adaptive filter
    pub filter_length: usize,
    /// Step size (learning rate)
    pub step_size: f64,
    /// Initial weights (None for zero initialization)
    pub initial_weights: Option<Vec<f64>>,
    /// Prediction mode (true) or noise cancellation mode (false)
    pub prediction_mode: bool,
    /// Delay for reference signal (in noise cancellation mode)
    pub delay: usize,
}

impl Default for AdaptiveLMSConfig {
    fn default() -> Self {
        Self {
            filter_length: 32,
            step_size: 0.01,
            initial_weights: None,
            prediction_mode: true,
            delay: 1,
        }
    }
}

/// Configuration for guided filtering
#[derive(Debug, Clone)]
pub struct GuidedFilterConfig {
    /// Window radius for guided filter
    pub radius: usize,
    /// Regularization parameter
    pub epsilon: f64,
}

impl Default for GuidedFilterConfig {
    fn default() -> Self {
        Self {
            radius: 8,
            epsilon: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_denoise_config_default() {
        let config = DenoiseConfig::default();
        assert_eq!(config.method, ThresholdMethod::Soft);
        assert_eq!(config.threshold_rule, ThresholdRule::SURE);
        assert!(config.use_simd);
        assert!(config.parallel);
    }

    #[test]
    fn test_threshold_method_variants() {
        let soft = ThresholdMethod::Soft;
        let hard = ThresholdMethod::Hard;
        let scad = ThresholdMethod::SCAD { a: 3.7 };

        assert_ne!(soft, hard);
        assert_eq!(scad, ThresholdMethod::SCAD { a: 3.7 });
    }

    #[test]
    fn test_config_structs_default() {
        let nlm_config = NonLocalMeansConfig::default();
        assert_eq!(nlm_config.patch_size, 7);
        assert_eq!(nlm_config.search_window, 21);

        let tv_config = TotalVariationConfig::default();
        assert_eq!(tv_config.max_iterations, 100);
        assert_eq!(tv_config.lambda, 0.1);

        let bilateral_config = BilateralConfig::default();
        assert_eq!(bilateral_config.window_size, 15);
        assert_eq!(bilateral_config.spatial_sigma, 2.0);

        let wiener_config = WienerConfig::default();
        assert_eq!(
            wiener_config.noise_estimation,
            WienerNoiseEstimation::HighFrequency
        );

        let lms_config = AdaptiveLMSConfig::default();
        assert_eq!(lms_config.filter_length, 32);
        assert_eq!(lms_config.step_size, 0.01);
    }
}
