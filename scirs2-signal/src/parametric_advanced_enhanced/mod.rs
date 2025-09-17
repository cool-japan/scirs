//! Advanced-enhanced parametric spectral estimation with SIMD acceleration
//!
//! This module provides high-performance implementations of parametric spectral
//! estimation methods with scirs2-core SIMD and parallel processing optimizations.
//!
//! # Module Organization
//!
//! - [`types`] - Data structures and configuration types
//! - [`core`] - Main ARMA estimation algorithms with SIMD acceleration
//! - [`adaptive`] - Adaptive AR estimation for non-stationary signals
//! - [`robust`] - Robust estimation with outlier rejection
//! - [`high_resolution`] - Eigenvalue-based high-resolution methods (MUSIC, ESPRIT)
//! - [`multitaper`] - Multitaper parametric estimation
//! - [`utils`] - Utility functions for spectral analysis and diagnostics
//!
//! # Key Features
//!
//! - SIMD-accelerated matrix operations for AR/ARMA parameter estimation
//! - Parallel order selection with cross-validation
//! - Enhanced numerical stability and convergence detection
//! - Memory-efficient processing for large signals
//! - Advanced model validation and diagnostics
//! - Real-time processing capabilities
//!
//! # Examples
//!
//! ## Basic ARMA Estimation
//!
//! ```rust
//! use scirs2_signal::parametric_advanced_enhanced::{advanced_enhanced_arma, AdvancedEnhancedConfig};
//! use ndarray::Array1;
//!
//! // Generate test signal
//! let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3]);
//! let config = AdvancedEnhancedConfig::default();
//!
//! // Estimate AR(2) MA(1) model
//! let result = advanced_enhanced_arma(&signal, 2, 1, &config)?;
//!
//! // Check results
//! assert!(result.convergence_info.converged);
//! assert!(result.diagnostics.is_stable);
//! println!("Noise variance: {}", result.noise_variance);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Adaptive AR Estimation
//!
//! ```rust
//! use scirs2_signal::parametric_advanced_enhanced::{adaptive_ar_spectral_estimation, AdaptiveARConfig};
//!
//! let signal = vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3, 1.9, 2.1];
//! let config = AdaptiveARConfig::default();
//!
//! let result = adaptive_ar_spectral_estimation(&signal, 2, &config)?;
//! println!("Number of time windows: {}", result.time_vector.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Robust Estimation
//!
//! ```rust
//! use scirs2_signal::parametric_advanced_enhanced::{robust_parametric_spectral_estimation, RobustParametricConfig};
//!
//! let mut signal = vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3];
//! signal[3] = 10.0; // Add outlier
//!
//! let config = RobustParametricConfig::default();
//! let result = robust_parametric_spectral_estimation(&signal, 2, 1, &config)?;
//!
//! println!("Robust scale: {}", result.robust_scale);
//! println!("Outliers detected: {}", result.outliers.iter().filter(|&&x| x).count());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// Declare all submodules
pub mod adaptive;
pub mod core;
pub mod high_resolution;
pub mod multitaper;
pub mod robust;
pub mod types;
pub mod utils;

// Re-export main types for public API
pub use types::*;

// Re-export main functions
pub use adaptive::adaptive_ar_spectral_estimation;
pub use core::advanced_enhanced_arma;
pub use high_resolution::high_resolution_spectral_estimation;
pub use multitaper::multitaper_parametric_estimation;
pub use robust::robust_parametric_spectral_estimation;

// Re-export commonly used utility functions
pub use utils::{
    compute_ar_psd, compute_ar_residuals, compute_arma_psd, compute_arma_residuals,
    compute_basic_diagnostics, compute_comprehensive_diagnostics, estimate_memory_usage,
    generate_frequency_grid,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Test that main types are accessible
        let _config = AdvancedEnhancedConfig::default();
        let _adaptive_config = AdaptiveARConfig::default();
        let _robust_config = RobustParametricConfig::default();
        let _hr_config = HighResolutionConfig::default();
        let _mt_config = MultitaperParametricConfig::default();

        // Test that enums are accessible
        let _method = HighResolutionMethod::MUSIC;
        let _combination = CombinationMethod::ArithmeticMean;
        let _criterion = OrderSelectionCriterion::AIC;
    }

    #[test]
    fn test_advanced_enhanced_arma_integration() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3, 1.9, 2.1]);
        let config = AdvancedEnhancedConfig::default();

        let result = advanced_enhanced_arma(&signal, 2, 1, &config);
        assert!(result.is_ok());

        let arma_result = result.unwrap();
        assert_eq!(arma_result.ar_coeffs.len(), 3); // [1, a1, a2]
        assert_eq!(arma_result.ma_coeffs.len(), 2); // [1, b1]
        assert!(arma_result.noise_variance > 0.0);
        assert!(arma_result.performance_stats.total_time_ms >= 0.0);
    }

    #[test]
    fn test_adaptive_ar_integration() {
        let signal = vec![
            1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3, 1.9, 2.1, 2.4, 1.6, 2.0,
        ];
        let config = AdaptiveARConfig::default();

        let result = adaptive_ar_spectral_estimation(&signal, 2, &config);
        assert!(result.is_ok());

        let adaptive_result = result.unwrap();
        assert!(!adaptive_result.time_vector.is_empty());
        assert!(!adaptive_result.orders.is_empty());
        assert_eq!(
            adaptive_result.spectral_estimates.nrows(),
            adaptive_result.time_vector.len()
        );
    }

    #[test]
    fn test_robust_estimation_integration() {
        let mut signal = vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3, 1.9, 2.1];
        signal[5] = 15.0; // Add outlier

        let config = RobustParametricConfig::default();
        let result = robust_parametric_spectral_estimation(&signal, 2, 1, &config);
        assert!(result.is_ok());

        let robust_result = result.unwrap();
        assert_eq!(robust_result.ar_coeffs.len(), 3);
        assert_eq!(robust_result.ma_coeffs.len(), 2);
        assert!(robust_result.robust_scale > 0.0);
        assert_eq!(robust_result.outliers.len(), signal.len());
    }

    #[test]
    fn test_high_resolution_integration() {
        let signal = vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3, 1.9, 2.1];
        let config = HighResolutionConfig::default();

        let result = high_resolution_spectral_estimation(&signal, &config);
        assert!(result.is_ok());

        let hr_result = result.unwrap();
        assert!(!hr_result.frequency_estimates.is_empty());
        assert!(!hr_result.eigenvalues.is_empty());
        assert!(hr_result.signal_subspace_dim > 0);
        assert!(hr_result.noise_subspace_dim >= 0);
    }

    #[test]
    fn test_multitaper_integration() {
        let signal = vec![
            1.0, 2.0, 1.5, 2.5, 1.8, 2.2, 1.7, 2.3, 1.9, 2.1, 2.4, 1.6, 2.0,
        ];
        let config = MultitaperParametricConfig::default();

        let result = multitaper_parametric_estimation(&signal, &config);
        assert!(result.is_ok());

        let mt_result = result.unwrap();
        assert!(!mt_result.combined_spectrum.psd.is_empty());
        assert!(!mt_result.individual_estimates.is_empty());
        assert!(mt_result.degrees_of_freedom > 0.0);
    }

    #[test]
    fn test_utility_functions_integration() {
        use ndarray::Array1;

        // Test PSD computation
        let ar_coeffs = Array1::from_vec(vec![1.0, -0.5, 0.2]);
        let noise_variance = 1.0;
        let frequencies = generate_frequency_grid(10, 2.0);

        let psd = compute_ar_psd(&ar_coeffs, noise_variance, &frequencies);
        assert!(psd.is_ok());
        assert_eq!(psd.unwrap().len(), 10);

        // Test residual computation
        let signal = Array1::from_vec(vec![1.0, 2.0, 1.5, 2.5, 1.8, 2.2]);
        let residuals = compute_ar_residuals(&signal, &ar_coeffs);
        assert!(residuals.is_ok());
        assert_eq!(residuals.unwrap().len(), signal.len());
    }

    #[test]
    fn test_error_handling_integration() {
        // Test with empty signal
        let empty_signal = Array1::from_vec(vec![]);
        let config = AdvancedEnhancedConfig::default();

        let result = advanced_enhanced_arma(&empty_signal, 2, 1, &config);
        assert!(result.is_err());

        // Test with invalid order
        let signal = Array1::from_vec(vec![1.0, 2.0]);
        let result = advanced_enhanced_arma(&signal, 10, 5, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_configuration_defaults() {
        let advanced_config = AdvancedEnhancedConfig::default();
        assert_eq!(advanced_config.max_iterations, 500);
        assert_eq!(advanced_config.tolerance, 1e-10);
        assert!(advanced_config.use_simd);
        assert!(advanced_config.use_parallel);

        let adaptive_config = AdaptiveARConfig::default();
        assert_eq!(adaptive_config.initial_order, 10);
        assert_eq!(adaptive_config.max_order, 50);
        assert_eq!(
            adaptive_config.order_selection,
            OrderSelectionCriterion::AIC
        );

        let robust_config = RobustParametricConfig::default();
        assert_eq!(robust_config.ar_order, 10);
        assert_eq!(robust_config.ma_order, 5);
        assert_eq!(robust_config.scale_estimator, ScaleEstimator::MAD);

        let hr_config = HighResolutionConfig::default();
        assert_eq!(hr_config.method, HighResolutionMethod::MUSIC);
        assert_eq!(hr_config.order, 20);
        assert_eq!(hr_config.frequency_bins, 1024);

        let mt_config = MultitaperParametricConfig::default();
        assert_eq!(mt_config.num_tapers, 7);
        assert_eq!(mt_config.time_bandwidth, 4.0);
        assert_eq!(mt_config.ar_order, 20);
    }
}
