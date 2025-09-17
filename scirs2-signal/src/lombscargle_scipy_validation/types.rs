//! Configuration and result types for Lomb-Scargle SciPy validation
//!
//! This module contains all the configuration structures and result types
//! used throughout the comprehensive validation framework.

use std::collections::HashMap;

/// Comprehensive SciPy validation configuration
#[derive(Debug, Clone)]
pub struct ScipyValidationConfig {
    /// Numerical tolerance for comparisons
    pub tolerance: f64,
    /// Relative tolerance for comparisons
    pub relative_tolerance: f64,
    /// Test signal lengths to validate
    pub test_lengths: Vec<usize>,
    /// Sampling frequencies for test signals
    pub sampling_frequencies: Vec<f64>,
    /// Test frequencies to evaluate
    pub test_frequencies: Vec<f64>,
    /// Whether to test different normalization methods
    pub test_normalizations: bool,
    /// Whether to test edge cases
    pub test_edge_cases: bool,
    /// Number of Monte Carlo trials for statistical validation
    pub monte_carlo_trials: usize,
    /// Maximum allowed percentage error
    pub max_error_percent: f64,
}

impl Default for ScipyValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            relative_tolerance: 1e-10,
            test_lengths: vec![32, 64, 128, 256, 512, 1024],
            sampling_frequencies: vec![1.0, 10.0, 100.0],
            test_frequencies: vec![0.1, 1.0, 5.0, 10.0],
            test_normalizations: true,
            test_edge_cases: true,
            monte_carlo_trials: 100,
            max_error_percent: 0.01, // 0.01% maximum error
        }
    }
}

/// Results from comprehensive SciPy validation
#[derive(Debug, Clone)]
pub struct ScipyValidationResult {
    /// Basic accuracy validation results
    pub accuracy_results: AccuracyValidationResult,
    /// Normalization method validation
    pub normalization_results: Option<NormalizationValidationResult>,
    /// Edge case validation results
    pub edge_case_results: Option<EdgeCaseValidationResult>,
    /// Statistical properties validation
    pub statistical_results: StatisticalValidationResult,
    /// Performance comparison results
    pub performance_results: PerformanceValidationResult,
    /// Overall validation summary
    pub summary: ValidationSummary,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// Basic accuracy validation results
#[derive(Debug, Clone)]
pub struct AccuracyValidationResult {
    /// Maximum absolute error across all tests
    pub max_absolute_error: f64,
    /// Maximum relative error across all tests
    pub max_relative_error: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Correlation coefficient with SciPy results
    pub correlation: f64,
    /// Number of test cases that passed
    pub passed_cases: usize,
    /// Total number of test cases
    pub total_cases: usize,
}

/// Normalization method validation results
#[derive(Debug, Clone)]
pub struct NormalizationValidationResult {
    /// Results for each normalization method
    pub method_results: HashMap<String, AccuracyValidationResult>,
    /// Best performing normalization method
    pub best_method: String,
    /// Consistency between methods
    pub consistency_score: f64,
}

/// Edge case validation results
#[derive(Debug, Clone)]
pub struct EdgeCaseValidationResult {
    /// Very sparse sampling test result
    pub sparse_sampling: bool,
    /// Extreme dynamic range test result
    pub extreme_dynamic_range: bool,
    /// Very short time series test result
    pub short_time_series: bool,
    /// High frequency resolution test result
    pub high_freq_resolution: bool,
    /// Numerical stability score
    pub stability_score: f64,
}

/// Statistical properties validation
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    /// False alarm rate validation
    pub false_alarm_rate: f64,
    /// Detection power validation
    pub detection_power: f64,
    /// Bootstrap confidence interval coverage
    pub ci_coverage: f64,
    /// Statistical consistency score
    pub consistency_score: f64,
}

/// Performance comparison results
#[derive(Debug, Clone)]
pub struct PerformanceValidationResult {
    /// Execution time comparison (our_time / scipy_time)
    pub speed_ratio: f64,
    /// Memory usage comparison
    pub memory_ratio: f64,
    /// Scalability comparison
    pub scalability_score: f64,
}

/// Overall validation summary
#[derive(Debug, Clone)]
pub struct ValidationSummary {
    /// Overall pass/fail status
    pub passed: bool,
    /// Overall accuracy score (0-100)
    pub accuracy_score: f64,
    /// Overall performance score (0-100)
    pub performance_score: f64,
    /// Overall reliability score (0-100)
    pub reliability_score: f64,
    /// Combined overall score (0-100)
    pub overall_score: f64,
}

/// Advanced Lomb-Scargle validation configuration for in-depth testing
#[derive(Debug, Clone)]
pub struct AdvancedValidationConfig {
    /// Base validation config
    pub base: ScipyValidationConfig,
    /// Test numerical conditioning
    pub test_conditioning: bool,
    /// Test aliasing effects
    pub test_aliasing: bool,
    /// Test with realistic astronomical data
    pub test_astronomical_data: bool,
    /// Test phase coherence
    pub test_phase_coherence: bool,
    /// Number of bootstrap samples for uncertainty quantification
    pub bootstrap_samples: usize,
    /// Test frequency resolution limits
    pub test_frequency_resolution: bool,
}

impl Default for AdvancedValidationConfig {
    fn default() -> Self {
        Self {
            base: ScipyValidationConfig::default(),
            test_conditioning: true,
            test_aliasing: true,
            test_astronomical_data: true,
            test_phase_coherence: true,
            bootstrap_samples: 1000,
            test_frequency_resolution: true,
        }
    }
}

/// Advanced validation results with extended metrics
#[derive(Debug, Clone)]
pub struct AdvancedValidationResult {
    /// Base validation results
    pub base_results: ScipyValidationResult,
    /// Numerical conditioning test results
    pub conditioning_results: Option<ConditioningTestResult>,
    /// Aliasing test results
    pub aliasing_results: Option<AliasingTestResult>,
    /// Astronomical data test results
    pub astronomical_results: Option<AstronomicalTestResult>,
    /// Phase coherence test results
    pub phase_coherence_results: Option<PhaseCoherenceResult>,
    /// Bootstrap uncertainty quantification
    pub uncertainty_results: Option<UncertaintyResult>,
    /// Frequency resolution test results
    pub frequency_resolution_results: Option<FrequencyResolutionResult>,
}

/// Numerical conditioning test results
#[derive(Debug, Clone)]
pub struct ConditioningTestResult {
    /// Condition number of the normal equations
    pub condition_number: f64,
    /// Stability under perturbations
    pub perturbation_stability: f64,
    /// Numerical rank deficiency detection
    pub rank_deficiency_detected: bool,
    /// Gradient-based stability measure
    pub gradient_stability: f64,
}

/// Aliasing test results
#[derive(Debug, Clone)]
pub struct AliasingTestResult {
    /// Nyquist aliasing detection accuracy
    pub nyquist_detection: f64,
    /// Sub-Nyquist aliasing handling
    pub sub_nyquist_handling: f64,
    /// False peak suppression
    pub false_peak_suppression: f64,
    /// Spectral leakage mitigation
    pub leakage_mitigation: f64,
}

/// Astronomical data test results
#[derive(Debug, Clone)]
pub struct AstronomicalTestResult {
    /// Variable star detection accuracy
    pub variable_star_detection: f64,
    /// Exoplanet transit detection
    pub transit_detection: f64,
    /// RR Lyrae period determination
    pub rr_lyrae_accuracy: f64,
    /// Multi-periodic source handling
    pub multi_periodic_handling: f64,
}

/// Phase coherence test results
#[derive(Debug, Clone)]
pub struct PhaseCoherenceResult {
    /// Phase preservation accuracy
    pub phase_accuracy: f64,
    /// Coherence stability over time
    pub coherence_stability: f64,
    /// Phase wrapping handling
    pub phase_wrapping_handling: f64,
}

/// Bootstrap uncertainty quantification results
#[derive(Debug, Clone)]
pub struct UncertaintyResult {
    /// Bootstrap confidence intervals
    pub confidence_intervals: Vec<(f64, f64)>,
    /// Bias estimation
    pub bias_estimate: f64,
    /// Variance estimation
    pub variance_estimate: f64,
    /// Coverage probability
    pub coverage_probability: f64,
}

/// Frequency resolution test results
#[derive(Debug, Clone)]
pub struct FrequencyResolutionResult {
    /// Minimum resolvable frequency separation
    pub min_frequency_separation: f64,
    /// Resolution vs baseline length scaling
    pub resolution_scaling: f64,
    /// Spectral window characterization
    pub spectral_window_quality: f64,
}
