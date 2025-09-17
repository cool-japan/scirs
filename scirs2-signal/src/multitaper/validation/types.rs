//! Type definitions and configuration structures for multitaper validation
//!
//! This module contains all the data structures, enums, and configuration types
//! used throughout the multitaper validation system.

/// Comprehensive validation result for multitaper methods
#[derive(Debug, Clone)]
pub struct MultitaperValidationResult {
    /// DPSS validation results
    pub dpss_validation: DpssValidationMetrics,
    /// Spectral estimation accuracy
    pub spectral_accuracy: SpectralAccuracyMetrics,
    /// Numerical stability metrics
    pub numerical_stability: NumericalStabilityMetrics,
    /// Performance comparison
    pub performance: PerformanceMetrics,
    /// Cross-validation with reference
    pub cross_validation: CrossValidationMetrics,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// Issues found during validation
    pub issues: Vec<String>,
}

/// DPSS validation metrics
#[derive(Debug, Clone)]
pub struct DpssValidationMetrics {
    /// Orthogonality error
    pub orthogonality_error: f64,
    /// Concentration ratio accuracy
    pub concentration_accuracy: f64,
    /// Eigenvalue ordering validity
    pub eigenvalue_ordering_valid: bool,
    /// Symmetry preservation
    pub symmetry_preserved: bool,
}

/// Spectral accuracy metrics
#[derive(Debug, Clone)]
pub struct SpectralAccuracyMetrics {
    /// Bias in spectral estimation
    pub bias: f64,
    /// Variance of spectral estimate
    pub variance: f64,
    /// Mean squared error
    pub mse: f64,
    /// Frequency resolution
    pub frequency_resolution: f64,
    /// Spectral leakage factor
    pub leakage_factor: f64,
}

/// Numerical stability metrics
#[derive(Debug, Clone)]
pub struct NumericalStabilityMetrics {
    /// Condition number of operations
    pub condition_number: f64,
    /// Numerical precision loss
    pub precision_loss: f64,
    /// Overflow/underflow occurrences
    pub numerical_issues: usize,
    /// Stability under extreme inputs
    pub extreme_input_stable: bool,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Standard implementation time (ms)
    pub standard_time_ms: f64,
    /// Enhanced implementation time (ms)
    pub enhanced_time_ms: f64,
    /// SIMD speedup factor
    pub simd_speedup: f64,
    /// Parallel speedup factor
    pub parallel_speedup: f64,
    /// Memory efficiency
    pub memory_efficiency: f64,
}

/// Cross-validation metrics
#[derive(Debug, Clone)]
pub struct CrossValidationMetrics {
    /// Correlation with reference implementation
    pub reference_correlation: f64,
    /// Maximum relative error
    pub max_relative_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Reference implementation available
    pub reference_available: bool,
    /// Confidence interval coverage
    pub confidence_coverage: f64,
}

/// Test signal configuration for validation
#[derive(Debug, Clone)]
pub struct TestSignalConfig {
    /// Length of test signals
    pub length: usize,
    /// Sampling frequency
    pub fs: f64,
    /// Number of test iterations
    pub num_tests: usize,
    /// Signal types to test
    pub signal_types: Vec<TestSignalType>,
    /// Time-bandwidth product
    pub nw: f64,
    /// Number of tapers
    pub k: usize,
    /// SNR levels to test (dB)
    pub snr_levels: Vec<f64>,
    /// Frequency content to test
    pub frequencies: Vec<f64>,
    /// Enable SIMD testing
    pub test_simd: bool,
    /// Enable robustness testing
    pub test_robustness: bool,
}

/// Types of test signals for validation
#[derive(Debug, Clone, PartialEq)]
pub enum TestSignalType {
    /// Pure sinusoid at specified frequency
    Sinusoid(f64),
    /// Sum of multiple sinusoids
    MultiSine(Vec<f64>),
    /// White Gaussian noise
    WhiteNoise,
    /// Colored noise (1/f^alpha)
    ColoredNoise(f64),
    /// Chirp signal (frequency sweep)
    Chirp { start_freq: f64, end_freq: f64 },
    /// Impulse response
    Impulse,
    /// Step function
    Step,
    /// Complex sinusoid
    ComplexSinusoid(f64),
}

impl Default for TestSignalConfig {
    fn default() -> Self {
        Self {
            length: 1024,
            fs: 1000.0,
            num_tests: 10,
            signal_types: vec![
                TestSignalType::Sinusoid(100.0),
                TestSignalType::WhiteNoise,
                TestSignalType::MultiSine(vec![50.0, 150.0, 250.0]),
            ],
            nw: 4.0,
            k: 7,
            snr_levels: vec![0.0, 10.0, 20.0, 30.0],
            frequencies: vec![50.0, 100.0, 150.0, 200.0, 250.0],
            test_simd: true,
            test_robustness: true,
        }
    }
}

/// Validation configuration parameters
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Enable verbose output
    pub verbose: bool,
    /// Number of Monte Carlo iterations
    pub monte_carlo_iterations: usize,
    /// Test extreme parameter ranges
    pub test_extreme_cases: bool,
    /// Enable performance profiling
    pub profile_performance: bool,
    /// Test different precision levels
    pub test_precision_levels: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-6,
            verbose: false,
            monte_carlo_iterations: 100,
            test_extreme_cases: true,
            profile_performance: true,
            test_precision_levels: false,
        }
    }
}

/// Robustness validation metrics
#[derive(Debug, Clone)]
pub struct RobustnessMetrics {
    /// Extreme case stability score
    pub extreme_case_stability: f64,
    /// Numerical consistency score
    pub numerical_consistency: f64,
    /// Memory scaling efficiency
    pub memory_scaling: f64,
    /// Convergence stability
    pub convergence_stability: f64,
    /// Noise robustness score
    pub noise_robustness: f64,
}

/// SIMD validation specific metrics
#[derive(Debug, Clone)]
pub struct SimdValidationMetrics {
    /// SIMD operations correctness
    pub correctness_score: f64,
    /// Performance improvement factor
    pub performance_improvement: f64,
    /// Memory access efficiency
    pub memory_efficiency: f64,
    /// Platform compatibility
    pub platform_compatible: bool,
}

/// Enhanced validation result with extended metrics
#[derive(Debug, Clone)]
pub struct EnhancedValidationResult {
    /// Standard validation metrics
    pub standard_metrics: MultitaperValidationResult,
    /// Robustness assessment
    pub robustness: RobustnessMetrics,
    /// SIMD-specific validation
    pub simd_metrics: SimdValidationMetrics,
    /// Parameter consistency validation
    pub parameter_consistency: f64,
    /// Numerical precision validation
    pub precision_validation: f64,
    /// Overall enhanced score
    pub enhanced_score: f64,
    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Precision test configuration
#[derive(Debug, Clone)]
pub struct PrecisionTestConfig {
    /// Test single precision (f32)
    pub test_f32: bool,
    /// Test double precision (f64)
    pub test_f64: bool,
    /// Test extended precision
    pub test_extended: bool,
    /// Precision loss tolerance
    pub precision_tolerance: f64,
}

impl Default for PrecisionTestConfig {
    fn default() -> Self {
        Self {
            test_f32: true,
            test_f64: true,
            test_extended: false,
            precision_tolerance: 1e-12,
        }
    }
}

/// Parameter range validation metrics
#[derive(Debug, Clone)]
pub struct ParameterRangeMetrics {
    /// Valid parameter range coverage
    pub range_coverage: f64,
    /// Edge case handling
    pub edge_case_handling: f64,
    /// Parameter interaction effects
    pub interaction_effects: f64,
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
}

/// Confidence interval validation results
#[derive(Debug, Clone)]
pub struct ConfidenceIntervalMetrics {
    /// Coverage probability
    pub coverage_probability: f64,
    /// Interval width consistency
    pub width_consistency: f64,
    /// Asymptotic validity
    pub asymptotic_validity: f64,
    /// Bootstrap validation
    pub bootstrap_validation: f64,
}

/// Signal quality assessment for test signals
#[derive(Debug, Clone)]
pub struct SignalQualityMetrics {
    /// Signal-to-noise ratio (dB)
    pub snr_db: f64,
    /// Spectral purity
    pub spectral_purity: f64,
    /// Dynamic range
    pub dynamic_range: f64,
    /// Frequency accuracy
    pub frequency_accuracy: f64,
}

/// Comprehensive test suite configuration
#[derive(Debug, Clone)]
pub struct ComprehensiveTestConfig {
    /// Basic signal configuration
    pub signal_config: TestSignalConfig,
    /// Validation configuration
    pub validation_config: ValidationConfig,
    /// Precision test configuration
    pub precision_config: PrecisionTestConfig,
    /// Enable extended testing
    pub extended_testing: bool,
    /// Parallel test execution
    pub parallel_execution: bool,
}

impl Default for ComprehensiveTestConfig {
    fn default() -> Self {
        Self {
            signal_config: TestSignalConfig::default(),
            validation_config: ValidationConfig::default(),
            precision_config: PrecisionTestConfig::default(),
            extended_testing: true,
            parallel_execution: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_configurations() {
        let signal_config = TestSignalConfig::default();
        assert_eq!(signal_config.length, 1024);
        assert_eq!(signal_config.k, 7);
        assert!(signal_config.test_simd);

        let validation_config = ValidationConfig::default();
        assert_eq!(validation_config.tolerance, 1e-6);
        assert!(validation_config.test_extreme_cases);

        let precision_config = PrecisionTestConfig::default();
        assert!(precision_config.test_f64);
    }

    #[test]
    fn test_signal_types() {
        let sinusoid = TestSignalType::Sinusoid(100.0);
        let white_noise = TestSignalType::WhiteNoise;
        let multi_sine = TestSignalType::MultiSine(vec![50.0, 100.0]);

        match sinusoid {
            TestSignalType::Sinusoid(freq) => assert_eq!(freq, 100.0),
            _ => panic!("Wrong signal type"),
        }

        assert_eq!(white_noise, TestSignalType::WhiteNoise);

        match multi_sine {
            TestSignalType::MultiSine(freqs) => assert_eq!(freqs, vec![50.0, 100.0]),
            _ => panic!("Wrong signal type"),
        }
    }

    #[test]
    fn test_comprehensive_config() {
        let config = ComprehensiveTestConfig::default();
        assert!(config.extended_testing);
        assert!(config.parallel_execution);
        assert_eq!(config.signal_config.length, 1024);
        assert_eq!(config.validation_config.tolerance, 1e-6);
    }
}