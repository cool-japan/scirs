//! Type definitions for comprehensive wavelet packet transform validation
//!
//! This module contains all the data structures, enums, and configuration types
//! used throughout the comprehensive WPT validation system.

use crate::dwt::Wavelet;
use crate::wpt_validation::WptValidationResult;
use scirs2_core::ndarray::Array2;

/// Comprehensive WPT validation result
#[derive(Debug, Clone)]
pub struct ComprehensiveWptValidationResult {
    /// Basic validation metrics
    pub basic_validation: WptValidationResult,
    /// Advanced frame theory validation
    pub frame_validation: FrameValidationMetrics,
    /// Multi-scale analysis validation
    pub multiscale_validation: MultiscaleValidationMetrics,
    /// Best basis algorithm validation
    pub best_basis_validation: BestBasisValidationMetrics,
    /// Statistical validation results
    pub statistical_validation: StatisticalValidationMetrics,
    /// Cross-validation with references
    pub cross_validation: CrossValidationMetrics,
    /// Robustness testing results
    pub robustness_testing: RobustnessTestingMetrics,
    /// Overall validation score (0-100)
    pub overall_score: f64,
    /// Critical issues that need attention
    pub issues: Vec<String>,
}

/// Frame theory validation metrics
#[derive(Debug, Clone)]
pub struct FrameValidationMetrics {
    /// Frame operator eigenvalue distribution
    pub eigenvalue_distribution: EigenvalueDistribution,
    /// Condition number of frame operator
    pub condition_number: f64,
    /// Frame coherence measure
    pub frame_coherence: f64,
    /// Redundancy factor
    pub redundancy_factor: f64,
    /// Frame reconstruction error bounds
    pub reconstruction_bounds: (f64, f64),
}

/// Eigenvalue distribution for frame analysis
#[derive(Debug, Clone)]
pub struct EigenvalueDistribution {
    /// Minimum eigenvalue
    pub min_eigenvalue: f64,
    /// Maximum eigenvalue
    pub max_eigenvalue: f64,
    /// Mean eigenvalue
    pub mean_eigenvalue: f64,
    /// Eigenvalue variance
    pub eigenvalue_variance: f64,
    /// Number of near-zero eigenvalues
    pub near_zero_count: usize,
}

/// Multi-scale validation metrics
#[derive(Debug, Clone)]
pub struct MultiscaleValidationMetrics {
    /// Scale-wise energy distribution
    pub scale_energy_distribution: Vec<f64>,
    /// Inter-scale correlation analysis
    pub inter_scale_correlations: Array2<f64>,
    /// Scale consistency measure
    pub scale_consistency: f64,
    /// Frequency localization accuracy
    pub frequency_localization: f64,
    /// Time localization accuracy
    pub time_localization: f64,
}

/// Best basis algorithm validation
#[derive(Debug, Clone)]
pub struct BestBasisValidationMetrics {
    /// Cost function convergence analysis
    pub convergence_analysis: ConvergenceAnalysis,
    /// Basis selection repeatability
    pub selection_repeatability: f64,
    /// Optimal basis characteristics
    pub optimal_basis_metrics: OptimalBasisMetrics,
    /// Algorithm efficiency metrics
    pub algorithm_efficiency: AlgorithmEfficiencyMetrics,
}

/// Convergence analysis for best basis algorithm
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysis {
    /// Number of iterations to convergence
    pub iterations_to_convergence: usize,
    /// Convergence rate estimate
    pub convergence_rate: f64,
    /// Final cost function value
    pub final_cost: f64,
    /// Cost reduction ratio
    pub cost_reduction_ratio: f64,
}

/// Optimal basis characteristics
#[derive(Debug, Clone)]
pub struct OptimalBasisMetrics {
    /// Basis sparsity measure
    pub sparsity_measure: f64,
    /// Energy concentration efficiency
    pub energy_concentration: f64,
    /// Basis adaptivity score
    pub adaptivity_score: f64,
    /// Local coherence measure
    pub local_coherence: f64,
}

/// Algorithm efficiency metrics
#[derive(Debug, Clone)]
pub struct AlgorithmEfficiencyMetrics {
    /// Computational complexity order
    pub complexity_order: f64,
    /// Memory efficiency score
    pub memory_efficiency: f64,
    /// Scalability factor
    pub scalability_factor: f64,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
}

/// Statistical validation metrics
#[derive(Debug, Clone)]
pub struct StatisticalValidationMetrics {
    /// Distribution of reconstruction errors
    pub error_distribution: ErrorDistribution,
    /// Confidence intervals for key metrics
    pub confidence_intervals: ConfidenceIntervals,
    /// Hypothesis testing results
    pub hypothesis_tests: HypothesisTestResults,
    /// Bootstrap validation results
    pub bootstrap_validation: BootstrapValidation,
}

/// Error distribution analysis
#[derive(Debug, Clone)]
pub struct ErrorDistribution {
    /// Mean error
    pub mean_error: f64,
    /// Error variance
    pub error_variance: f64,
    /// Error skewness
    pub error_skewness: f64,
    /// Error kurtosis
    pub error_kurtosis: f64,
    /// Maximum error percentile (99th)
    pub max_error_percentile: f64,
}

/// Confidence intervals for validation metrics
#[derive(Debug, Clone)]
pub struct ConfidenceIntervals {
    /// Energy conservation (95% CI)
    pub energy_conservation_ci: (f64, f64),
    /// Reconstruction error (95% CI)
    pub reconstruction_error_ci: (f64, f64),
    /// Frame bounds (95% CI)
    pub frame_bounds_ci: ((f64, f64), (f64, f64)),
}

/// Hypothesis testing results
#[derive(Debug, Clone)]
pub struct HypothesisTestResults {
    /// Perfect reconstruction test p-value
    pub perfect_reconstruction_pvalue: f64,
    /// Orthogonality test p-value
    pub orthogonality_pvalue: f64,
    /// Energy conservation test p-value
    pub energy_conservation_pvalue: f64,
    /// Frame property test p-value
    pub frame_property_pvalue: f64,
}

/// Bootstrap validation results
#[derive(Debug, Clone)]
pub struct BootstrapValidation {
    /// Bootstrap sample size
    pub sample_size: usize,
    /// Bootstrap mean estimates
    pub bootstrap_means: Vec<f64>,
    /// Bootstrap confidence intervals
    pub bootstrap_confidence_intervals: Vec<(f64, f64)>,
    /// Metric stability across bootstrap samples
    pub metric_stability: f64,
}

/// Cross-validation metrics
#[derive(Debug, Clone)]
pub struct CrossValidationMetrics {
    /// Reference implementation comparison results
    pub reference_comparison: ReferenceComparisonMetrics,
    /// Alternative algorithm comparison
    pub alternative_algorithm_comparison: AlgorithmComparisonMetrics,
    /// Implementation robustness score
    pub implementation_robustness: f64,
}

/// Reference implementation comparison
#[derive(Debug, Clone)]
pub struct ReferenceComparisonMetrics {
    /// Agreement with PyWavelets
    pub pywavelets_agreement: f64,
    /// Agreement with MATLAB Wavelet Toolbox
    pub matlab_agreement: f64,
    /// Cross-platform consistency
    pub cross_platform_consistency: f64,
}

/// Algorithm comparison metrics
#[derive(Debug, Clone)]
pub struct AlgorithmComparisonMetrics {
    /// Performance relative to standard algorithms
    pub relative_performance: f64,
    /// Accuracy compared to reference algorithms
    pub accuracy_comparison: f64,
    /// Computational efficiency ratio
    pub efficiency_ratio: f64,
}

/// Robustness testing metrics
#[derive(Debug, Clone)]
pub struct RobustnessTestingMetrics {
    /// Noise robustness score
    pub noise_robustness: f64,
    /// Outlier resistance
    pub outlier_resistance: f64,
    /// Parameter sensitivity analysis
    pub parameter_sensitivity: ParameterSensitivityMetrics,
    /// Extreme condition stability
    pub extreme_condition_stability: f64,
}

/// Parameter sensitivity analysis
#[derive(Debug, Clone)]
pub struct ParameterSensitivityMetrics {
    /// Sensitivity to signal length changes
    pub signal_length_sensitivity: f64,
    /// Sensitivity to decomposition level changes
    pub level_sensitivity: f64,
    /// Sensitivity to wavelet choice
    pub wavelet_sensitivity: f64,
    /// Overall parameter robustness
    pub overall_robustness: f64,
}

/// Configuration for comprehensive WPT validation
#[derive(Debug, Clone)]
pub struct ComprehensiveWptValidationConfig {
    /// Wavelets to test
    pub test_wavelets: Vec<Wavelet>,
    /// Signal lengths to test
    pub test_signal_lengths: Vec<usize>,
    /// Decomposition levels to test
    pub test_levels: Vec<usize>,
    /// Number of random trials per configuration
    pub random_trials: usize,
    /// Number of bootstrap samples for statistical validation
    pub bootstrap_samples: usize,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Numerical tolerance for comparisons
    pub tolerance: f64,
    /// Enable parallel processing where possible
    pub enable_parallel: bool,
    /// Types of test signals to generate
    pub test_signal_types: Vec<TestSignalType>,
}

/// Types of test signals for validation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TestSignalType {
    /// White noise
    WhiteNoise,
    /// Sinusoidal signals
    Sinusoidal,
    /// Chirp signals
    Chirp,
    /// Piecewise constant
    PiecewiseConstant,
    /// Piecewise polynomial
    PiecewisePolynomial,
    /// Fractal signals
    Fractal,
    /// Natural signals (images, audio characteristics)
    Natural,
}

impl Default for ComprehensiveWptValidationConfig {
    fn default() -> Self {
        Self {
            test_wavelets: vec![
                Wavelet::DB(4),
                Wavelet::DB(8),
                Wavelet::BiorNrNd { nr: 2, nd: 2 },
                Wavelet::Coif(3),
                Wavelet::Haar,
            ],
            test_signal_lengths: vec![64, 128, 256, 512, 1024],
            test_levels: vec![1, 2, 3, 4, 5],
            random_trials: 100,
            bootstrap_samples: 1000,
            confidence_level: 0.95,
            tolerance: 1e-12,
            enable_parallel: true,
            test_signal_types: vec![
                TestSignalType::WhiteNoise,
                TestSignalType::Sinusoidal,
                TestSignalType::Chirp,
                TestSignalType::PiecewiseConstant,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ComprehensiveWptValidationConfig::default();
        assert_eq!(config.random_trials, 100);
        assert_eq!(config.bootstrap_samples, 1000);
        assert_eq!(config.confidence_level, 0.95);
        assert_eq!(config.tolerance, 1e-12);
        assert!(config.enable_parallel);
    }

    #[test]
    fn test_test_signal_types() {
        let signal_types = vec![
            TestSignalType::WhiteNoise,
            TestSignalType::Sinusoidal,
            TestSignalType::Chirp,
        ];

        assert_eq!(signal_types.len(), 3);
        assert!(signal_types.contains(&TestSignalType::WhiteNoise));
    }
}