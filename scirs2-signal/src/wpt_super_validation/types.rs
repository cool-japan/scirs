//! Type definitions for advanced WPT validation framework
//!
//! This module contains all the data structures and enums used throughout
//! the comprehensive WPT validation system, including result types for
//! mathematical property validation, SIMD testing, performance analysis,
//! and statistical validation.

use crate::dwt::Wavelet;
use crate::wpt_validation::{OrthogonalityMetrics, PerformanceMetrics, WptValidationResult};
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Advanced-comprehensive WPT validation result
#[derive(Debug, Clone)]
pub struct AdvancedWptValidationResult {
    /// Basic validation results
    pub basic_validation: WptValidationResult,
    /// Advanced mathematical property validation
    pub mathematical_properties: MathematicalPropertyValidation,
    /// SIMD implementation validation
    pub simd_validation: SimdValidationResult,
    /// Cross-platform consistency results
    pub platform_consistency: PlatformConsistencyResult,
    /// Statistical validation of basis selection
    pub statistical_validation: StatisticalValidationResult,
    /// Performance regression analysis
    pub performance_regression: PerformanceRegressionResult,
    /// Memory safety validation
    pub memory_safety: MemorySafetyResult,
    /// Real-time processing validation
    pub realtime_validation: RealtimeValidationResult,
    /// Overall validation status
    pub overall_status: ValidationStatus,
}

/// Mathematical property validation
#[derive(Debug, Clone)]
pub struct MathematicalPropertyValidation {
    /// Perfect reconstruction validation
    pub perfect_reconstruction: PerfectReconstructionValidation,
    /// Tight frame property validation
    pub tight_frame_validation: TightFrameValidation,
    /// Orthogonality validation with advanced metrics
    pub orthogonality_advanced: AdvancedOrthogonalityValidation,
    /// Energy conservation validation
    pub energy_conservation: EnergyConservationValidation,
    /// Coefficient distribution analysis
    pub coefficient_analysis: CoefficientDistributionAnalysis,
}

/// Perfect reconstruction validation
#[derive(Debug, Clone)]
pub struct PerfectReconstructionValidation {
    /// Maximum reconstruction error across all test signals
    pub max_error: f64,
    /// RMS reconstruction error
    pub rms_error: f64,
    /// Frequency domain reconstruction accuracy
    pub frequency_domain_error: f64,
    /// Reconstruction quality by frequency band
    pub frequency_band_errors: Array1<f64>,
    /// Signal type specific errors
    pub signal_type_errors: HashMap<String, f64>,
}

/// Tight frame property validation
#[derive(Debug, Clone)]
pub struct TightFrameValidation {
    /// Frame bound verification
    pub frame_bounds_verified: bool,
    /// Lower frame bound
    pub lower_bound: f64,
    /// Upper frame bound
    pub upper_bound: f64,
    /// Frame bound ratio (should be 1.0 for tight frames)
    pub bound_ratio: f64,
    /// Parseval relation validation
    pub parseval_verified: bool,
    /// Parseval error
    pub parseval_error: f64,
}

/// Advanced orthogonality validation
#[derive(Debug, Clone)]
pub struct AdvancedOrthogonalityValidation {
    /// Basic orthogonality metrics
    pub basic_metrics: OrthogonalityMetrics,
    /// Bi-orthogonality validation (for non-orthogonal wavelets)
    pub biorthogonality_verified: bool,
    /// Cross-correlation matrix analysis
    pub correlation_matrix_analysis: CorrelationMatrixAnalysis,
    /// Coherence analysis
    pub coherence_analysis: CoherenceAnalysis,
}

/// Correlation matrix analysis
#[derive(Debug, Clone)]
pub struct CorrelationMatrixAnalysis {
    /// Maximum off-diagonal element
    pub max_off_diagonal: f64,
    /// Frobenius norm of off-diagonal part
    pub off_diagonal_frobenius_norm: f64,
    /// Condition number of correlation matrix
    pub condition_number: f64,
    /// Eigenvalue distribution
    pub eigenvalue_statistics: EigenvalueStatistics,
}

/// Eigenvalue statistics
#[derive(Debug, Clone)]
pub struct EigenvalueStatistics {
    pub min_eigenvalue: f64,
    pub max_eigenvalue: f64,
    pub eigenvalue_spread: f64,
    pub null_space_dimension: usize,
}

/// Coherence analysis for overcomplete representations
#[derive(Debug, Clone)]
pub struct CoherenceAnalysis {
    /// Mutual coherence (maximum correlation between different atoms)
    pub mutual_coherence: f64,
    /// Cumulative coherence
    pub cumulative_coherence: Array1<f64>,
    /// Coherence distribution statistics
    pub coherence_statistics: CoherenceStatistics,
}

/// Coherence statistics
#[derive(Debug, Clone)]
pub struct CoherenceStatistics {
    pub mean_coherence: f64,
    pub std_coherence: f64,
    pub median_coherence: f64,
    pub coherence_percentiles: Array1<f64>,
}

/// Energy conservation validation
#[derive(Debug, Clone)]
pub struct EnergyConservationValidation {
    /// Energy preservation ratio
    pub energy_ratio: f64,
    /// Energy distribution across subbands
    pub subband_energy_distribution: Array1<f64>,
    /// Energy concentration measure
    pub energy_concentration: f64,
    /// Energy leakage between subbands
    pub energy_leakage: f64,
}

/// Coefficient distribution analysis
#[derive(Debug, Clone)]
pub struct CoefficientDistributionAnalysis {
    /// Sparsity measures per subband
    pub sparsity_per_subband: Array1<f64>,
    /// Distribution types detected
    pub distribution_types: Vec<DistributionType>,
    /// Heavy-tail analysis
    pub heavy_tail_analysis: HeavyTailAnalysis,
    /// Anomaly detection results
    pub anomaly_detection: AnomalyDetectionResult,
}

/// Distribution types for coefficients
#[derive(Debug, Clone, PartialEq)]
pub enum DistributionType {
    Gaussian,
    Laplacian,
    GeneralizedGaussian { shape_parameter: f64 },
    HeavyTailed,
    Uniform,
    Unknown,
}

/// Heavy-tail analysis
#[derive(Debug, Clone)]
pub struct HeavyTailAnalysis {
    /// Tail index estimates
    pub tail_indices: Array1<f64>,
    /// Kurtosis values
    pub kurtosis_values: Array1<f64>,
    /// Heavy-tail test p-values
    pub heavy_tail_p_values: Array1<f64>,
}

/// Anomaly detection in coefficient patterns
#[derive(Debug, Clone)]
pub struct AnomalyDetectionResult {
    /// Anomalous coefficient locations
    pub anomaly_locations: Vec<(usize, usize)>, // (level, index)
    /// Anomaly scores
    pub anomaly_scores: Array1<f64>,
    /// Anomaly types detected
    pub anomaly_types: Vec<AnomalyType>,
}

/// Types of anomalies in wavelet coefficients
#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    OutlierCoefficient,
    UnexpectedSparsity,
    EnergyConcentration,
    StructuralAnomaly,
    NumericalInstability,
}

/// SIMD implementation validation
#[derive(Debug, Clone)]
pub struct SimdValidationResult {
    /// SIMD capabilities detected
    pub simd_capabilities: String,
    /// SIMD vs scalar accuracy comparison
    pub simd_scalar_accuracy: f64,
    /// SIMD operation correctness per function
    pub operation_correctness: HashMap<String, SimdCorrectnessResult>,
    /// SIMD performance validation
    pub performance_validation: SimdPerformanceValidation,
    /// Cross-architecture consistency
    pub architecture_consistency: ArchitectureConsistencyResult,
}

/// SIMD correctness result for individual operations
#[derive(Debug, Clone)]
pub struct SimdCorrectnessResult {
    pub function_name: String,
    pub max_error: f64,
    pub rms_error: f64,
    pub test_cases_passed: usize,
    pub test_cases_total: usize,
    pub numerical_stability_score: f64,
}

/// SIMD performance validation
#[derive(Debug, Clone)]
pub struct SimdPerformanceValidation {
    /// SIMD speedup factors per operation
    pub speedup_factors: HashMap<String, f64>,
    /// Memory bandwidth utilization
    pub memory_bandwidth_utilization: f64,
    /// Vectorization efficiency
    pub vectorization_efficiency: f64,
    /// Performance regression indicators
    pub performance_regressions: Vec<String>,
}

/// Architecture consistency results
#[derive(Debug, Clone)]
pub struct ArchitectureConsistencyResult {
    /// Results consistent across architectures
    pub is_consistent: bool,
    /// Maximum deviation between architectures
    pub max_deviation: f64,
    /// Architecture-specific results
    pub architecture_results: HashMap<String, f64>,
}

/// Cross-platform consistency validation
#[derive(Debug, Clone)]
pub struct PlatformConsistencyResult {
    /// Platforms tested
    pub platforms_tested: Vec<String>,
    /// Consistency verification
    pub is_consistent: bool,
    /// Maximum inter-platform deviation
    pub max_platform_deviation: f64,
    /// Platform-specific issues
    pub platform_issues: HashMap<String, Vec<String>>,
    /// Numerical precision comparison
    pub precision_comparison: PrecisionComparisonResult,
}

/// Numerical precision comparison
#[derive(Debug, Clone)]
pub struct PrecisionComparisonResult {
    /// Single vs double precision comparison
    pub single_double_deviation: f64,
    /// Extended precision validation
    pub extended_precision_verified: bool,
    /// Platform-specific precision issues
    pub precision_issues: Vec<String>,
}

/// Statistical validation of basis selection algorithms
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    /// Best basis selection consistency
    pub basis_selection_consistency: BasisSelectionConsistency,
    /// Cost function validation
    pub cost_function_validation: CostFunctionValidation,
    /// Statistical significance testing
    pub significance_testing: SignificanceTestingResult,
    /// Robustness analysis
    pub robustness_analysis: RobustnessAnalysisResult,
}

/// Basis selection consistency analysis
#[derive(Debug, Clone)]
pub struct BasisSelectionConsistency {
    /// Consistency across multiple runs
    pub multi_run_consistency: f64,
    /// Stability under noise
    pub noise_stability: f64,
    /// Sensitivity to initial conditions
    pub initial_condition_sensitivity: f64,
    /// Selection entropy measure
    pub selection_entropy: f64,
}

/// Cost function validation
#[derive(Debug, Clone)]
pub struct CostFunctionValidation {
    /// Cost function monotonicity
    pub monotonicity_verified: bool,
    /// Convexity analysis
    pub convexity_analysis: ConvexityAnalysisResult,
    /// Local minima detection
    pub local_minima_count: usize,
    /// Convergence analysis
    pub convergence_analysis: ConvergenceAnalysisResult,
}

/// Convexity analysis result
#[derive(Debug, Clone)]
pub struct ConvexityAnalysisResult {
    pub is_convex: bool,
    pub convexity_score: f64,
    pub non_convex_regions: Vec<(f64, f64)>,
}

/// Convergence analysis result
#[derive(Debug, Clone)]
pub struct ConvergenceAnalysisResult {
    pub convergence_rate: f64,
    pub iterations_to_convergence: usize,
    pub convergence_guaranteed: bool,
    pub stopping_criterion_analysis: StoppingCriterionAnalysis,
}

/// Stopping criterion analysis
#[derive(Debug, Clone)]
pub struct StoppingCriterionAnalysis {
    pub criterion_effectiveness: f64,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
    pub optimal_threshold: f64,
}

/// Statistical significance testing
#[derive(Debug, Clone)]
pub struct SignificanceTestingResult {
    /// Hypothesis testing results
    pub hypothesis_tests: Vec<HypothesisTestResult>,
    /// Multiple comparison corrections
    pub multiple_comparison_correction: MultipleComparisonResult,
    /// Power analysis
    pub power_analysis: PowerAnalysisResult,
}

/// Individual hypothesis test result
#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    pub test_name: String,
    pub null_hypothesis: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub effect_size: f64,
    pub confidence_interval: (f64, f64),
    pub rejected: bool,
}

/// Multiple comparison correction result
#[derive(Debug, Clone)]
pub struct MultipleComparisonResult {
    pub correction_method: String,
    pub adjusted_p_values: Array1<f64>,
    pub family_wise_error_rate: f64,
    pub false_discovery_rate: f64,
}

/// Statistical power analysis
#[derive(Debug, Clone)]
pub struct PowerAnalysisResult {
    pub statistical_power: f64,
    pub minimum_detectable_effect: f64,
    pub sample_size_recommendation: usize,
    pub power_curve: Array2<f64>, // effect_size vs power
}

/// Robustness analysis result
#[derive(Debug, Clone)]
pub struct RobustnessAnalysisResult {
    /// Robustness to noise
    pub noise_robustness: NoiseRobustnessResult,
    /// Robustness to parameter variations
    pub parameter_robustness: ParameterRobustnessResult,
    /// Breakdown point analysis
    pub breakdown_analysis: BreakdownAnalysisResult,
}

/// Noise robustness analysis
#[derive(Debug, Clone)]
pub struct NoiseRobustnessResult {
    /// Performance vs noise level
    pub noise_performance_curve: Array2<f64>,
    /// Noise threshold for acceptable performance
    pub noise_threshold: f64,
    /// Robustness score
    pub robustness_score: f64,
}

/// Parameter robustness analysis
#[derive(Debug, Clone)]
pub struct ParameterRobustnessResult {
    /// Sensitivity to each parameter
    pub parameter_sensitivities: HashMap<String, f64>,
    /// Parameter stability regions
    pub stability_regions: HashMap<String, (f64, f64)>,
    /// Most critical parameters
    pub critical_parameters: Vec<String>,
}

/// Breakdown analysis result
#[derive(Debug, Clone)]
pub struct BreakdownAnalysisResult {
    /// Breakdown point
    pub breakdown_point: f64,
    /// Failure modes identified
    pub failure_modes: Vec<FailureMode>,
    /// Recovery strategies
    pub recovery_strategies: Vec<String>,
}

/// Failure mode types
#[derive(Debug, Clone, PartialEq)]
pub enum FailureMode {
    NumericalInstability,
    MemoryExhaustion,
    PerformanceDegradation,
    QualityLoss,
    Convergence,
}

/// Performance regression analysis
#[derive(Debug, Clone)]
pub struct PerformanceRegressionResult {
    /// Historical performance comparison
    pub historical_comparison: HistoricalComparisonResult,
    /// Performance benchmarks
    pub benchmarks: PerformanceBenchmarkResult,
    /// Scalability analysis
    pub scalability_analysis: ScalabilityAnalysisResult,
    /// Resource utilization analysis
    pub resource_utilization: ResourceUtilizationResult,
}

/// Historical performance comparison
#[derive(Debug, Clone)]
pub struct HistoricalComparisonResult {
    /// Performance relative to baseline
    pub relative_performance: f64,
    /// Performance trend analysis
    pub trend_analysis: TrendAnalysisResult,
    /// Regression detection
    pub regressions_detected: Vec<PerformanceRegression>,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysisResult {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_significance: f64,
    pub projection: f64, // Projected performance for next version
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Performance regression
#[derive(Debug, Clone)]
pub struct PerformanceRegression {
    pub metric_name: String,
    pub regression_magnitude: f64,
    pub suspected_cause: String,
    pub severity: RegressionSeverity,
}

/// Regression severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum RegressionSeverity {
    Critical,
    Major,
    Minor,
    Negligible,
}

/// Performance benchmark result
#[derive(Debug, Clone)]
pub struct PerformanceBenchmarkResult {
    /// Benchmark suite results
    pub benchmark_results: HashMap<String, BenchmarkResult>,
    /// Comparative analysis
    pub comparative_analysis: ComparativeAnalysisResult,
    /// Performance profile
    pub performance_profile: PerformanceProfile,
}

/// Individual benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub test_name: String,
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub throughput: f64,
    pub efficiency_score: f64,
}

/// Comparative analysis against other implementations
#[derive(Debug, Clone)]
pub struct ComparativeAnalysisResult {
    /// Relative performance ranking
    pub performance_ranking: usize,
    /// Performance gaps
    pub performance_gaps: HashMap<String, f64>,
    /// Strengths and weaknesses
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
}

/// Performance profile
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Computational complexity
    pub time_complexity: f64,
    pub space_complexity: f64,
    /// Bottleneck analysis
    pub bottlenecks: Vec<String>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
}

/// Scalability analysis result
#[derive(Debug, Clone)]
pub struct ScalabilityAnalysisResult {
    /// Scaling behavior
    pub scaling_behavior: ScalingBehavior,
    /// Parallel efficiency
    pub parallel_efficiency: f64,
    /// Memory scaling
    pub memory_scaling: f64,
    /// Scalability limits
    pub scalability_limits: ScalabilityLimits,
}

/// Scaling behavior characterization
#[derive(Debug, Clone)]
pub struct ScalingBehavior {
    pub time_scaling_exponent: f64,
    pub memory_scaling_exponent: f64,
    pub parallel_scaling_efficiency: f64,
    pub scaling_quality: ScalingQuality,
}

/// Scaling quality assessment
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingQuality {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Unacceptable,
}

/// Scalability limits
#[derive(Debug, Clone)]
pub struct ScalabilityLimits {
    pub maximum_signal_size: Option<usize>,
    pub maximum_decomposition_level: Option<usize>,
    pub memory_limit_factor: f64,
    pub performance_limit_factor: f64,
}

/// Resource utilization analysis
#[derive(Debug, Clone)]
pub struct ResourceUtilizationResult {
    /// CPU utilization
    pub cpu_utilization: CpuUtilizationResult,
    /// Memory utilization
    pub memory_utilization: MemoryUtilizationResult,
    /// Cache utilization
    pub cache_utilization: CacheUtilizationResult,
    /// I/O utilization
    pub io_utilization: IoUtilizationResult,
}

/// CPU utilization analysis
#[derive(Debug, Clone)]
pub struct CpuUtilizationResult {
    pub average_utilization: f64,
    pub peak_utilization: f64,
    pub core_balance: f64,
    pub instruction_mix: InstructionMixResult,
}

/// Instruction mix analysis
#[derive(Debug, Clone)]
pub struct InstructionMixResult {
    pub arithmetic_operations: f64,
    pub memory_operations: f64,
    pub control_operations: f64,
    pub vectorized_operations: f64,
}

/// Memory utilization analysis
#[derive(Debug, Clone)]
pub struct MemoryUtilizationResult {
    pub peak_memory_usage: f64,
    pub average_memory_usage: f64,
    pub memory_fragmentation: f64,
    pub allocation_efficiency: f64,
}

/// Cache utilization analysis
#[derive(Debug, Clone)]
pub struct CacheUtilizationResult {
    pub l1_cache_hit_rate: f64,
    pub l2_cache_hit_rate: f64,
    pub l3_cache_hit_rate: f64,
    pub cache_miss_penalty: f64,
}

/// I/O utilization analysis
#[derive(Debug, Clone)]
pub struct IoUtilizationResult {
    pub read_throughput: f64,
    pub write_throughput: f64,
    pub io_wait_time: f64,
    pub bandwidth_utilization: f64,
}

/// Memory safety validation
#[derive(Debug, Clone)]
pub struct MemorySafetyResult {
    /// Memory leaks detected
    pub memory_leaks_detected: usize,
    /// Buffer overflow/underflow detection
    pub buffer_safety_verified: bool,
    /// Use-after-free detection
    pub use_after_free_detected: usize,
    /// Double-free detection
    pub double_free_detected: usize,
    /// Memory alignment verification
    pub alignment_verified: bool,
    /// Memory safety score
    pub safety_score: f64,
}

/// Real-time processing validation
#[derive(Debug, Clone)]
pub struct RealtimeValidationResult {
    /// Latency analysis
    pub latency_analysis: LatencyAnalysisResult,
    /// Jitter analysis
    pub jitter_analysis: JitterAnalysisResult,
    /// Throughput analysis
    pub throughput_analysis: ThroughputAnalysisResult,
    /// Quality under real-time constraints
    pub realtime_quality: RealtimeQualityResult,
}

/// Latency analysis
#[derive(Debug, Clone)]
pub struct LatencyAnalysisResult {
    pub average_latency_ms: f64,
    pub maximum_latency_ms: f64,
    pub latency_percentiles: Array1<f64>,
    pub latency_target_met: bool,
}

/// Jitter analysis
#[derive(Debug, Clone)]
pub struct JitterAnalysisResult {
    pub average_jitter_ms: f64,
    pub maximum_jitter_ms: f64,
    pub jitter_stability: f64,
    pub jitter_distribution: JitterDistribution,
}

/// Jitter distribution characterization
#[derive(Debug, Clone)]
pub struct JitterDistribution {
    pub distribution_type: String,
    pub parameters: HashMap<String, f64>,
    pub outlier_rate: f64,
}

/// Throughput analysis
#[derive(Debug, Clone)]
pub struct ThroughputAnalysisResult {
    pub average_throughput: f64,
    pub peak_throughput: f64,
    pub throughput_stability: f64,
    pub bottleneck_analysis: Vec<String>,
}

/// Real-time quality assessment
#[derive(Debug, Clone)]
pub struct RealtimeQualityResult {
    pub quality_degradation: f64,
    pub quality_consistency: f64,
    pub adaptive_quality_control: bool,
    pub quality_vs_latency_tradeoff: f64,
}

/// Overall validation status
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationStatus {
    Pass,
    PassWithWarnings,
    Fail,
    Incomplete,
}

/// Configuration for advanced-comprehensive WPT validation
#[derive(Debug, Clone)]
pub struct AdvancedWptValidationConfig {
    /// Enable mathematical property validation
    pub validate_mathematical_properties: bool,
    /// Enable SIMD validation
    pub validate_simd: bool,
    /// Enable cross-platform validation
    pub validate_cross_platform: bool,
    /// Enable statistical validation
    pub validate_statistical: bool,
    /// Enable performance regression testing
    pub validate_performance_regression: bool,
    /// Enable memory safety validation
    pub validate_memory_safety: bool,
    /// Enable real-time validation
    pub validate_realtime: bool,
    /// Numerical tolerance for comparisons
    pub tolerance: f64,
    /// Number of Monte Carlo samples for statistical tests
    pub monte_carlo_samples: usize,
    /// Test signal configurations
    pub test_signals: Vec<TestSignalConfig>,
    /// Wavelet types to test
    pub wavelets_to_test: Vec<Wavelet>,
    /// Maximum decomposition levels to test
    pub max_levels_to_test: Vec<usize>,
}

/// Test signal configuration
#[derive(Debug, Clone)]
pub struct TestSignalConfig {
    pub signal_type: TestSignalType,
    pub length: usize,
    pub parameters: HashMap<String, f64>,
}

/// Test signal types
#[derive(Debug, Clone, PartialEq)]
pub enum TestSignalType {
    Sinusoid,
    Chirp,
    WhiteNoise,
    PinkNoise,
    Impulse,
    Step,
    Polynomial,
    Piecewise,
    Fractal,
    Composite,
}

/// Reconstruction error structure
#[derive(Debug, Clone)]
pub struct ReconstructionError {
    pub max_error: f64,
    pub rms_error: f64,
}

impl Default for AdvancedWptValidationConfig {
    fn default() -> Self {
        Self {
            validate_mathematical_properties: true,
            validate_simd: true,
            validate_cross_platform: true,
            validate_statistical: true,
            validate_performance_regression: true,
            validate_memory_safety: true,
            validate_realtime: false,
            tolerance: 1e-12,
            monte_carlo_samples: 10000,
            test_signals: vec![
                TestSignalConfig {
                    signal_type: TestSignalType::Sinusoid,
                    length: 1024,
                    parameters: [("frequency".to_string(), 10.0)].iter().cloned().collect(),
                },
                TestSignalConfig {
                    signal_type: TestSignalType::Chirp,
                    length: 2048,
                    parameters: [
                        ("start_freq".to_string(), 1.0),
                        ("end_freq".to_string(), 50.0),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                },
                TestSignalConfig {
                    signal_type: TestSignalType::WhiteNoise,
                    length: 4096,
                    parameters: [("variance".to_string(), 1.0)].iter().cloned().collect(),
                },
            ],
            wavelets_to_test: vec![
                Wavelet::DB(4),
                Wavelet::BiorNrNd { nr: 2, nd: 2 },
                Wavelet::Coif(2),
            ],
            max_levels_to_test: vec![3, 5, 7],
        }
    }
}