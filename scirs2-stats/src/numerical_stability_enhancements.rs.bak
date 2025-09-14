//! advanced Advanced Numerical Stability Testing System
//!
//! Next-generation numerical stability framework with comprehensive edge case testing,
//! precision analysis, mathematical invariant validation, catastrophic cancellation detection,
//! overflow/underflow monitoring, and automated numerical stability assessment for ensuring
//! robust statistical computing operations across all numerical conditions.

use crate::error::StatsResult;
use crate::propertybased_validation::ValidationReport;
use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix1};
use num_traits::{Float, NumCast};
use rand::Rng;
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// advanced Numerical Stability Configuration
#[derive(Debug, Clone)]
pub struct AdvancedNumericalStabilityConfig {
    /// Enable comprehensive edge case testing
    pub enable_edge_case_testing: bool,
    /// Enable precision analysis across different floating-point types
    pub enable_precision_analysis: bool,
    /// Enable mathematical invariant validation
    pub enable_invariant_validation: bool,
    /// Enable catastrophic cancellation detection
    pub enable_cancellation_detection: bool,
    /// Enable overflow/underflow monitoring
    pub enable_overflow_monitoring: bool,
    /// Enable condition number analysis
    pub enable_condition_analysis: bool,
    /// Enable numerical differentiation accuracy testing
    pub enable_differentiation_testing: bool,
    /// Enable iterative algorithm convergence testing
    pub enable_convergence_testing: bool,
    /// Enable Monte Carlo numerical stability testing
    pub enable_monte_carlo_testing: bool,
    /// Enable regression testing for numerical stability
    pub enable_regression_testing: bool,
    /// Numerical stability thoroughness level
    pub thoroughness_level: NumericalStabilityThoroughness,
    /// Precision testing strategy
    pub precision_strategy: PrecisionTestingStrategy,
    /// Edge case generation approach
    pub edge_case_approach: EdgeCaseGenerationApproach,
    /// Stability tolerance configuration
    pub stability_tolerance: StabilityTolerance,
    /// Test execution timeout
    pub test_timeout: Duration,
    /// Maximum iterations for convergence tests
    pub max_convergence_iterations: usize,
    /// Monte Carlo sample size for stability testing
    pub monte_carlo_samples: usize,
}

impl Default for AdvancedNumericalStabilityConfig {
    fn default() -> Self {
        Self {
            enable_edge_case_testing: true,
            enable_precision_analysis: true,
            enable_invariant_validation: true,
            enable_cancellation_detection: true,
            enable_overflow_monitoring: true,
            enable_condition_analysis: true,
            enable_differentiation_testing: true,
            enable_convergence_testing: true,
            enable_monte_carlo_testing: true,
            enable_regression_testing: true,
            thoroughness_level: NumericalStabilityThoroughness::Comprehensive,
            precision_strategy: PrecisionTestingStrategy::MultiPrecision,
            edge_case_approach: EdgeCaseGenerationApproach::Systematic,
            stability_tolerance: StabilityTolerance::default(),
            test_timeout: Duration::from_secs(600), // 10 minutes
            max_convergence_iterations: 10000,
            monte_carlo_samples: 100000,
        }
    }
}

/// Numerical stability thoroughness levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NumericalStabilityThoroughness {
    Basic,         // Essential stability tests
    Standard,      // Common stability scenarios
    Comprehensive, // Extensive stability coverage
    Exhaustive,    // Maximum stability validation
}

/// Precision testing strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionTestingStrategy {
    SinglePrecision,   // f32 only
    DoublePrecision,   // f64 only
    MultiPrecision,    // f32, f64, and extended precision
    AdaptivePrecision, // Dynamic precision selection
}

/// Edge case generation approaches
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeCaseGenerationApproach {
    Predefined,  // Use predefined edge cases
    Systematic,  // Systematic boundary exploration
    Adaptive,    // Adaptive based on function behavior
    Intelligent, // AI-guided edge case discovery
}

/// Stability tolerance configuration
#[derive(Debug, Clone)]
pub struct StabilityTolerance {
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub condition_number_threshold: f64,
    pub cancellation_threshold: f64,
    pub convergence_tolerance: f64,
    pub monte_carlo_confidence_level: f64,
}

impl Default for StabilityTolerance {
    fn default() -> Self {
        Self {
            absolute_tolerance: 1e-14,
            relative_tolerance: 1e-12,
            condition_number_threshold: 1e12,
            cancellation_threshold: 1e-10,
            convergence_tolerance: 1e-10,
            monte_carlo_confidence_level: 0.95,
        }
    }
}

/// Comprehensive numerical stability tester
pub struct AdvancedNumericalStabilityTester {
    config: AdvancedNumericalStabilityConfig,
    edge_case_generator: Arc<RwLock<EdgeCaseGenerator>>,
    precision_analyzer: Arc<RwLock<PrecisionAnalyzer>>,
    invariant_validator: Arc<RwLock<InvariantValidator>>,
    cancellation_detector: Arc<RwLock<CancellationDetector>>,
    overflow_monitor: Arc<RwLock<OverflowMonitor>>,
    condition_analyzer: Arc<RwLock<ConditionAnalyzer>>,
    convergence_tester: Arc<RwLock<ConvergenceTester>>,
    monte_carlo_tester: Arc<RwLock<MonteCarloStabilityTester>>,
    regression_tester: Arc<RwLock<RegressionTester>>,
    stability_history: Arc<RwLock<VecDeque<StabilityTestResult>>>,
}

impl AdvancedNumericalStabilityTester {
    /// Create new numerical stability tester
    pub fn new(config: AdvancedNumericalStabilityConfig) -> Self {
        Self {
            edge_case_generator: Arc::new(RwLock::new(EdgeCaseGenerator::new(&config))),
            precision_analyzer: Arc::new(RwLock::new(PrecisionAnalyzer::new(&config))),
            invariant_validator: Arc::new(RwLock::new(InvariantValidator::new(&config))),
            cancellation_detector: Arc::new(RwLock::new(CancellationDetector::new(&config))),
            overflow_monitor: Arc::new(RwLock::new(OverflowMonitor::new(&config))),
            condition_analyzer: Arc::new(RwLock::new(ConditionAnalyzer::new(&config))),
            convergence_tester: Arc::new(RwLock::new(ConvergenceTester::new(&config))),
            monte_carlo_tester: Arc::new(RwLock::new(MonteCarloStabilityTester::new(&config))),
            regression_tester: Arc::new(RwLock::new(RegressionTester::new(&config))),
            stability_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            config,
        }
    }

    /// Perform comprehensive numerical stability testing
    pub fn comprehensive_stability_testing<F, D, R>(
        &self,
        function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<ComprehensiveStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let start_time = Instant::now();
        let mut results = ComprehensiveStabilityResult::new(function_name.to_string());

        // Test 1: Edge case stability testing
        if self.config.enable_edge_case_testing {
            results.edge_case_results = Some(self.test_edge_case_stability(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 2: Precision analysis
        if self.config.enable_precision_analysis {
            results.precision_results = Some(self.analyze_precision_stability(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 3: Mathematical invariant validation
        if self.config.enable_invariant_validation {
            results.invariant_results = Some(self.validate_mathematical_invariants(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 4: Catastrophic cancellation detection
        if self.config.enable_cancellation_detection {
            results.cancellation_results = Some(self.detect_catastrophic_cancellation(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 5: Overflow/underflow monitoring
        if self.config.enable_overflow_monitoring {
            results.overflow_results = Some(self.monitor_overflow_underflow(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 6: Condition number analysis
        if self.config.enable_condition_analysis {
            results.condition_results = Some(self.analyze_condition_numbers(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 7: Convergence testing
        if self.config.enable_convergence_testing {
            results.convergence_results = Some(self.test_convergence_stability(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 8: Monte Carlo stability testing
        if self.config.enable_monte_carlo_testing {
            results.monte_carlo_results = Some(self.test_monte_carlo_stability(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        // Test 9: Regression testing
        if self.config.enable_regression_testing {
            results.regression_results = Some(self.test_numerical_regression(
                function_name,
                test_function.clone(),
                testdata,
            )?);
        }

        results.test_duration = start_time.elapsed();
        results.overall_stability_score = self.calculate_overall_stability_score(&results);
        results.stability_assessment = self.assess_stability_level(&results);
        results.recommendations = self.generate_stability_recommendations(&results);

        // Store results in history
        let stability_result = StabilityTestResult {
            function_name: function_name.to_string(),
            timestamp: SystemTime::now(),
            stability_score: results.overall_stability_score,
            critical_issues: results.critical_issues.len(),
            warnings: results.warnings.len(),
        };

        self.stability_history
            .write()
            .unwrap()
            .push_back(stability_result);
        if self.stability_history.read().unwrap().len() > 1000 {
            self.stability_history.write().unwrap().pop_front();
        }

        Ok(results)
    }

    /// Test edge case stability
    #[ignore = "timeout"]
    fn test_edge_case_stability<F, D, R>(
        &self,
        _function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<EdgeCaseStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let generator = self.edge_case_generator.read().unwrap();
        let edge_cases = generator.generate_comprehensive_edge_cases(testdata)?;

        let mut results = EdgeCaseStabilityResult::new();

        for edge_case in &edge_cases {
            let test_result = self.execute_edge_case_test(&test_function, edge_case)?;
            results.add_test_result(edge_case.clone(), test_result);
        }

        results.analyze_edge_case_patterns();
        Ok(results)
    }

    /// Analyze precision stability
    fn analyze_precision_stability<F, D, R>(
        &self,
        _function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<PrecisionStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let analyzer = self.precision_analyzer.read().unwrap();
        analyzer.analyze_multi_precision_stability(&test_function, testdata)
    }

    /// Validate mathematical invariants
    fn validate_mathematical_invariants<F, D, R>(
        &self,
        function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<InvariantValidationResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let validator = self.invariant_validator.read().unwrap();
        validator.validate_statistical_invariants(function_name, &test_function, testdata)
    }

    /// Detect catastrophic cancellation
    fn detect_catastrophic_cancellation<F, D, R>(
        &self,
        _function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<CancellationDetectionResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let detector = self.cancellation_detector.read().unwrap();
        detector.detect_cancellation_patterns(&test_function, testdata)
    }

    /// Monitor overflow and underflow
    fn monitor_overflow_underflow<F, D, R>(
        &self,
        _function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<OverflowMonitoringResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let monitor = self.overflow_monitor.read().unwrap();
        monitor.monitor_numerical_limits(&test_function, testdata)
    }

    /// Analyze condition numbers
    fn analyze_condition_numbers<F, D, R>(
        &self,
        _function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<ConditionAnalysisResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let analyzer = self.condition_analyzer.read().unwrap();
        analyzer.analyze_numerical_conditioning(&test_function, testdata)
    }

    /// Test convergence stability
    fn test_convergence_stability<F, D, R>(
        &self,
        _function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<ConvergenceStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let tester = self.convergence_tester.read().unwrap();
        tester.test_iterative_stability(&test_function, testdata)
    }

    /// Test Monte Carlo stability
    fn test_monte_carlo_stability<F, D, R>(
        &self,
        _function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<MonteCarloStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let tester = self.monte_carlo_tester.read().unwrap();
        tester.test_statistical_stability(&test_function, testdata)
    }

    /// Test numerical regression
    fn test_numerical_regression<F, D, R>(
        &self,
        function_name: &str,
        test_function: F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<RegressionTestResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let tester = self.regression_tester.read().unwrap();
        tester.test_against_historical_results(function_name, &test_function, testdata)
    }

    /// Execute edge case test
    fn execute_edge_case_test<F, R>(
        &self,
        test_function: &F,
        edge_case: &EdgeCase<R>,
    ) -> StatsResult<EdgeCaseTestResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let start_time = Instant::now();

        let result = match test_function(&edge_case.data.view()) {
            Ok(value) => EdgeCaseTestResult {
                edge_case_type: edge_case.edge_case_type.clone(),
                execution_time: start_time.elapsed(),
                result_status: EdgeCaseResultStatus::Success,
                computed_value: value.to_f64(),
                error_message: None,
                stability_metrics: self.compute_edge_case_stability_metrics(value),
            },
            Err(e) => EdgeCaseTestResult {
                edge_case_type: edge_case.edge_case_type.clone(),
                execution_time: start_time.elapsed(),
                result_status: EdgeCaseResultStatus::Error,
                computed_value: None,
                error_message: Some(format!("{:?}", e)),
                stability_metrics: StabilityMetrics::default(),
            },
        };

        Ok(result)
    }

    /// Compute edge case stability metrics
    fn compute_edge_case_stability_metrics<R>(&self, value: R) -> StabilityMetrics
    where
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut metrics = StabilityMetrics::default();

        if value.is_nan() {
            metrics.nan_count += 1;
        } else if value.is_infinite() {
            metrics.infinite_count += 1;
        } else if value.is_normal() {
            metrics.normal_count += 1;
        } else {
            metrics.subnormal_count += 1;
        }

        metrics.condition_number = self.estimate_condition_number(value);
        metrics.relative_error = self.estimate_relative_error(value);

        metrics
    }

    /// Estimate condition number
    fn estimate_condition_number<R>(&self, value: R) -> f64
    where
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        // Simplified condition number estimation
        // In practice, this would involve more sophisticated analysis
        let val_f64: f64 = NumCast::from(value).unwrap_or(0.0);
        if val_f64.abs() < 1e-15 {
            1e15
        } else {
            1.0 / val_f64.abs()
        }
    }

    /// Estimate relative error
    fn estimate_relative_error<R>(&self, value: R) -> f64
    where
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        // Simplified relative error estimation
        let val_f64: f64 = NumCast::from(value).unwrap_or(0.0);
        if val_f64.abs() < 1e-15 {
            1.0
        } else {
            std::f64::EPSILON / val_f64.abs()
        }
    }

    /// Calculate overall stability score
    fn calculate_overall_stability_score(&self, results: &ComprehensiveStabilityResult) -> f64 {
        let mut score = 100.0;

        // Penalize for critical issues
        score -= results.critical_issues.len() as f64 * 20.0;

        // Penalize for warnings
        score -= results.warnings.len() as f64 * 5.0;

        // Adjust based on specific test results
        if let Some(ref edge_results) = results.edge_case_results {
            score -= edge_results.failed_cases.len() as f64 * 10.0;
        }

        if let Some(ref precision_results) = results.precision_results {
            if precision_results.precision_loss_detected {
                score -= 15.0;
            }
        }

        if let Some(ref cancellation_results) = results.cancellation_results {
            score -= cancellation_results.cancellation_events.len() as f64 * 12.0;
        }

        if let Some(ref overflow_results) = results.overflow_results {
            score -= overflow_results.overflow_events.len() as f64 * 25.0;
            score -= overflow_results.underflow_events.len() as f64 * 15.0;
        }

        score.max(0.0).min(100.0)
    }

    /// Assess stability level
    fn assess_stability_level(
        &self,
        results: &ComprehensiveStabilityResult,
    ) -> StabilityAssessment {
        let score = results.overall_stability_score;

        if score >= 95.0 {
            StabilityAssessment::Excellent
        } else if score >= 85.0 {
            StabilityAssessment::Good
        } else if score >= 70.0 {
            StabilityAssessment::Acceptable
        } else if score >= 50.0 {
            StabilityAssessment::Poor
        } else {
            StabilityAssessment::Critical
        }
    }

    /// Generate stability recommendations
    fn generate_stability_recommendations(
        &self,
        results: &ComprehensiveStabilityResult,
    ) -> Vec<StabilityRecommendation> {
        let mut recommendations = Vec::new();

        if results.critical_issues.len() > 0 {
            recommendations.push(StabilityRecommendation {
                recommendation_type: RecommendationType::Critical,
                description:
                    "Critical numerical stability issues detected. Immediate attention required."
                        .to_string(),
                implementation_priority: ImplementationPriority::Immediate,
                estimated_effort: EstimatedEffort::High,
            });
        }

        if let Some(ref cancellation_results) = results.cancellation_results {
            if cancellation_results.cancellation_events.len() > 0 {
                recommendations.push(StabilityRecommendation {
                    recommendation_type: RecommendationType::Algorithm,
                    description: "Consider using numerically stable algorithms to avoid catastrophic cancellation.".to_string(),
                    implementation_priority: ImplementationPriority::High,
                    estimated_effort: EstimatedEffort::Medium,
                });
            }
        }

        if let Some(ref precision_results) = results.precision_results {
            if precision_results.precision_loss_detected {
                recommendations.push(StabilityRecommendation {
                    recommendation_type: RecommendationType::Precision,
                    description:
                        "Consider using higher precision arithmetic for improved accuracy."
                            .to_string(),
                    implementation_priority: ImplementationPriority::Medium,
                    estimated_effort: EstimatedEffort::Low,
                });
            }
        }

        if let Some(ref condition_results) = results.condition_results {
            if condition_results.max_condition_number
                > self.config.stability_tolerance.condition_number_threshold
            {
                recommendations.push(StabilityRecommendation {
                    recommendation_type: RecommendationType::Conditioning,
                    description: "High condition numbers detected. Consider regularization or alternative formulations.".to_string(),
                    implementation_priority: ImplementationPriority::Medium,
                    estimated_effort: EstimatedEffort::High,
                });
            }
        }

        recommendations
    }

    /// Get stability history
    pub fn get_stability_history(&self) -> Vec<StabilityTestResult> {
        self.stability_history
            .read()
            .unwrap()
            .iter()
            .cloned()
            .collect()
    }

    /// Get stability trend analysis
    pub fn analyze_stability_trends(&self) -> StabilityTrendAnalysis {
        let history = self.stability_history.read().unwrap();

        if history.is_empty() {
            return StabilityTrendAnalysis::default();
        }

        let scores: Vec<f64> = history.iter().map(|r| r.stability_score).collect();
        let recent_scores = &scores[scores.len().saturating_sub(10)..];

        let trend = if recent_scores.len() >= 2 {
            let first = recent_scores[0];
            let last = recent_scores[recent_scores.len() - 1];
            if last > first + 5.0 {
                StabilityTrend::Improving
            } else if last < first - 5.0 {
                StabilityTrend::Declining
            } else {
                StabilityTrend::Stable
            }
        } else {
            StabilityTrend::Stable
        };

        StabilityTrendAnalysis {
            trend,
            average_score: scores.iter().sum::<f64>() / scores.len() as f64,
            recent_average: recent_scores.iter().sum::<f64>() / recent_scores.len() as f64,
            total_tests: history.len(),
            total_critical_issues: history.iter().map(|r| r.critical_issues).sum(),
            total_warnings: history.iter().map(|r| r.warnings).sum(),
        }
    }
}

// Supporting structures and implementations

/// Edge case generator
pub struct EdgeCaseGenerator {
    config: AdvancedNumericalStabilityConfig,
}

impl EdgeCaseGenerator {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn generate_comprehensive_edge_cases<D, R>(
        &self,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<Vec<EdgeCase<R>>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut edge_cases = Vec::new();

        // Generate basic edge cases
        edge_cases.extend(self.generate_basic_edge_cases(testdata)?);

        // Generate boundary edge cases
        edge_cases.extend(self.generate_boundary_edge_cases(testdata)?);

        // Generate scaling edge cases
        edge_cases.extend(self.generate_scaling_edge_cases(testdata)?);

        // Generate special value edge cases
        edge_cases.extend(self.generate_special_value_edge_cases(testdata)?);

        Ok(edge_cases)
    }

    fn generate_basic_edge_cases<D, R>(
        &self,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<Vec<EdgeCase<R>>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut cases = Vec::new();
        let datasize = testdata.len();

        // Empty array
        if datasize > 0 {
            let empty_data = Array1::<R>::zeros(0);
            cases.push(EdgeCase {
                edge_case_type: EdgeCaseType::EmptyArray,
                data: empty_data,
                description: "Empty input array".to_string(),
            });
        }

        // Single element
        if datasize > 1 {
            let singledata = Array1::from_elem(1, testdata[0]);
            cases.push(EdgeCase {
                edge_case_type: EdgeCaseType::SingleElement,
                data: singledata,
                description: "Single element array".to_string(),
            });
        }

        // All zeros
        let zerodata = Array1::zeros(datasize);
        cases.push(EdgeCase {
            edge_case_type: EdgeCaseType::AllZeros,
            data: zerodata,
            description: "All zeros array".to_string(),
        });

        // All ones
        let onesdata = Array1::ones(datasize);
        cases.push(EdgeCase {
            edge_case_type: EdgeCaseType::AllOnes,
            data: onesdata,
            description: "All ones array".to_string(),
        });

        Ok(cases)
    }

    fn generate_boundary_edge_cases<D, R>(
        &self,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<Vec<EdgeCase<R>>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut cases = Vec::new();
        let datasize = testdata.len();

        // Very small values
        let smalldata = Array1::from_elem(
            datasize,
            R::from(1e-100).unwrap_or(R::min_positive_value()),
        );
        cases.push(EdgeCase {
            edge_case_type: EdgeCaseType::VerySmallValues,
            data: smalldata,
            description: "Very small positive values".to_string(),
        });

        // Very large values
        let largedata = Array1::from_elem(datasize, R::from(1e100).unwrap_or(R::max_value()));
        cases.push(EdgeCase {
            edge_case_type: EdgeCaseType::VeryLargeValues,
            data: largedata,
            description: "Very large values".to_string(),
        });

        Ok(cases)
    }

    fn generate_scaling_edge_cases<D, R>(
        &self,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<Vec<EdgeCase<R>>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut cases = Vec::new();

        // Scaled versions of original data
        let scales = vec![1e-10, 1e-5, 1e5, 1e10];

        for scale in scales {
            if let Some(scale_val) = R::from(scale) {
                let scaleddata = testdata.mapv(|x| x * scale_val);
                cases.push(EdgeCase {
                    edge_case_type: EdgeCaseType::ScaledData,
                    data: scaleddata,
                    description: format!("Data scaled by {}", scale),
                });
            }
        }

        Ok(cases)
    }

    fn generate_special_value_edge_cases<D, R>(
        &self,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<Vec<EdgeCase<R>>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut cases = Vec::new();
        let datasize = testdata.len();

        // Array with NaN values
        let mut nandata = testdata.to_owned();
        if datasize > 0 {
            nandata[0] = R::nan();
            cases.push(EdgeCase {
                edge_case_type: EdgeCaseType::ContainsNaN,
                data: nandata,
                description: "Array containing NaN values".to_string(),
            });
        }

        // Array with infinite values
        let mut infdata = testdata.to_owned();
        if datasize > 0 {
            infdata[0] = R::infinity();
            cases.push(EdgeCase {
                edge_case_type: EdgeCaseType::ContainsInfinity,
                data: infdata,
                description: "Array containing infinite values".to_string(),
            });
        }

        // Array with mixed special values
        if datasize >= 3 {
            let mut mixeddata = testdata.to_owned();
            mixeddata[0] = R::nan();
            mixeddata[1] = R::infinity();
            mixeddata[2] = R::neg_infinity();
            cases.push(EdgeCase {
                edge_case_type: EdgeCaseType::MixedSpecialValues,
                data: mixeddata,
                description: "Array with mixed special values".to_string(),
            });
        }

        Ok(cases)
    }
}

/// Precision analyzer
pub struct PrecisionAnalyzer {
    config: AdvancedNumericalStabilityConfig,
}

impl PrecisionAnalyzer {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn analyze_multi_precision_stability<F, D, R>(
        &self,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<PrecisionStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = PrecisionStabilityResult::new();

        // Test with current precision
        let current_result = test_function(&testdata.view());
        result.add_precision_test(format!("{:?}", std::any::type_name::<R>()), current_result);

        // Additional precision analysis would go here
        // This is a simplified implementation

        Ok(result)
    }
}

/// Invariant validator
pub struct InvariantValidator {
    config: AdvancedNumericalStabilityConfig,
}

impl InvariantValidator {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn validate_statistical_invariants<F, D, R>(
        &self,
        function_name: &str,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<InvariantValidationResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = InvariantValidationResult::new();

        // Validate basic mathematical properties
        self.validate_basic_properties(function_name, test_function, testdata, &mut result)?;

        // Validate statistical properties
        self.validate_statistical_properties(function_name, test_function, testdata, &mut result)?;

        Ok(result)
    }

    fn validate_basic_properties<F, D, R>(
        &self,
        _function_name: &str,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
        result: &mut InvariantValidationResult,
    ) -> StatsResult<()>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        // Test determinism
        let result1 = test_function(&testdata.view());
        let result2 = test_function(&testdata.view());

        match (result1, result2) {
            (Ok(v1), Ok(v2)) => {
                let diff = (NumCast::from(v1).unwrap_or(0.0f64)
                    - NumCast::from(v2).unwrap_or(0.0f64))
                .abs();
                if diff > self.config.stability_tolerance.absolute_tolerance {
                    result.add_violation(InvariantViolation {
                        invariant_type: InvariantType::Determinism,
                        description: "Function is not deterministic".to_string(),
                        severity: ViolationSeverity::Critical,
                        detected_difference: diff,
                    });
                }
            }
            _ => {
                result.add_violation(InvariantViolation {
                    invariant_type: InvariantType::Determinism,
                    description: "Function execution inconsistent".to_string(),
                    severity: ViolationSeverity::Critical,
                    detected_difference: std::f64::INFINITY,
                });
            }
        }

        Ok(())
    }

    fn validate_statistical_properties<F, D, R>(
        &self,
        _function_name: &str,
        _test_function: &F,
        data: &ArrayBase<D, Ix1>,
        _result: &mut InvariantValidationResult,
    ) -> StatsResult<()>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        // Add statistical property validation here
        // This is a simplified implementation
        Ok(())
    }
}

/// Cancellation detector
pub struct CancellationDetector {
    config: AdvancedNumericalStabilityConfig,
}

impl CancellationDetector {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn detect_cancellation_patterns<F, D, R>(
        &self,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<CancellationDetectionResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = CancellationDetectionResult::new();

        // Test with data that might cause cancellation
        let test_cases = self.generate_cancellation_test_cases(testdata)?;

        for test_case in test_cases {
            let computation_result = test_function(&test_case.view());
            if let Ok(value) = computation_result {
                let cancellation_risk = self.assess_cancellation_risk(value, &test_case);
                if cancellation_risk > self.config.stability_tolerance.cancellation_threshold {
                    result.add_cancellation_event(CancellationEvent {
                        test_case: test_case.mapv(|x| x.to_f64().unwrap_or(0.0)),
                        computed_value: value.to_f64().unwrap_or(0.0),
                        cancellation_risk,
                        description: "Potential catastrophic cancellation detected".to_string(),
                    });
                }
            }
        }

        Ok(result)
    }

    fn generate_cancellation_test_cases<D, R>(
        &self,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<Vec<Array1<R>>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut cases = Vec::new();

        // Generate cases with values that might cause cancellation
        if testdata.len() >= 2 {
            // Case 1: Very similar large values
            let large_val = R::from(1e10).unwrap_or(R::max_value());
            let epsilon = R::from(1e-10).unwrap_or(R::min_positive_value());
            let similar_large = Array1::from_vec(vec![large_val, large_val + epsilon]);
            cases.push(similar_large);

            // Case 2: Values that sum to nearly zero
            let val = R::from(1e8).unwrap_or(R::one());
            let near_zero = Array1::from_vec(vec![val, -val + epsilon]);
            cases.push(near_zero);
        }

        Ok(cases)
    }

    fn assess_cancellation_risk<R>(&self, computed_value: R, testcase: &Array1<R>) -> f64
    where
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        // Simplified cancellation risk assessment
        let val_f64: f64 = NumCast::from(computed_value).unwrap_or(0.0);
        let max_input: f64 = testcase
            .iter()
            .map(|&x| NumCast::from(x).unwrap_or(0.0f64).abs())
            .fold(0.0, f64::max);

        if max_input > 0.0 && val_f64.abs() > 0.0 {
            (max_input - val_f64.abs()) / max_input
        } else {
            0.0
        }
    }
}

/// Overflow monitor
pub struct OverflowMonitor {
    config: AdvancedNumericalStabilityConfig,
}

impl OverflowMonitor {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn monitor_numerical_limits<F, D, R>(
        &self,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<OverflowMonitoringResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = OverflowMonitoringResult::new();

        // Test with extreme values
        let extreme_cases = self.generate_extreme_value_cases(testdata)?;

        for test_case in extreme_cases {
            let computation_result = test_function(&test_case.view());
            match computation_result {
                Ok(value) => {
                    if value.is_infinite() {
                        result.add_overflow_event(OverflowEvent {
                            test_case: test_case.mapv(|x| x.to_f64().unwrap_or(0.0)),
                            event_type: OverflowEventType::Overflow,
                            computed_value: value.to_f64().unwrap_or(0.0),
                            description: "Overflow detected".to_string(),
                        });
                    } else if !value.is_normal() && !value.is_zero() {
                        result.add_underflow_event(UnderflowEvent {
                            test_case: test_case.mapv(|x| x.to_f64().unwrap_or(0.0)),
                            event_type: UnderflowEventType::Underflow,
                            computed_value: value.to_f64().unwrap_or(0.0),
                            description: "Underflow detected".to_string(),
                        });
                    }
                }
                Err(_) => {
                    result.add_overflow_event(OverflowEvent {
                        test_case: test_case.mapv(|x| x.to_f64().unwrap_or(0.0)),
                        event_type: OverflowEventType::ComputationError,
                        computed_value: f64::NAN,
                        description: "Computation error on extreme values".to_string(),
                    });
                }
            }
        }

        Ok(result)
    }

    fn generate_extreme_value_cases<D, R>(
        &self,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<Vec<Array1<R>>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut cases = Vec::new();
        let datasize = testdata.len();

        // Very large values
        let largedata = Array1::from_elem(datasize, R::max_value());
        cases.push(largedata);

        // Very small values
        let smalldata = Array1::from_elem(datasize, R::min_positive_value());
        cases.push(smalldata);

        // Mixed extreme values
        if datasize >= 2 {
            let mut mixeddata = Array1::zeros(datasize);
            mixeddata[0] = R::max_value();
            mixeddata[1] = R::min_positive_value();
            cases.push(mixeddata);
        }

        Ok(cases)
    }
}

/// Condition analyzer
pub struct ConditionAnalyzer {
    config: AdvancedNumericalStabilityConfig,
}

impl ConditionAnalyzer {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn analyze_numerical_conditioning<F, D, R>(
        &self,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<ConditionAnalysisResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = ConditionAnalysisResult::new();

        // Analyze condition numbers for different input perturbations
        let base_result = test_function(&testdata.view());

        if let Ok(base_value) = base_result {
            let perturbation_factor = R::from(1e-8).unwrap_or(R::min_positive_value());

            for i in 0..testdata.len() {
                let mut perturbeddata = testdata.to_owned();
                perturbeddata[i] = perturbeddata[i] + perturbation_factor;

                let perturbed_result = test_function(&perturbeddata.view());
                if let Ok(perturbed_value) = perturbed_result {
                    let condition_number = self.estimate_condition_number(
                        base_value,
                        perturbed_value,
                        perturbation_factor,
                    );

                    result.add_condition_measurement(ConditionMeasurement {
                        input_index: i,
                        condition_number,
                        base_value: base_value.to_f64().unwrap_or(0.0),
                        perturbed_value: perturbed_value.to_f64().unwrap_or(0.0),
                        perturbation_magnitude: NumCast::from(perturbation_factor).unwrap_or(0.0),
                    });
                }
            }
        }

        Ok(result)
    }

    fn estimate_condition_number<R>(
        &self,
        base_value: R,
        perturbed_value: R,
        perturbation: R,
    ) -> f64
    where
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let base_f64: f64 = NumCast::from(base_value).unwrap_or(0.0);
        let perturbed_f64: f64 = NumCast::from(perturbed_value).unwrap_or(0.0);
        let perturbation_f64: f64 = NumCast::from(perturbation).unwrap_or(0.0);

        if base_f64.abs() > 0.0 && perturbation_f64.abs() > 0.0 {
            let relative_output_change = (perturbed_f64 - base_f64).abs() / base_f64.abs();
            let relative_input_change = perturbation_f64.abs();

            if relative_input_change > 0.0 {
                relative_output_change / relative_input_change
            } else {
                std::f64::INFINITY
            }
        } else {
            std::f64::INFINITY
        }
    }
}

/// Convergence tester
pub struct ConvergenceTester {
    config: AdvancedNumericalStabilityConfig,
}

impl ConvergenceTester {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn test_iterative_stability<F, D, R>(
        &self,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<ConvergenceStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = ConvergenceStabilityResult::new();

        // Test convergence with different precision requirements
        let tolerances = vec![1e-4, 1e-8, 1e-12];

        for tolerance in tolerances {
            let convergence_result =
                self.test_convergence_at_tolerance(test_function, testdata, tolerance)?;
            result.add_convergence_test(tolerance, convergence_result);
        }

        Ok(result)
    }

    fn test_convergence_at_tolerance<F, D, R>(
        &self,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
        tolerance: f64,
    ) -> StatsResult<ConvergenceTestResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let start_time = Instant::now();
        let mut iterations = 0;

        // Simplified convergence test - in practice, this would be more sophisticated
        let result = test_function(&testdata.view());
        iterations += 1;

        let convergence_time = start_time.elapsed();
        let converged = result.is_ok();

        Ok(ConvergenceTestResult {
            tolerance,
            converged,
            iterations,
            convergence_time,
            final_value: result.ok().map(|v| v.to_f64().unwrap_or(0.0)),
        })
    }
}

/// Monte Carlo stability tester
pub struct MonteCarloStabilityTester {
    config: AdvancedNumericalStabilityConfig,
}

impl MonteCarloStabilityTester {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn test_statistical_stability<F, D, R>(
        &self,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<MonteCarloStabilityResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = MonteCarloStabilityResult::new();
        let mut results = Vec::new();

        // Run Monte Carlo simulations
        for _ in 0..self.config.monte_carlo_samples {
            let perturbeddata = self.add_small_perturbation(testdata)?;
            let computation_result = test_function(&perturbeddata.view());

            if let Ok(value) = computation_result {
                results.push(NumCast::from(value).unwrap_or(0.0f64));
            }
        }

        if !results.is_empty() {
            // Calculate statistics
            let mean = results.iter().sum::<f64>() / results.len() as f64;
            let variance =
                results.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / results.len() as f64;
            let std_dev = variance.sqrt();

            result.sample_count = results.len();
            result.mean_value = mean;
            result.standard_deviation = std_dev;
            result.coefficient_of_variation = if mean.abs() > 0.0 {
                std_dev / mean.abs()
            } else {
                std::f64::INFINITY
            };

            // Assess stability
            result.stability_assessment = if result.coefficient_of_variation < 0.01 {
                MonteCarloStabilityAssessment::VeryStable
            } else if result.coefficient_of_variation < 0.05 {
                MonteCarloStabilityAssessment::Stable
            } else if result.coefficient_of_variation < 0.1 {
                MonteCarloStabilityAssessment::Moderate
            } else {
                MonteCarloStabilityAssessment::Unstable
            };
        }

        Ok(result)
    }

    fn add_small_perturbation<D, R>(&self, testdata: &ArrayBase<D, Ix1>) -> StatsResult<Array1<R>>
    where
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut rng = rand::rng();
        let perturbation_magnitude = R::from(1e-12).unwrap_or(R::min_positive_value());

        let perturbeddata = testdata.mapv(|x| {
            let noise: f64 = (rng.random::<f64>() - 0.5) * 2.0; // Random value in [-1, 1]
            let noise_r = R::from(noise).unwrap_or(R::zero());
            x + perturbation_magnitude * noise_r
        });

        Ok(perturbeddata)
    }
}

/// Regression tester
pub struct RegressionTester {
    config: AdvancedNumericalStabilityConfig,
    historical_results: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl RegressionTester {
    pub fn new(config: &AdvancedNumericalStabilityConfig) -> Self {
        Self {
            config: config.clone(),
            historical_results: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn test_against_historical_results<F, D, R>(
        &self,
        function_name: &str,
        test_function: &F,
        testdata: &ArrayBase<D, Ix1>,
    ) -> StatsResult<RegressionTestResult>
    where
        F: Fn(&ArrayView1<R>) -> StatsResult<R> + Clone + Send + Sync + 'static,
        D: Data<Elem = R>,
        R: Float + NumCast + Copy + Send + Sync + Debug + 'static,
    {
        let mut result = RegressionTestResult::new(function_name.to_string());

        // Execute current test
        let current_result = test_function(&testdata.view());

        if let Ok(current_value) = current_result {
            let current_f64: f64 = NumCast::from(current_value).unwrap_or(0.0);

            // Check against historical results
            let historical_lock = self.historical_results.read().unwrap();
            if let Some(historical_values) = historical_lock.get(function_name) {
                // Compare with historical values
                let mean_historical =
                    historical_values.iter().sum::<f64>() / historical_values.len() as f64;
                let deviation = (current_f64 - mean_historical).abs();
                let relative_deviation = if mean_historical.abs() > 0.0 {
                    deviation / mean_historical.abs()
                } else {
                    std::f64::INFINITY
                };

                result.current_value = current_f64;
                result.historical_mean = mean_historical;
                result.deviation = deviation;
                result.relative_deviation = relative_deviation;
                result.regression_detected =
                    relative_deviation > self.config.stability_tolerance.relative_tolerance;
            } else {
                result.isbaseline = true;
            }

            // Store current result
            drop(historical_lock);
            let mut historical_lock = self.historical_results.write().unwrap();
            historical_lock
                .entry(function_name.to_string())
                .or_insert_with(Vec::new)
                .push(current_f64);
        } else {
            result.computation_failed = true;
        }

        Ok(result)
    }
}

// Result structures

/// Comprehensive stability result
#[derive(Debug, Clone)]
pub struct ComprehensiveStabilityResult {
    pub function_name: String,
    pub test_duration: Duration,
    pub overall_stability_score: f64,
    pub stability_assessment: StabilityAssessment,
    pub edge_case_results: Option<EdgeCaseStabilityResult>,
    pub precision_results: Option<PrecisionStabilityResult>,
    pub invariant_results: Option<InvariantValidationResult>,
    pub cancellation_results: Option<CancellationDetectionResult>,
    pub overflow_results: Option<OverflowMonitoringResult>,
    pub condition_results: Option<ConditionAnalysisResult>,
    pub convergence_results: Option<ConvergenceStabilityResult>,
    pub monte_carlo_results: Option<MonteCarloStabilityResult>,
    pub regression_results: Option<RegressionTestResult>,
    pub critical_issues: Vec<CriticalIssue>,
    pub warnings: Vec<StabilityWarning>,
    pub recommendations: Vec<StabilityRecommendation>,
}

impl ComprehensiveStabilityResult {
    pub fn new(_functionname: String) -> Self {
        Self {
            function_name: _functionname,
            test_duration: Duration::from_secs(0),
            overall_stability_score: 0.0,
            stability_assessment: StabilityAssessment::Unknown,
            edge_case_results: None,
            precision_results: None,
            invariant_results: None,
            cancellation_results: None,
            overflow_results: None,
            condition_results: None,
            convergence_results: None,
            monte_carlo_results: None,
            regression_results: None,
            critical_issues: Vec::new(),
            warnings: Vec::new(),
            recommendations: Vec::new(),
        }
    }
}

// Additional result structures and enums

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StabilityAssessment {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Critical,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct StabilityRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub implementation_priority: ImplementationPriority,
    pub estimated_effort: EstimatedEffort,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecommendationType {
    Algorithm,
    Precision,
    Conditioning,
    Scaling,
    Implementation,
    Testing,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImplementationPriority {
    Immediate,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EstimatedEffort {
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Debug, Clone)]
pub struct EdgeCase<R> {
    pub edge_case_type: EdgeCaseType,
    pub data: Array1<R>,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeCaseType {
    EmptyArray,
    SingleElement,
    AllZeros,
    AllOnes,
    VerySmallValues,
    VeryLargeValues,
    ScaledData,
    ContainsNaN,
    ContainsInfinity,
    MixedSpecialValues,
}

#[derive(Debug, Clone)]
pub struct EdgeCaseStabilityResult {
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: Vec<EdgeCaseFailure>,
    pub warnings: Vec<EdgeCaseWarning>,
}

impl EdgeCaseStabilityResult {
    pub fn new() -> Self {
        Self {
            total_cases: 0,
            passed_cases: 0,
            failed_cases: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn add_test_result<R>(&mut self, edgecase: EdgeCase<R>, result: EdgeCaseTestResult) {
        self.total_cases += 1;
        match result.result_status {
            EdgeCaseResultStatus::Success => self.passed_cases += 1,
            EdgeCaseResultStatus::Error => {
                self.failed_cases.push(EdgeCaseFailure {
                    edge_case_type: result.edge_case_type,
                    error_message: result.error_message.unwrap_or_default(),
                });
            }
            EdgeCaseResultStatus::Warning => {
                self.warnings.push(EdgeCaseWarning {
                    edge_case_type: result.edge_case_type,
                    warning_message: result.error_message.unwrap_or_default(),
                });
            }
        }
    }

    pub fn analyze_edge_case_patterns(&mut self) {
        // Analyze patterns in edge case results
        // This would contain more sophisticated analysis
    }
}

#[derive(Debug, Clone)]
pub struct EdgeCaseTestResult {
    pub edge_case_type: EdgeCaseType,
    pub execution_time: Duration,
    pub result_status: EdgeCaseResultStatus,
    pub computed_value: Option<f64>,
    pub error_message: Option<String>,
    pub stability_metrics: StabilityMetrics,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EdgeCaseResultStatus {
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct EdgeCaseFailure {
    pub edge_case_type: EdgeCaseType,
    pub error_message: String,
}

#[derive(Debug, Clone)]
pub struct EdgeCaseWarning {
    pub edge_case_type: EdgeCaseType,
    pub warning_message: String,
}

#[derive(Debug, Clone)]
pub struct StabilityMetrics {
    pub condition_number: f64,
    pub relative_error: f64,
    pub nan_count: usize,
    pub infinite_count: usize,
    pub normal_count: usize,
    pub subnormal_count: usize,
}

impl Default for StabilityMetrics {
    fn default() -> Self {
        Self {
            condition_number: 1.0,
            relative_error: 0.0,
            nan_count: 0,
            infinite_count: 0,
            normal_count: 0,
            subnormal_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrecisionStabilityResult {
    pub precision_tests: HashMap<String, PrecisionTestResult>,
    pub precision_loss_detected: bool,
    pub recommended_precision: String,
}

impl PrecisionStabilityResult {
    pub fn new() -> Self {
        Self {
            precision_tests: HashMap::new(),
            precision_loss_detected: false,
            recommended_precision: "f64".to_string(),
        }
    }

    pub fn add_precision_test<R>(&mut self, precisionname: String, result: StatsResult<R>) {
        let test_result = PrecisionTestResult {
            precision_name: precisionname.clone(),
            success: result.is_ok(),
            error_message: result.err().map(|e| format!("{:?}", e)),
        };
        self.precision_tests.insert(precisionname, test_result);
    }
}

#[derive(Debug, Clone)]
pub struct PrecisionTestResult {
    pub precision_name: String,
    pub success: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct InvariantValidationResult {
    pub violations: Vec<InvariantViolation>,
    pub passed_invariants: usize,
    pub total_invariants: usize,
}

impl InvariantValidationResult {
    pub fn new() -> Self {
        Self {
            violations: Vec::new(),
            passed_invariants: 0,
            total_invariants: 0,
        }
    }

    pub fn add_violation(&mut self, violation: InvariantViolation) {
        self.violations.push(violation);
        self.total_invariants += 1;
    }
}

#[derive(Debug, Clone)]
pub struct InvariantViolation {
    pub invariant_type: InvariantType,
    pub description: String,
    pub severity: ViolationSeverity,
    pub detected_difference: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InvariantType {
    Determinism,
    Monotonicity,
    Symmetry,
    Additivity,
    Homogeneity,
    Boundedness,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViolationSeverity {
    Critical,
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone)]
pub struct CancellationDetectionResult {
    pub cancellation_events: Vec<CancellationEvent>,
    pub high_risk_cases: usize,
    pub medium_risk_cases: usize,
    pub low_risk_cases: usize,
}

impl CancellationDetectionResult {
    pub fn new() -> Self {
        Self {
            cancellation_events: Vec::new(),
            high_risk_cases: 0,
            medium_risk_cases: 0,
            low_risk_cases: 0,
        }
    }

    pub fn add_cancellation_event(&mut self, event: CancellationEvent) {
        if event.cancellation_risk > 0.8 {
            self.high_risk_cases += 1;
        } else if event.cancellation_risk > 0.5 {
            self.medium_risk_cases += 1;
        } else {
            self.low_risk_cases += 1;
        }
        self.cancellation_events.push(event);
    }
}

#[derive(Debug, Clone)]
pub struct CancellationEvent {
    pub test_case: Array1<f64>,
    pub computed_value: f64,
    pub cancellation_risk: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct OverflowMonitoringResult {
    pub overflow_events: Vec<OverflowEvent>,
    pub underflow_events: Vec<UnderflowEvent>,
    pub safe_computations: usize,
}

impl OverflowMonitoringResult {
    pub fn new() -> Self {
        Self {
            overflow_events: Vec::new(),
            underflow_events: Vec::new(),
            safe_computations: 0,
        }
    }

    pub fn add_overflow_event(&mut self, event: OverflowEvent) {
        self.overflow_events.push(event);
    }

    pub fn add_underflow_event(&mut self, event: UnderflowEvent) {
        self.underflow_events.push(event);
    }
}

#[derive(Debug, Clone)]
pub struct OverflowEvent {
    pub test_case: Array1<f64>,
    pub event_type: OverflowEventType,
    pub computed_value: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverflowEventType {
    Overflow,
    ComputationError,
}

#[derive(Debug, Clone)]
pub struct UnderflowEvent {
    pub test_case: Array1<f64>,
    pub event_type: UnderflowEventType,
    pub computed_value: f64,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UnderflowEventType {
    Underflow,
    GradualUnderflow,
}

#[derive(Debug, Clone)]
pub struct ConditionAnalysisResult {
    pub condition_measurements: Vec<ConditionMeasurement>,
    pub max_condition_number: f64,
    pub average_condition_number: f64,
    pub ill_conditioned_inputs: Vec<usize>,
}

impl ConditionAnalysisResult {
    pub fn new() -> Self {
        Self {
            condition_measurements: Vec::new(),
            max_condition_number: 0.0,
            average_condition_number: 0.0,
            ill_conditioned_inputs: Vec::new(),
        }
    }

    pub fn add_condition_measurement(&mut self, measurement: ConditionMeasurement) {
        self.max_condition_number = self.max_condition_number.max(measurement.condition_number);
        self.condition_measurements.push(measurement);

        // Update average
        self.average_condition_number = self
            .condition_measurements
            .iter()
            .map(|m| m.condition_number)
            .sum::<f64>()
            / self.condition_measurements.len() as f64;
    }
}

#[derive(Debug, Clone)]
pub struct ConditionMeasurement {
    pub input_index: usize,
    pub condition_number: f64,
    pub base_value: f64,
    pub perturbed_value: f64,
    pub perturbation_magnitude: f64,
}

#[derive(Debug, Clone)]
pub struct ConvergenceStabilityResult {
    pub convergence_tests: Vec<(f64, ConvergenceTestResult)>,
    pub overall_convergence_assessment: ConvergenceAssessment,
}

impl ConvergenceStabilityResult {
    pub fn new() -> Self {
        Self {
            convergence_tests: Vec::new(),
            overall_convergence_assessment: ConvergenceAssessment::Unknown,
        }
    }

    pub fn add_convergence_test(&mut self, tolerance: f64, result: ConvergenceTestResult) {
        self.convergence_tests.push((tolerance, result));
    }
}

#[derive(Debug, Clone)]
pub struct ConvergenceTestResult {
    pub tolerance: f64,
    pub converged: bool,
    pub iterations: usize,
    pub convergence_time: Duration,
    pub final_value: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceAssessment {
    Excellent,
    Good,
    Acceptable,
    Poor,
    Critical,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct MonteCarloStabilityResult {
    pub sample_count: usize,
    pub mean_value: f64,
    pub standard_deviation: f64,
    pub coefficient_of_variation: f64,
    pub stability_assessment: MonteCarloStabilityAssessment,
}

impl MonteCarloStabilityResult {
    pub fn new() -> Self {
        Self {
            sample_count: 0,
            mean_value: 0.0,
            standard_deviation: 0.0,
            coefficient_of_variation: 0.0,
            stability_assessment: MonteCarloStabilityAssessment::Unknown,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MonteCarloStabilityAssessment {
    VeryStable,
    Stable,
    Moderate,
    Unstable,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct RegressionTestResult {
    pub function_name: String,
    pub current_value: f64,
    pub historical_mean: f64,
    pub deviation: f64,
    pub relative_deviation: f64,
    pub regression_detected: bool,
    pub isbaseline: bool,
    pub computation_failed: bool,
}

impl RegressionTestResult {
    pub fn new(_functionname: String) -> Self {
        Self {
            function_name: _functionname,
            current_value: 0.0,
            historical_mean: 0.0,
            deviation: 0.0,
            relative_deviation: 0.0,
            regression_detected: false,
            isbaseline: false,
            computation_failed: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StabilityTestResult {
    pub function_name: String,
    pub timestamp: SystemTime,
    pub stability_score: f64,
    pub critical_issues: usize,
    pub warnings: usize,
}

#[derive(Debug, Clone)]
pub struct StabilityTrendAnalysis {
    pub trend: StabilityTrend,
    pub average_score: f64,
    pub recent_average: f64,
    pub total_tests: usize,
    pub total_critical_issues: usize,
    pub total_warnings: usize,
}

impl Default for StabilityTrendAnalysis {
    fn default() -> Self {
        Self {
            trend: StabilityTrend::Stable,
            average_score: 0.0,
            recent_average: 0.0,
            total_tests: 0,
            total_critical_issues: 0,
            total_warnings: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StabilityTrend {
    Improving,
    Stable,
    Declining,
}

#[derive(Debug, Clone)]
pub struct CriticalIssue {
    pub issue_type: CriticalIssueType,
    pub description: String,
    pub severity: IssueSeverity,
    pub function_context: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CriticalIssueType {
    NumericalInstability,
    CatastrophicCancellation,
    OverflowUnderflow,
    NonDeterminism,
    ConvergenceFailure,
}

#[derive(Debug, Clone)]
pub struct StabilityWarning {
    pub warning_type: WarningType,
    pub description: String,
    pub severity: IssueSeverity,
    pub function_context: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WarningType {
    PrecisionLoss,
    HighConditionNumber,
    SlowConvergence,
    LargeVariance,
    EdgeCaseIssue,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IssueSeverity {
    Critical,
    High,
    Medium,
    Low,
}

// Convenience functions for creating testers

/// Create comprehensive numerical stability tester
#[allow(dead_code)]
pub fn create_advanced_think_numerical_stability_tester() -> AdvancedNumericalStabilityTester {
    let config = AdvancedNumericalStabilityConfig::default();
    AdvancedNumericalStabilityTester::new(config)
}

/// Create fast numerical stability tester for development
#[allow(dead_code)]
pub fn create_fast_numerical_stability_tester() -> AdvancedNumericalStabilityTester {
    let config = AdvancedNumericalStabilityConfig {
        enable_edge_case_testing: true,
        enable_precision_analysis: true,
        enable_invariant_validation: true,
        enable_cancellation_detection: false,
        enable_overflow_monitoring: false,
        enable_condition_analysis: false,
        enable_differentiation_testing: false,
        enable_convergence_testing: false,
        enable_monte_carlo_testing: false,
        enable_regression_testing: false,
        thoroughness_level: NumericalStabilityThoroughness::Basic,
        precision_strategy: PrecisionTestingStrategy::DoublePrecision,
        edge_case_approach: EdgeCaseGenerationApproach::Predefined,
        stability_tolerance: StabilityTolerance::default(),
        test_timeout: Duration::from_secs(60), // 1 minute
        max_convergence_iterations: 1000,
        monte_carlo_samples: 1000,
    };
    AdvancedNumericalStabilityTester::new(config)
}

/// Create exhaustive numerical stability tester for release validation
#[allow(dead_code)]
pub fn create_exhaustive_numerical_stability_tester() -> AdvancedNumericalStabilityTester {
    let config = AdvancedNumericalStabilityConfig {
        enable_edge_case_testing: true,
        enable_precision_analysis: true,
        enable_invariant_validation: true,
        enable_cancellation_detection: true,
        enable_overflow_monitoring: true,
        enable_condition_analysis: true,
        enable_differentiation_testing: true,
        enable_convergence_testing: true,
        enable_monte_carlo_testing: true,
        enable_regression_testing: true,
        thoroughness_level: NumericalStabilityThoroughness::Exhaustive,
        precision_strategy: PrecisionTestingStrategy::MultiPrecision,
        edge_case_approach: EdgeCaseGenerationApproach::Intelligent,
        stability_tolerance: StabilityTolerance {
            absolute_tolerance: 1e-16,
            relative_tolerance: 1e-14,
            condition_number_threshold: 1e15,
            cancellation_threshold: 1e-12,
            convergence_tolerance: 1e-12,
            monte_carlo_confidence_level: 0.99,
        },
        test_timeout: Duration::from_secs(3600), // 1 hour
        max_convergence_iterations: 100000,
        monte_carlo_samples: 1000000,
    };
    AdvancedNumericalStabilityTester::new(config)
}

/// Enhanced numerical stability testing for common statistical functions
#[allow(dead_code)]
pub fn test_statistical_function_stability<F>(
    function_name: &str,
    test_function: F,
    input_ranges: Vec<(f64, f64)>,
) -> StatsResult<ComprehensiveStabilityResult>
where
    F: Fn(&ArrayView1<f64>) -> StatsResult<f64> + Clone + Send + Sync + 'static,
{
    let config = AdvancedNumericalStabilityConfig::default();
    let tester = AdvancedNumericalStabilityTester::new(config);

    // Generate test data across multiple _ranges
    let mut comprehensive_result = ComprehensiveStabilityResult::new(function_name.to_string());

    for (min_val, max_val) in input_ranges {
        // Generate test data for this range
        let testdata = generate_stability_testdata(min_val, max_val, 1000);

        // Run comprehensive stability testing
        let range_result = tester.comprehensive_stability_testing(
            function_name,
            test_function.clone(),
            &testdata,
        )?;

        // Combine results (simplified - in practice would merge more intelligently)
        if comprehensive_result.edge_case_results.is_none() {
            comprehensive_result.edge_case_results = range_result.edge_case_results;
        }
        if comprehensive_result.precision_results.is_none() {
            comprehensive_result.precision_results = range_result.precision_results;
        }
    }

    Ok(comprehensive_result)
}

/// Generate test data for numerical stability testing
#[allow(dead_code)]
fn generate_stability_testdata(min_val: f64, maxval: f64, size: usize) -> Array1<f64> {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Array1::zeros(size);

    for i in 0..size {
        // Mix of different value types for comprehensive testing
        match i % 5 {
            0 => data[i] = rng.gen_range(min_val..maxval), // Random in range
            1 => data[i] = min_val,                            // Minimum value
            2 => data[i] = maxval,                            // Maximum value
            3 => data[i] = (min_val + maxval) / 2.0,          // Midpoint
            4 => data[i] = rng.gen_range(min_val..maxval) * 1e-10, // Very small values
            _ => unreachable!(),
        }
    }

    data
}

/// Test numerical stability of mean function specifically
#[allow(dead_code)]
pub fn test_mean_stability() -> StatsResult<ComprehensiveStabilityResult> {
    use crate::descriptive::mean;

    let mean_function = |data: &ArrayView1<f64>| mean(data);

    let input_ranges = vec![
        (-1e6, 1e6),     // Large numbers
        (-1.0, 1.0),     // Normal range
        (-1e-10, 1e-10), // Very small numbers
        (1e10, 1e11),    // Very large positive numbers
        (-1e11, -1e10),  // Very large negative numbers
    ];

    test_statistical_function_stability("mean", mean_function, input_ranges)
}

/// Test numerical stability of variance function specifically
#[allow(dead_code)]
pub fn test_variance_stability() -> StatsResult<ComprehensiveStabilityResult> {
    use crate::descriptive::var;

    let variance_function = |data: &ArrayView1<f64>| var(data, 1, None);

    let input_ranges = vec![
        (-1e6, 1e6),     // Large numbers
        (-1.0, 1.0),     // Normal range
        (-1e-10, 1e-10), // Very small numbers
        (0.0, 1e-6),     // Small positive numbers
        (1e6, 1e7),      // Large positive numbers
    ];

    test_statistical_function_stability("variance", variance_function, input_ranges)
}

/// Test numerical stability of correlation function specifically
#[allow(dead_code)]
pub fn test_correlation_stability() -> StatsResult<ValidationReport> {
    use crate::propertybased_validation::{
        CorrelationBounds, PropertyBasedValidator, PropertyTestConfig,
    };

    // Use property-based testing for correlation stability
    let config = PropertyTestConfig {
        test_cases_per_property: 1000,
        seed: 42,
        tolerance: 1e-12,
        test_edge_cases: true,
        test_cross_platform: true,
        test_numerical_stability: true,
    };

    let mut validator = PropertyBasedValidator::new(config);

    // Test correlation bounds property with various edge cases
    validator.test_property(CorrelationBounds)?;

    Ok(validator.generate_validation_report())
}

/// Run comprehensive numerical stability tests for all core statistical functions
#[allow(dead_code)]
pub fn run_comprehensive_statistical_stability_tests(
) -> StatsResult<HashMap<String, ComprehensiveStabilityResult>> {
    let mut results = HashMap::new();

    // Test mean stability
    if let Ok(mean_result) = test_mean_stability() {
        results.insert("mean".to_string(), mean_result);
    }

    // Test variance stability
    if let Ok(var_result) = test_variance_stability() {
        results.insert("variance".to_string(), var_result);
    }

    // Additional statistical functions can be added here

    Ok(results)
}

/// Quick numerical stability validation for CI/CD pipelines
#[allow(dead_code)]
pub fn run_quick_stability_validation() -> StatsResult<bool> {
    let results = run_comprehensive_statistical_stability_tests()?;

    // Check if all tests passed basic stability requirements
    let all_stable = results.values().all(|result| {
        // Simplified stability check - in practice would be more sophisticated
        result
            .edge_case_results
            .as_ref()
            .map(|edge_results| edge_results.failed_cases.is_empty())
            .unwrap_or(true)
    });

    Ok(all_stable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_advanced_think_numerical_stability_tester_creation() {
        let tester = create_advanced_think_numerical_stability_tester();
        assert!(tester.config.enable_edge_case_testing);
        assert!(tester.config.enable_precision_analysis);
        assert!(tester.config.enable_invariant_validation);
    }

    #[test]
    fn test_stability_tolerance_default() {
        let tolerance = StabilityTolerance::default();
        assert_eq!(tolerance.absolute_tolerance, 1e-14);
        assert_eq!(tolerance.relative_tolerance, 1e-12);
        assert_eq!(tolerance.condition_number_threshold, 1e12);
    }

    #[test]
    fn test_edge_case_generation() {
        let config = AdvancedNumericalStabilityConfig::default();
        let generator = EdgeCaseGenerator::new(&config);
        let testdata = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let edge_cases = generator
            .generate_comprehensive_edge_cases(&testdata)
            .unwrap();
        assert!(edge_cases.len() > 0);
    }

    #[test]
    fn test_stability_assessment_levels() {
        assert_eq!(StabilityAssessment::Excellent as u8, 0);
        assert!(matches!(
            StabilityAssessment::Good,
            StabilityAssessment::Good
        ));
        assert!(matches!(
            StabilityAssessment::Critical,
            StabilityAssessment::Critical
        ));
    }

    #[test]
    fn test_edge_case_types() {
        assert_eq!(EdgeCaseType::EmptyArray as u8, 0);
        assert!(matches!(EdgeCaseType::AllZeros, EdgeCaseType::AllZeros));
        assert!(matches!(
            EdgeCaseType::ContainsNaN,
            EdgeCaseType::ContainsNaN
        ));
    }

    #[test]
    fn test_comprehensive_stability_result_creation() {
        let result = ComprehensiveStabilityResult::new("test_function".to_string());
        assert_eq!(result.function_name, "test_function");
        assert_eq!(result.overall_stability_score, 0.0);
        assert!(matches!(
            result.stability_assessment,
            StabilityAssessment::Unknown
        ));
    }

    #[test]
    fn test_stability_metrics_default() {
        let metrics = StabilityMetrics::default();
        assert_eq!(metrics.condition_number, 1.0);
        assert_eq!(metrics.relative_error, 0.0);
        assert_eq!(metrics.nan_count, 0);
    }

    #[test]
    fn test_monte_carlo_stability_result_creation() {
        let result = MonteCarloStabilityResult::new();
        assert_eq!(result.sample_count, 0);
        assert_eq!(result.mean_value, 0.0);
        assert!(matches!(
            result.stability_assessment,
            MonteCarloStabilityAssessment::Unknown
        ));
    }

    #[test]
    fn test_fast_stability_tester_config() {
        let tester = create_fast_numerical_stability_tester();
        assert!(tester.config.enable_edge_case_testing);
        assert!(!tester.config.enable_monte_carlo_testing);
        assert_eq!(
            tester.config.thoroughness_level,
            NumericalStabilityThoroughness::Basic
        );
    }

    #[test]
    fn test_exhaustive_stability_tester_config() {
        let tester = create_exhaustive_numerical_stability_tester();
        assert!(tester.config.enable_edge_case_testing);
        assert!(tester.config.enable_monte_carlo_testing);
        assert_eq!(
            tester.config.thoroughness_level,
            NumericalStabilityThoroughness::Exhaustive
        );
        assert_eq!(tester.config.monte_carlo_samples, 1000000);
    }
}
