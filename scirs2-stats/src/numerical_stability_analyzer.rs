//! Numerical stability analysis framework
//!
//! This module provides comprehensive analysis of numerical stability
//! for statistical functions across various conditions and edge cases.
//!
//! ## Features
//!
//! - Condition number analysis
//! - Error propagation tracking  
//! - Edge case robustness testing
//! - Precision loss detection
//! - Algorithm stability recommendations

use crate::error::StatsResult;
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comprehensive numerical stability analyzer
#[derive(Debug)]
pub struct NumericalStabilityAnalyzer {
    config: StabilityConfig,
    analysis_results: HashMap<String, StabilityAnalysisResult>,
}

/// Configuration for stability analysis
#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// Tolerance for considering values as numerically zero
    pub zero_tolerance: f64,
    /// Tolerance for detecting loss of precision
    pub precision_tolerance: f64,
    /// Maximum condition number considered stable
    pub max_condition_number: f64,
    /// Number of perturbation tests to run
    pub perturbation_tests: usize,
    /// Magnitude of perturbations for sensitivity analysis
    pub perturbation_magnitude: f64,
    /// Enable testing with extreme values
    pub test_extreme_values: bool,
    /// Enable testing with nearly singular matrices
    pub test_singular_cases: bool,
}

/// Result of numerical stability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityAnalysisResult {
    /// Function name analyzed
    pub function_name: String,
    /// Overall stability grade
    pub stability_grade: StabilityGrade,
    /// Condition number analysis
    pub condition_analysis: ConditionNumberAnalysis,
    /// Error propagation analysis
    pub error_propagation: ErrorPropagationAnalysis,
    /// Edge case robustness
    pub edge_case_robustness: EdgeCaseRobustness,
    /// Precision loss analysis
    pub precision_analysis: PrecisionAnalysis,
    /// Recommendations for improvement
    pub recommendations: Vec<StabilityRecommendation>,
    /// Overall stability score (0-100)
    pub stability_score: f64,
}

/// Stability grading scale
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StabilityGrade {
    /// Excellent numerical stability
    Excellent,
    /// Good stability with minor issues
    Good,
    /// Acceptable stability with some concerns
    Acceptable,
    /// Poor stability with significant issues
    Poor,
    /// Numerically unstable
    Unstable,
}

/// Condition number analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionNumberAnalysis {
    /// Estimated condition number
    pub condition_number: f64,
    /// Classification of conditioning
    pub conditioning_class: ConditioningClass,
    /// Loss of accuracy estimate (digits)
    pub accuracy_loss_digits: f64,
    /// Sensitivity to input perturbations
    pub input_sensitivity: f64,
}

/// Classification of matrix conditioning
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConditioningClass {
    /// Well-conditioned (condition number < 1e12)
    WellConditioned,
    /// Moderately conditioned (1e12 <= cond < 1e14)
    ModeratelyConditioned,
    /// Poorly conditioned (1e14 <= cond < 1e16)
    PoorlyConditioned,
    /// Nearly singular (condition number >= 1e16)
    NearlySingular,
}

/// Error propagation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPropagationAnalysis {
    /// Forward error bound
    pub forward_error_bound: f64,
    /// Backward error bound
    pub backward_error_bound: f64,
    /// Error amplification factor
    pub error_amplification: f64,
    /// Stability in the presence of rounding errors
    pub rounding_error_stability: f64,
}

/// Edge case robustness analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCaseRobustness {
    /// Handles infinite values correctly
    pub handles_infinity: bool,
    /// Handles NaN values correctly
    pub handles_nan: bool,
    /// Handles zero values correctly
    pub handles_zero: bool,
    /// Handles very large values correctly
    pub handles_large_values: bool,
    /// Handles very small values correctly
    pub handles_small_values: bool,
    /// Proportion of edge cases handled correctly
    pub edge_case_success_rate: f64,
}

/// Precision loss analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionAnalysis {
    /// Estimated bits of precision lost
    pub precision_loss_bits: f64,
    /// Relative precision remaining
    pub relative_precision: f64,
    /// Cancellation errors detected
    pub cancellation_errors: Vec<CancellationError>,
    /// Overflow/underflow risks
    pub overflow_underflow_risk: OverflowRisk,
}

/// Cancellation error detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CancellationError {
    /// Location where cancellation occurs
    pub location: String,
    /// Magnitude of precision loss
    pub precision_loss: f64,
    /// Suggested mitigation
    pub mitigation: String,
}

/// Overflow/underflow risk assessment
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OverflowRisk {
    /// No overflow/underflow risk
    None,
    /// Low risk
    Low,
    /// Moderate risk
    Moderate,
    /// High risk
    High,
    /// Certain overflow/underflow
    Certain,
}

/// Stability improvement recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityRecommendation {
    /// Type of recommendation
    pub recommendation_type: RecommendationType,
    /// Description of the issue
    pub description: String,
    /// Suggested improvement
    pub suggestion: String,
    /// Priority of this recommendation
    pub priority: RecommendationPriority,
    /// Expected improvement in stability
    pub expected_improvement: f64,
}

/// Types of stability recommendations
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendationType {
    /// Algorithm modification
    Algorithm,
    /// Numerical technique improvement
    Numerical,
    /// Input validation enhancement
    InputValidation,
    /// Precision management
    Precision,
    /// Error handling improvement
    ErrorHandling,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    /// Critical - must be addressed
    Critical,
    /// High priority
    High,
    /// Medium priority
    Medium,
    /// Low priority
    Low,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            zero_tolerance: 1e-15,
            precision_tolerance: 1e-12,
            max_condition_number: 1e12,
            perturbation_tests: 100,
            perturbation_magnitude: 1e-10,
            test_extreme_values: true,
            test_singular_cases: true,
        }
    }
}

impl NumericalStabilityAnalyzer {
    /// Create a new numerical stability analyzer
    pub fn new(config: StabilityConfig) -> Self {
        Self {
            config,
            analysis_results: HashMap::new(),
        }
    }

    /// Create analyzer with default configuration
    pub fn default() -> Self {
        Self::new(StabilityConfig::default())
    }

    /// Analyze numerical stability of a statistical function
    pub fn analyze_function<F>(
        &mut self,
        function_name: &str,
        function: F,
        testdata: &ArrayView1<f64>,
    ) -> StatsResult<StabilityAnalysisResult>
    where
        F: Fn(&ArrayView1<f64>) -> StatsResult<f64>,
    {
        // Condition number analysis
        let condition_analysis = self.analyze_condition_number(testdata)?;

        // Error propagation analysis
        let error_propagation = self.analyze_error_propagation(&function, testdata)?;

        // Edge case robustness testing
        let edge_case_robustness = self.test_edge_case_robustness(&function)?;

        // Precision loss analysis
        let precision_analysis = self.analyze_precision_loss(&function, testdata)?;

        // Generate recommendations
        let recommendations = self.generate_recommendations(
            &condition_analysis,
            &error_propagation,
            &edge_case_robustness,
            &precision_analysis,
        );

        // Calculate overall stability score
        let stability_score = self.calculate_stability_score(
            &condition_analysis,
            &error_propagation,
            &edge_case_robustness,
            &precision_analysis,
        );

        // Determine stability grade
        let stability_grade = self.grade_stability(stability_score);

        let result = StabilityAnalysisResult {
            function_name: function_name.to_string(),
            stability_grade,
            condition_analysis,
            error_propagation,
            edge_case_robustness,
            precision_analysis,
            recommendations,
            stability_score,
        };

        self.analysis_results
            .insert(function_name.to_string(), result.clone());
        Ok(result)
    }

    /// Analyze condition number and sensitivity
    fn analyze_condition_number(
        &self,
        data: &ArrayView1<f64>,
    ) -> StatsResult<ConditionNumberAnalysis> {
        // For 1D data, we estimate conditioning based on data characteristics
        let data_range = data.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let data_min = data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x.abs()));

        // Estimate condition number as ratio of max to min (simplified)
        let condition_number = if data_min > self.config.zero_tolerance {
            data_range / data_min
        } else {
            f64::INFINITY
        };

        let conditioning_class = if condition_number < 1e12 {
            ConditioningClass::WellConditioned
        } else if condition_number < 1e14 {
            ConditioningClass::ModeratelyConditioned
        } else if condition_number < 1e16 {
            ConditioningClass::PoorlyConditioned
        } else {
            ConditioningClass::NearlySingular
        };

        // Estimate accuracy loss (simplified)
        let accuracy_loss_digits = condition_number.log10().max(0.0);

        // Input sensitivity (simplified estimate)
        let input_sensitivity = condition_number / 1e16;

        Ok(ConditionNumberAnalysis {
            condition_number,
            conditioning_class,
            accuracy_loss_digits,
            input_sensitivity,
        })
    }

    /// Analyze error propagation through perturbation testing
    fn analyze_error_propagation<F>(
        &self,
        function: &F,
        data: &ArrayView1<f64>,
    ) -> StatsResult<ErrorPropagationAnalysis>
    where
        F: Fn(&ArrayView1<f64>) -> StatsResult<f64>,
    {
        // Get reference result
        let reference_result = function(data)?;

        let mut forward_errors = Vec::new();
        let mut backward_errors = Vec::new();

        // Perform perturbation tests
        for i in 0..self.config.perturbation_tests.min(data.len()) {
            let mut perturbeddata = data.to_owned();
            let perturbation = self.config.perturbation_magnitude * perturbeddata[i].abs().max(1.0);
            perturbeddata[i] += perturbation;

            if let Ok(perturbed_result) = function(&perturbeddata.view()) {
                let forward_error = (perturbed_result - reference_result).abs();
                let backward_error = perturbation.abs();

                forward_errors.push(forward_error);
                backward_errors.push(backward_error);
            }
        }

        let forward_error_bound = forward_errors.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let backward_error_bound = backward_errors.iter().fold(0.0f64, |acc, &x| acc.max(x));

        // Error amplification factor
        let error_amplification = if backward_error_bound > 0.0 {
            forward_error_bound / backward_error_bound
        } else {
            1.0
        };

        // Simplified rounding error stability estimate
        let rounding_error_stability = 1.0 / (1.0 + error_amplification);

        Ok(ErrorPropagationAnalysis {
            forward_error_bound,
            backward_error_bound,
            error_amplification,
            rounding_error_stability,
        })
    }

    /// Test robustness with edge cases
    fn test_edge_case_robustness<F>(&self, function: &F) -> StatsResult<EdgeCaseRobustness>
    where
        F: Fn(&ArrayView1<f64>) -> StatsResult<f64>,
    {
        let mut tests_passed = 0;
        let mut total_tests = 0;

        let mut handles_infinity = false;
        let mut handles_nan = false;
        let mut handles_zero = false;
        let mut handles_large_values = false;
        let mut handles_small_values = false;

        // Test with infinity
        if self.config.test_extreme_values {
            total_tests += 1;
            let infdata = Array1::from_vec(vec![f64::INFINITY, 1.0, 2.0]);
            if let Ok(result) = function(&infdata.view()) {
                if result.is_finite() || result.is_infinite() {
                    handles_infinity = true;
                    tests_passed += 1;
                }
            }

            // Test with NaN
            total_tests += 1;
            let nandata = Array1::from_vec(vec![f64::NAN, 1.0, 2.0]);
            if let Ok(result) = function(&nandata.view()) {
                if result.is_nan() || result.is_finite() {
                    handles_nan = true;
                    tests_passed += 1;
                }
            }

            // Test with zeros
            total_tests += 1;
            let zerodata = Array1::from_vec(vec![0.0, 0.0, 0.0]);
            if let Ok(_) = function(&zerodata.view()) {
                handles_zero = true;
                tests_passed += 1;
            }

            // Test with very large values
            total_tests += 1;
            let largedata = Array1::from_vec(vec![1e100, 1e200, 1e300]);
            if let Ok(_) = function(&largedata.view()) {
                handles_large_values = true;
                tests_passed += 1;
            }

            // Test with very small values
            total_tests += 1;
            let smalldata = Array1::from_vec(vec![1e-100, 1e-200, 1e-300]);
            if let Ok(_) = function(&smalldata.view()) {
                handles_small_values = true;
                tests_passed += 1;
            }
        }

        let edge_case_success_rate = if total_tests > 0 {
            tests_passed as f64 / total_tests as f64
        } else {
            1.0
        };

        Ok(EdgeCaseRobustness {
            handles_infinity,
            handles_nan,
            handles_zero,
            handles_large_values,
            handles_small_values,
            edge_case_success_rate,
        })
    }

    /// Analyze precision loss and numerical issues
    fn analyze_precision_loss<F>(
        &self,
        function: &F,
        data: &ArrayView1<f64>,
    ) -> StatsResult<PrecisionAnalysis>
    where
        F: Fn(&ArrayView1<f64>) -> StatsResult<f64>,
    {
        // Simplified precision analysis
        let result = function(data)?;

        // Estimate precision loss based on result characteristics
        let precision_loss_bits = if result.abs() < self.config.precision_tolerance {
            16.0 // Significant precision loss
        } else if result.abs() < 1e-10 {
            8.0 // Moderate precision loss
        } else {
            2.0 // Minimal precision loss
        };

        let relative_precision = 1.0 - (precision_loss_bits / 64.0);

        // Detect potential cancellation errors (simplified)
        let mut cancellation_errors = Vec::new();
        if data.iter().any(|&x| x.abs() < self.config.zero_tolerance) {
            cancellation_errors.push(CancellationError {
                location: "inputdata".to_string(),
                precision_loss: precision_loss_bits,
                mitigation: "Use higher precision arithmetic or alternative algorithm".to_string(),
            });
        }

        // Assess overflow/underflow risk
        let max_val = data.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
        let overflow_underflow_risk = if max_val > 1e100 {
            OverflowRisk::High
        } else if max_val > 1e50 {
            OverflowRisk::Moderate
        } else if max_val < 1e-100 {
            OverflowRisk::Moderate
        } else {
            OverflowRisk::Low
        };

        Ok(PrecisionAnalysis {
            precision_loss_bits,
            relative_precision,
            cancellation_errors,
            overflow_underflow_risk,
        })
    }

    /// Generate stability improvement recommendations
    fn generate_recommendations(
        &self,
        condition_analysis: &ConditionNumberAnalysis,
        error_propagation: &ErrorPropagationAnalysis,
        edge_case_robustness: &EdgeCaseRobustness,
        precision_analysis: &PrecisionAnalysis,
    ) -> Vec<StabilityRecommendation> {
        let mut recommendations = Vec::new();

        // Condition number recommendations
        if matches!(
            condition_analysis.conditioning_class,
            ConditioningClass::PoorlyConditioned | ConditioningClass::NearlySingular
        ) {
            recommendations.push(StabilityRecommendation {
                recommendation_type: RecommendationType::Algorithm,
                description: "Poor conditioning detected".to_string(),
                suggestion: "Consider using regularization or alternative algorithms for ill-conditioned problems".to_string(),
                priority: RecommendationPriority::High,
                expected_improvement: 30.0,
            });
        }

        // Error _propagation recommendations
        if error_propagation.error_amplification > 100.0 {
            recommendations.push(StabilityRecommendation {
                recommendation_type: RecommendationType::Numerical,
                description: "High error amplification detected".to_string(),
                suggestion: "Implement error _analysis and use more stable numerical methods"
                    .to_string(),
                priority: RecommendationPriority::High,
                expected_improvement: 25.0,
            });
        }

        // Edge case recommendations
        if edge_case_robustness.edge_case_success_rate < 0.8 {
            recommendations.push(StabilityRecommendation {
                recommendation_type: RecommendationType::InputValidation,
                description: "Poor edge case handling".to_string(),
                suggestion:
                    "Improve input validation and add special case handling for extreme values"
                        .to_string(),
                priority: RecommendationPriority::Medium,
                expected_improvement: 20.0,
            });
        }

        // Precision recommendations
        if precision_analysis.precision_loss_bits > 10.0 {
            recommendations.push(StabilityRecommendation {
                recommendation_type: RecommendationType::Precision,
                description: "Significant precision loss detected".to_string(),
                suggestion:
                    "Consider using higher precision arithmetic or numerically stable algorithms"
                        .to_string(),
                priority: RecommendationPriority::High,
                expected_improvement: 35.0,
            });
        }

        recommendations
    }

    /// Calculate overall stability score
    fn calculate_stability_score(
        &self,
        condition_analysis: &ConditionNumberAnalysis,
        error_propagation: &ErrorPropagationAnalysis,
        edge_case_robustness: &EdgeCaseRobustness,
        precision_analysis: &PrecisionAnalysis,
    ) -> f64 {
        let mut score = 100.0;

        // Penalize poor conditioning
        score -= match condition_analysis.conditioning_class {
            ConditioningClass::WellConditioned => 0.0,
            ConditioningClass::ModeratelyConditioned => 10.0,
            ConditioningClass::PoorlyConditioned => 25.0,
            ConditioningClass::NearlySingular => 40.0,
        };

        // Penalize high error amplification
        score -= (error_propagation.error_amplification.log10() * 5.0).min(30.0);

        // Penalize poor edge case handling
        score -= (1.0 - edge_case_robustness.edge_case_success_rate) * 20.0;

        // Penalize precision loss
        score -= (precision_analysis.precision_loss_bits / 64.0) * 30.0;

        score.max(0.0)
    }

    /// Grade overall stability
    fn grade_stability(&self, score: f64) -> StabilityGrade {
        if score >= 90.0 {
            StabilityGrade::Excellent
        } else if score >= 75.0 {
            StabilityGrade::Good
        } else if score >= 60.0 {
            StabilityGrade::Acceptable
        } else if score >= 40.0 {
            StabilityGrade::Poor
        } else {
            StabilityGrade::Unstable
        }
    }

    /// Generate comprehensive stability report
    pub fn generate_stability_report(&self) -> StabilityReport {
        let results: Vec<_> = self.analysis_results.values().cloned().collect();

        let total_functions = results.len();
        let excellent_count = results
            .iter()
            .filter(|r| r.stability_grade == StabilityGrade::Excellent)
            .count();
        let good_count = results
            .iter()
            .filter(|r| r.stability_grade == StabilityGrade::Good)
            .count();
        let acceptable_count = results
            .iter()
            .filter(|r| r.stability_grade == StabilityGrade::Acceptable)
            .count();
        let poor_count = results
            .iter()
            .filter(|r| r.stability_grade == StabilityGrade::Poor)
            .count();
        let unstable_count = results
            .iter()
            .filter(|r| r.stability_grade == StabilityGrade::Unstable)
            .count();

        let average_score = if total_functions > 0 {
            results.iter().map(|r| r.stability_score).sum::<f64>() / total_functions as f64
        } else {
            0.0
        };

        StabilityReport {
            total_functions,
            excellent_count,
            good_count,
            acceptable_count,
            poor_count,
            unstable_count,
            average_score,
            function_results: results,
            generated_at: chrono::Utc::now(),
        }
    }
}

/// Comprehensive stability analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    /// Total number of functions analyzed
    pub total_functions: usize,
    /// Number of functions with excellent stability
    pub excellent_count: usize,
    /// Number of functions with good stability
    pub good_count: usize,
    /// Number of functions with acceptable stability
    pub acceptable_count: usize,
    /// Number of functions with poor stability
    pub poor_count: usize,
    /// Number of unstable functions
    pub unstable_count: usize,
    /// Average stability score across all functions
    pub average_score: f64,
    /// Detailed results for each function
    pub function_results: Vec<StabilityAnalysisResult>,
    /// Timestamp when report was generated
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptive::mean;

    #[test]
    fn test_stability_analyzer_creation() {
        let analyzer = NumericalStabilityAnalyzer::default();
        assert_eq!(analyzer.config.zero_tolerance, 1e-15);
        assert_eq!(analyzer.config.precision_tolerance, 1e-12);
    }

    #[test]
    fn test_condition_number_analysis() {
        let analyzer = NumericalStabilityAnalyzer::default();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = analyzer.analyze_condition_number(&data.view()).unwrap();

        assert_eq!(
            result.conditioning_class,
            ConditioningClass::WellConditioned
        );
        assert!(result.condition_number > 0.0);
    }

    #[test]
    fn test_stability_grading() {
        let analyzer = NumericalStabilityAnalyzer::default();

        assert_eq!(analyzer.grade_stability(95.0), StabilityGrade::Excellent);
        assert_eq!(analyzer.grade_stability(80.0), StabilityGrade::Good);
        assert_eq!(analyzer.grade_stability(65.0), StabilityGrade::Acceptable);
        assert_eq!(analyzer.grade_stability(45.0), StabilityGrade::Poor);
        assert_eq!(analyzer.grade_stability(20.0), StabilityGrade::Unstable);
    }

    #[test]
    fn test_mean_stability_analysis() {
        let mut analyzer = NumericalStabilityAnalyzer::default();
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = analyzer
            .analyze_function("mean", |x| mean(x), &data.view())
            .unwrap();

        assert_eq!(result.function_name, "mean");
        assert!(matches!(
            result.stability_grade,
            StabilityGrade::Excellent | StabilityGrade::Good
        ));
        assert!(result.stability_score > 50.0);
    }
}
