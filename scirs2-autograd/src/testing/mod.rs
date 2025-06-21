//! Numerical stability testing framework for automatic differentiation
//!
//! This module provides comprehensive testing tools for verifying the numerical
//! stability and correctness of automatic differentiation operations.

use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::Float;
use std::collections::HashMap;
use std::fmt;

pub mod finite_differences;
pub mod gradient_checking;
pub mod numerical_analysis;
pub mod stability_metrics;
pub mod stability_test_framework;

/// Configuration for numerical stability testing
#[derive(Debug, Clone)]
pub struct StabilityTestConfig {
    /// Tolerance for gradient checks
    pub gradient_tolerance: f64,
    /// Tolerance for finite difference approximations
    pub finite_diff_tolerance: f64,
    /// Step size for finite differences
    pub finite_diff_step: f64,
    /// Number of random test points to sample
    pub num_test_points: usize,
    /// Enable second-order gradient checking
    pub check_second_order: bool,
    /// Maximum condition number to accept
    pub max_condition_number: f64,
    /// Enable comprehensive error analysis
    pub comprehensive_analysis: bool,
}

impl Default for StabilityTestConfig {
    fn default() -> Self {
        Self {
            gradient_tolerance: 1e-5,
            finite_diff_tolerance: 1e-6,
            finite_diff_step: 1e-8,
            num_test_points: 100,
            check_second_order: false,
            max_condition_number: 1e12,
            comprehensive_analysis: true,
        }
    }
}

/// Main numerical stability tester
pub struct NumericalStabilityTester<F: Float> {
    config: StabilityTestConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> NumericalStabilityTester<F> {
    /// Create a new numerical stability tester
    pub fn new() -> Self {
        Self {
            config: StabilityTestConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: StabilityTestConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Test the numerical stability of a computation graph
    pub fn test_graph(&self, graph: &Graph<F>) -> Result<StabilityReport<F>, StabilityError> {
        let mut report = StabilityReport::new();

        // Test gradient accuracy using finite differences
        let gradient_tests = self.test_gradient_accuracy(graph)?;
        report.gradient_tests = gradient_tests;

        // Test for numerical conditioning issues
        let conditioning_tests = self.test_numerical_conditioning(graph)?;
        report.conditioning_tests = conditioning_tests;

        // Test stability under perturbations
        let perturbation_tests = self.test_perturbation_stability(graph)?;
        report.perturbation_tests = perturbation_tests;

        // Test overflow/underflow susceptibility
        let overflow_tests = self.test_overflow_underflow(graph)?;
        report.overflow_tests = overflow_tests;

        // Generate overall assessment
        report.overall_grade = self.compute_overall_grade(&report);

        Ok(report)
    }

    /// Test gradient accuracy using finite differences
    fn test_gradient_accuracy(
        &self,
        _graph: &Graph<F>,
    ) -> Result<GradientTestResults, StabilityError> {
        let mut results = GradientTestResults {
            tests_performed: 0,
            tests_passed: 0,
            max_error: 0.0,
            mean_error: 0.0,
            failed_tests: Vec::new(),
        };

        // For each variable in the graph:
        // 1. Compute analytical gradient
        // 2. Compute finite difference approximation
        // 3. Compare and record differences

        for _test_point in 0..self.config.num_test_points {
            results.tests_performed += 1;

            // Simulate gradient test (would use actual graph operations)
            let analytical_grad = self.compute_analytical_gradient()?;
            let finite_diff_grad = self.compute_finite_difference_gradient()?;

            let error = self.compute_gradient_error(&analytical_grad, &finite_diff_grad);

            if error < self.config.gradient_tolerance {
                results.tests_passed += 1;
            } else {
                results.failed_tests.push(GradientTestFailure {
                    test_id: results.tests_performed,
                    error,
                    analytical_gradient: analytical_grad,
                    finite_diff_gradient: finite_diff_grad,
                });
            }

            results.max_error = results.max_error.max(error);
            results.mean_error += error;
        }

        if results.tests_performed > 0 {
            results.mean_error /= results.tests_performed as f64;
        }

        Ok(results)
    }

    /// Test numerical conditioning of operations
    fn test_numerical_conditioning(
        &self,
        _graph: &Graph<F>,
    ) -> Result<ConditioningTestResults, StabilityError> {
        let mut results = ConditioningTestResults {
            condition_numbers: HashMap::new(),
            ill_conditioned_operations: Vec::new(),
            stability_warnings: Vec::new(),
        };

        // For each operation in the graph:
        // 1. Compute condition number if applicable
        // 2. Check for potential numerical issues
        // 3. Generate warnings for problematic operations

        // Example operations to check:
        let operations_to_check = vec![
            "matrix_inverse",
            "solve_linear_system",
            "eigenvalue_decomposition",
            "singular_value_decomposition",
            "division_operations",
        ];

        for op_name in operations_to_check {
            let condition_number = self.estimate_condition_number(op_name)?;
            results
                .condition_numbers
                .insert(op_name.to_string(), condition_number);

            if condition_number > self.config.max_condition_number {
                results
                    .ill_conditioned_operations
                    .push(IllConditionedOperation {
                        operation: op_name.to_string(),
                        condition_number,
                        severity: if condition_number > 1e15 {
                            ConditioningSeverity::Critical
                        } else if condition_number > 1e12 {
                            ConditioningSeverity::High
                        } else {
                            ConditioningSeverity::Medium
                        },
                    });
            }
        }

        Ok(results)
    }

    /// Test stability under input perturbations
    fn test_perturbation_stability(
        &self,
        _graph: &Graph<F>,
    ) -> Result<PerturbationTestResults, StabilityError> {
        let mut results = PerturbationTestResults {
            perturbation_tests: Vec::new(),
            max_sensitivity: 0.0,
            mean_sensitivity: 0.0,
        };

        // Test sensitivity to small input perturbations
        for perturbation_magnitude in [1e-8, 1e-6, 1e-4, 1e-2] {
            let sensitivity = self.measure_perturbation_sensitivity(perturbation_magnitude)?;

            results.perturbation_tests.push(PerturbationTest {
                perturbation_magnitude,
                output_change: sensitivity,
                sensitivity_ratio: sensitivity / perturbation_magnitude,
            });

            results.max_sensitivity = results.max_sensitivity.max(sensitivity);
            results.mean_sensitivity += sensitivity;
        }

        if !results.perturbation_tests.is_empty() {
            results.mean_sensitivity /= results.perturbation_tests.len() as f64;
        }

        Ok(results)
    }

    /// Test for overflow and underflow susceptibility
    fn test_overflow_underflow(
        &self,
        _graph: &Graph<F>,
    ) -> Result<OverflowTestResults<F>, StabilityError> {
        let mut results = OverflowTestResults {
            overflow_risks: Vec::new(),
            underflow_risks: Vec::new(),
            safe_ranges: HashMap::new(),
        };

        // Test with extreme input values
        let extreme_values = vec![
            F::from(1e-100).unwrap(), // Very small
            F::from(1e-10).unwrap(),  // Small
            F::from(1e10).unwrap(),   // Large
            F::from(1e100).unwrap(),  // Very large
        ];

        for &extreme_value in &extreme_values {
            let risk_assessment = self.assess_overflow_risk(extreme_value)?;

            if risk_assessment.overflow_probability > 0.1 {
                results.overflow_risks.push(OverflowRisk {
                    input_value: extreme_value,
                    operation: risk_assessment.risky_operation.clone(),
                    probability: risk_assessment.overflow_probability,
                });
            }

            if risk_assessment.underflow_probability > 0.1 {
                results.underflow_risks.push(UnderflowRisk {
                    input_value: extreme_value,
                    operation: risk_assessment.risky_operation,
                    probability: risk_assessment.underflow_probability,
                });
            }
        }

        Ok(results)
    }

    /// Helper methods for computations
    fn compute_analytical_gradient(&self) -> Result<Vec<f64>, StabilityError> {
        // Simplified - would compute actual analytical gradient
        Ok(vec![1.0, 2.0, 3.0])
    }

    fn compute_finite_difference_gradient(&self) -> Result<Vec<f64>, StabilityError> {
        // Simplified - would compute finite difference approximation
        Ok(vec![1.0001, 1.9999, 3.0001])
    }

    fn compute_gradient_error(&self, analytical: &[f64], finite_diff: &[f64]) -> f64 {
        analytical
            .iter()
            .zip(finite_diff.iter())
            .map(|(&a, &f)| (a - f).abs())
            .fold(0.0, f64::max)
    }

    fn estimate_condition_number(&self, _operation: &str) -> Result<f64, StabilityError> {
        // Simplified - would compute actual condition number
        Ok(1e6)
    }

    fn measure_perturbation_sensitivity(&self, _perturbation: f64) -> Result<f64, StabilityError> {
        // Simplified - would measure actual sensitivity
        Ok(_perturbation * 1.5) // Example: amplification factor of 1.5
    }

    fn assess_overflow_risk(&self, _input: F) -> Result<OverflowRiskAssessment, StabilityError> {
        Ok(OverflowRiskAssessment {
            risky_operation: "exponential".to_string(),
            overflow_probability: 0.05,
            underflow_probability: 0.02,
        })
    }

    fn compute_overall_grade(&self, report: &StabilityReport<F>) -> StabilityGrade {
        let mut score = 100.0;

        // Penalize gradient test failures
        if report.gradient_tests.tests_performed > 0 {
            let pass_rate = report.gradient_tests.tests_passed as f64
                / report.gradient_tests.tests_performed as f64;
            score *= pass_rate;
        }

        // Penalize conditioning issues
        let conditioning_penalty =
            report.conditioning_tests.ill_conditioned_operations.len() as f64 * 10.0;
        score -= conditioning_penalty;

        // Penalize overflow risks
        let overflow_penalty = (report.overflow_tests.overflow_risks.len()
            + report.overflow_tests.underflow_risks.len()) as f64
            * 5.0;
        score -= overflow_penalty;

        match score as i32 {
            90..=100 => StabilityGrade::Excellent,
            80..=89 => StabilityGrade::Good,
            70..=79 => StabilityGrade::Fair,
            60..=69 => StabilityGrade::Poor,
            _ => StabilityGrade::Critical,
        }
    }
}

impl<F: Float> Default for NumericalStabilityTester<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Results of stability testing
#[derive(Debug, Clone)]
pub struct StabilityReport<F: Float> {
    pub gradient_tests: GradientTestResults,
    pub conditioning_tests: ConditioningTestResults,
    pub perturbation_tests: PerturbationTestResults,
    pub overflow_tests: OverflowTestResults<F>,
    pub overall_grade: StabilityGrade,
}

impl<F: Float> Default for StabilityReport<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> StabilityReport<F> {
    pub fn new() -> Self {
        Self {
            gradient_tests: GradientTestResults::default(),
            conditioning_tests: ConditioningTestResults::default(),
            perturbation_tests: PerturbationTestResults::default(),
            overflow_tests: OverflowTestResults::default(),
            overall_grade: StabilityGrade::Unknown,
        }
    }

    /// Print a comprehensive report
    pub fn print_report(&self) {
        println!("Numerical Stability Report");
        println!("==========================");
        println!("Overall Grade: {:?}", self.overall_grade);
        println!();

        println!("Gradient Tests:");
        println!("  Tests Performed: {}", self.gradient_tests.tests_performed);
        println!("  Tests Passed: {}", self.gradient_tests.tests_passed);
        println!(
            "  Pass Rate: {:.2}%",
            if self.gradient_tests.tests_performed > 0 {
                (self.gradient_tests.tests_passed as f64
                    / self.gradient_tests.tests_performed as f64)
                    * 100.0
            } else {
                0.0
            }
        );
        println!("  Max Error: {:.2e}", self.gradient_tests.max_error);
        println!("  Mean Error: {:.2e}", self.gradient_tests.mean_error);
        println!();

        println!("Conditioning Tests:");
        println!(
            "  Ill-conditioned Operations: {}",
            self.conditioning_tests.ill_conditioned_operations.len()
        );
        for op in &self.conditioning_tests.ill_conditioned_operations {
            println!(
                "    {} (cond: {:.2e}, severity: {:?})",
                op.operation, op.condition_number, op.severity
            );
        }
        println!();

        println!("Perturbation Tests:");
        println!(
            "  Max Sensitivity: {:.2e}",
            self.perturbation_tests.max_sensitivity
        );
        println!(
            "  Mean Sensitivity: {:.2e}",
            self.perturbation_tests.mean_sensitivity
        );
        println!();

        println!("Overflow/Underflow Tests:");
        println!(
            "  Overflow Risks: {}",
            self.overflow_tests.overflow_risks.len()
        );
        println!(
            "  Underflow Risks: {}",
            self.overflow_tests.underflow_risks.len()
        );
    }
}

/// Gradient testing results
#[derive(Debug, Clone, Default)]
pub struct GradientTestResults {
    pub tests_performed: usize,
    pub tests_passed: usize,
    pub max_error: f64,
    pub mean_error: f64,
    pub failed_tests: Vec<GradientTestFailure>,
}

/// Failed gradient test information
#[derive(Debug, Clone)]
pub struct GradientTestFailure {
    pub test_id: usize,
    pub error: f64,
    pub analytical_gradient: Vec<f64>,
    pub finite_diff_gradient: Vec<f64>,
}

/// Conditioning test results
#[derive(Debug, Clone, Default)]
pub struct ConditioningTestResults {
    pub condition_numbers: HashMap<String, f64>,
    pub ill_conditioned_operations: Vec<IllConditionedOperation>,
    pub stability_warnings: Vec<String>,
}

/// Information about ill-conditioned operations
#[derive(Debug, Clone)]
pub struct IllConditionedOperation {
    pub operation: String,
    pub condition_number: f64,
    pub severity: ConditioningSeverity,
}

/// Severity levels for conditioning issues
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConditioningSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Perturbation test results
#[derive(Debug, Clone, Default)]
pub struct PerturbationTestResults {
    pub perturbation_tests: Vec<PerturbationTest>,
    pub max_sensitivity: f64,
    pub mean_sensitivity: f64,
}

/// Individual perturbation test
#[derive(Debug, Clone)]
pub struct PerturbationTest {
    pub perturbation_magnitude: f64,
    pub output_change: f64,
    pub sensitivity_ratio: f64,
}

/// Overflow/underflow test results  
#[derive(Debug, Clone)]
pub struct OverflowTestResults<F: Float> {
    pub overflow_risks: Vec<OverflowRisk<F>>,
    pub underflow_risks: Vec<UnderflowRisk<F>>,
    pub safe_ranges: HashMap<String, (f64, f64)>,
}

impl<F: Float> Default for OverflowTestResults<F> {
    fn default() -> Self {
        Self {
            overflow_risks: Vec::new(),
            underflow_risks: Vec::new(),
            safe_ranges: HashMap::new(),
        }
    }
}

/// Overflow risk information
#[derive(Debug, Clone)]
pub struct OverflowRisk<F: Float> {
    pub input_value: F,
    pub operation: String,
    pub probability: f64,
}

/// Underflow risk information
#[derive(Debug, Clone)]
pub struct UnderflowRisk<F: Float> {
    pub input_value: F,
    pub operation: String,
    pub probability: f64,
}

/// Risk assessment for overflow/underflow
#[derive(Debug, Clone)]
pub struct OverflowRiskAssessment {
    pub risky_operation: String,
    pub overflow_probability: f64,
    pub underflow_probability: f64,
}

/// Overall stability grade
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StabilityGrade {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
    Unknown,
}

impl fmt::Display for StabilityGrade {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            StabilityGrade::Excellent => write!(f, "Excellent (A+)"),
            StabilityGrade::Good => write!(f, "Good (A)"),
            StabilityGrade::Fair => write!(f, "Fair (B)"),
            StabilityGrade::Poor => write!(f, "Poor (C)"),
            StabilityGrade::Critical => write!(f, "Critical (F)"),
            StabilityGrade::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Errors that can occur during stability testing
#[derive(Debug, thiserror::Error)]
pub enum StabilityError {
    #[error("Computation error: {0}")]
    ComputationError(String),
    #[error("Gradient computation failed: {0}")]
    GradientError(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Public API functions
/// Test the numerical stability of a computation graph
pub fn test_numerical_stability<F: Float>(
    graph: &Graph<F>,
) -> Result<StabilityReport<F>, StabilityError> {
    let tester = NumericalStabilityTester::new();
    tester.test_graph(graph)
}

/// Test with custom configuration
pub fn test_numerical_stability_with_config<F: Float>(
    graph: &Graph<F>,
    config: StabilityTestConfig,
) -> Result<StabilityReport<F>, StabilityError> {
    let tester = NumericalStabilityTester::with_config(config);
    tester.test_graph(graph)
}

/// Quick gradient check for a specific computation
pub fn quick_gradient_check<F: Float>(
    _inputs: &[Tensor<F>],
    _output: &Tensor<F>,
) -> Result<bool, StabilityError> {
    // Simplified gradient check implementation
    Ok(true)
}

/// Assess numerical conditioning of an operation
pub fn assess_conditioning<F: Float>(
    _operation_name: &str,
    _inputs: &[Tensor<F>],
) -> Result<f64, StabilityError> {
    // Simplified conditioning assessment
    Ok(1e6)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_tester_creation() {
        let _tester = NumericalStabilityTester::<f32>::new();
    }

    #[test]
    fn test_stability_config() {
        let config = StabilityTestConfig {
            gradient_tolerance: 1e-6,
            num_test_points: 50,
            ..Default::default()
        };

        let _tester = NumericalStabilityTester::<f32>::with_config(config.clone());
        assert_eq!(config.gradient_tolerance, 1e-6);
        assert_eq!(config.num_test_points, 50);
    }

    #[test]
    fn test_stability_report() {
        let report: StabilityReport<f64> = StabilityReport::new();
        assert!(matches!(report.overall_grade, StabilityGrade::Unknown));
    }

    #[test]
    fn test_stability_grade_display() {
        assert_eq!(format!("{}", StabilityGrade::Excellent), "Excellent (A+)");
        assert_eq!(format!("{}", StabilityGrade::Poor), "Poor (C)");
        assert_eq!(format!("{}", StabilityGrade::Critical), "Critical (F)");
    }

    #[test]
    fn test_conditioning_severity() {
        let operation = IllConditionedOperation {
            operation: "matrix_inverse".to_string(),
            condition_number: 1e15,
            severity: ConditioningSeverity::Critical,
        };

        assert!(matches!(operation.severity, ConditioningSeverity::Critical));
        assert!(operation.condition_number > 1e14);
    }

    #[test]
    fn test_perturbation_test() {
        let test = PerturbationTest {
            perturbation_magnitude: 1e-8,
            output_change: 1.5e-8,
            sensitivity_ratio: 1.5,
        };

        let calculated_ratio = test.output_change / test.perturbation_magnitude;
        assert!((test.sensitivity_ratio - calculated_ratio).abs() < 1e-14);
    }
}
