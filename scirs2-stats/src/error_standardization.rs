//! Error message standardization for consistent error handling
//!
//! This module provides standardized error messages and recovery suggestions
//! that are used consistently across all statistical modules.

use crate::error::{StatsError, StatsResult};
use std::collections::HashMap;

/// Standardized error message templates
pub struct ErrorMessages;

impl ErrorMessages {
    /// Standard dimension mismatch messages
    pub fn dimension_mismatch(expected: &str, actual: &str) -> StatsError {
        StatsError::dimension_mismatch(format!(
            "Array dimension mismatch: _expected {}, got {}. {}",
            expected,
            actual,
            "Ensure all input arrays have compatible dimensions for the operation."
        ))
    }

    /// Standard array length mismatch messages
    pub fn length_mismatch(
        array1_name: &str,
        len1: usize,
        array2_name: &str,
        len2: usize,
    ) -> StatsError {
        StatsError::dimension_mismatch(format!(
            "Array length mismatch: {} has {} elements, {} has {} elements. {}",
            array1_name,
            len1,
            array2_name,
            len2,
            "Both arrays must have the same number of elements."
        ))
    }

    /// Standard empty array messages
    pub fn empty_array(arrayname: &str) -> StatsError {
        StatsError::invalid_argument(format!(
            "Array '{}' cannot be empty. {}",
            arrayname, "Provide an array with at least one element."
        ))
    }

    /// Standard insufficient data messages
    pub fn insufficientdata(operation: &str, required: usize, actual: usize) -> StatsError {
        StatsError::invalid_argument(format!(
            "Insufficient data for {}: requires at least {} elements, got {}. {}",
            operation,
            required,
            actual,
            if required == 2 {
                "Statistical calculations typically require at least 2 data points."
            } else {
                "Increase the sample size or use a different method."
            }
        ))
    }

    /// Standard non-positive value messages
    pub fn non_positive_value(parameter: &str, value: f64) -> StatsError {
        StatsError::domain(format!(
            "Parameter '{}' must be positive, got {}. {}",
            parameter, value, "Ensure the value is greater than 0."
        ))
    }

    /// Standard probability range messages
    pub fn invalid_probability(parameter: &str, value: f64) -> StatsError {
        StatsError::domain(format!(
            "Parameter '{}' must be a valid probability between 0 and 1, got {}. {}",
            parameter, value, "Probability values must be in the range [0, 1]."
        ))
    }

    /// Standard NaN detection messages
    pub fn nan_detected(context: &str) -> StatsError {
        StatsError::invalid_argument(format!(
            "NaN (Not a Number) values detected in {}. {}",
            context, "Remove NaN values or use functions that handle missing data explicitly."
        ))
    }

    /// Standard infinite value messages
    pub fn infinite_value_detected(context: &str) -> StatsError {
        StatsError::invalid_argument(format!(
            "Infinite values detected in {}. {}",
            context, "Check for overflow conditions or extreme values in your data."
        ))
    }

    /// Standard matrix not positive definite messages
    pub fn not_positive_definite(matrixname: &str) -> StatsError {
        StatsError::computation(format!(
            "Matrix '{}' is not positive definite. {}",
            matrixname,
            "Ensure the matrix is symmetric and all eigenvalues are positive, or use regularization."
        ))
    }

    /// Standard singular matrix messages
    pub fn singular_matrix(matrixname: &str) -> StatsError {
        StatsError::computation(format!(
            "Matrix '{}' is singular (non-invertible). {}",
            matrixname, "Check for linear dependencies in your data or add regularization."
        ))
    }

    /// Standard convergence failure messages
    pub fn convergence_failure(algorithm: &str, iterations: usize) -> StatsError {
        StatsError::ConvergenceError(format!(
            "{} failed to converge after {} iterations. {}",
            algorithm, iterations,
            "Try increasing the maximum iterations, adjusting tolerance, or using different initial values."
        ))
    }

    /// Standard numerical instability messages
    pub fn numerical_instability(operation: &str, suggestion: &str) -> StatsError {
        StatsError::computation(format!(
            "Numerical instability detected in {}. {}",
            operation, suggestion
        ))
    }

    /// Standard unsupported operation messages
    pub fn unsupported_operation(operation: &str, context: &str) -> StatsError {
        StatsError::not_implemented(format!(
            "Operation '{}' is not supported for {}. {}",
            operation,
            context,
            "Check the documentation for supported operations or consider alternative methods."
        ))
    }
}

/// Context-aware error validation
pub struct ErrorValidator;

impl ErrorValidator {
    /// Validate array for common issues
    pub fn validate_array<T>(data: &[T], name: &str) -> StatsResult<()>
    where
        T: PartialOrd + Copy,
    {
        if data.is_empty() {
            return Err(ErrorMessages::empty_array(name));
        }
        Ok(())
    }

    /// Validate array for finite values (for float types)
    pub fn validate_finite_array(data: &[f64], name: &str) -> StatsResult<()> {
        if data.is_empty() {
            return Err(ErrorMessages::empty_array(name));
        }

        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() {
                return Err(ErrorMessages::nan_detected(&format!("{}[{}]", name, i)));
            }
            if value.is_infinite() {
                return Err(ErrorMessages::infinite_value_detected(&format!(
                    "{}[{}]",
                    name, i
                )));
            }
        }
        Ok(())
    }

    /// Validate probability value
    pub fn validate_probability(value: f64, name: &str) -> StatsResult<()> {
        if !(0.0..=1.0).contains(&value) {
            return Err(ErrorMessages::invalid_probability(name, value));
        }
        if value.is_nan() {
            return Err(ErrorMessages::nan_detected(name));
        }
        Ok(())
    }

    /// Validate positive value
    pub fn validate_positive(value: f64, name: &str) -> StatsResult<()> {
        if value <= 0.0 {
            return Err(ErrorMessages::non_positive_value(name, value));
        }
        if value.is_nan() {
            return Err(ErrorMessages::nan_detected(name));
        }
        if value.is_infinite() {
            return Err(ErrorMessages::infinite_value_detected(name));
        }
        Ok(())
    }

    /// Validate arrays have same length
    pub fn validate_same_length<T, U>(
        arr1: &[T],
        arr1_name: &str,
        arr2: &[U],
        arr2_name: &str,
    ) -> StatsResult<()> {
        if arr1.len() != arr2.len() {
            return Err(ErrorMessages::length_mismatch(
                arr1_name,
                arr1.len(),
                arr2_name,
                arr2.len(),
            ));
        }
        Ok(())
    }

    /// Validate minimum sample size
    pub fn validate_samplesize(size: usize, minimum: usize, operation: &str) -> StatsResult<()> {
        if size < minimum {
            return Err(ErrorMessages::insufficientdata(operation, minimum, size));
        }
        Ok(())
    }
}

/// Performance impact assessment for error recovery
#[derive(Debug, Clone, Copy)]
pub enum PerformanceImpact {
    /// No performance impact
    None,
    /// Minimal performance impact (< 5%)
    Minimal,
    /// Moderate performance impact (5-20%)
    Moderate,
    /// Significant performance impact (> 20%)
    Significant,
}

/// Standardized error recovery suggestions
pub struct RecoverySuggestions;

impl RecoverySuggestions {
    /// Get recovery suggestions for common statistical errors
    pub fn get_suggestions(error: &StatsError) -> Vec<(String, PerformanceImpact)> {
        match error {
            StatsError::DimensionMismatch(_) => vec![
                (
                    "Reshape arrays to have compatible dimensions".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Use broadcasting-compatible operations".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Transpose matrices if needed".to_string(),
                    PerformanceImpact::Minimal,
                ),
            ],
            StatsError::InvalidArgument(msg) if msg.contains("empty") => vec![
                (
                    "Provide non-empty input arrays".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Use default values for empty inputs".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Filter out empty arrays before processing".to_string(),
                    PerformanceImpact::Minimal,
                ),
            ],
            StatsError::InvalidArgument(msg) if msg.contains("NaN") => vec![
                (
                    "Remove NaN values using data.dropna()".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Use interpolation to fill NaN values".to_string(),
                    PerformanceImpact::Moderate,
                ),
                (
                    "Use statistical methods that handle NaN explicitly".to_string(),
                    PerformanceImpact::Minimal,
                ),
            ],
            StatsError::ComputationError(msg) if msg.contains("singular") => vec![
                (
                    "Add regularization (e.g., ridge regression)".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Use pseudo-inverse instead of inverse".to_string(),
                    PerformanceImpact::Moderate,
                ),
                (
                    "Check for multicollinearity in data".to_string(),
                    PerformanceImpact::None,
                ),
            ],
            StatsError::ConvergenceError(_) => vec![
                (
                    "Increase maximum iterations".to_string(),
                    PerformanceImpact::Moderate,
                ),
                (
                    "Adjust convergence tolerance".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Use different initial values".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Try a different optimization algorithm".to_string(),
                    PerformanceImpact::Significant,
                ),
            ],
            StatsError::DomainError(_) => vec![
                (
                    "Check parameter bounds and constraints".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Scale or normalize input data".to_string(),
                    PerformanceImpact::Minimal,
                ),
                (
                    "Use robust statistical methods".to_string(),
                    PerformanceImpact::Moderate,
                ),
            ],
            _ => vec![
                (
                    "Check input data for validity".to_string(),
                    PerformanceImpact::None,
                ),
                (
                    "Refer to function documentation".to_string(),
                    PerformanceImpact::None,
                ),
            ],
        }
    }

    /// Get context-specific suggestions for statistical operations
    pub fn get_context_suggestions(operation: &str) -> HashMap<String, Vec<String>> {
        let mut suggestions = HashMap::new();

        match operation {
            "correlation" => {
                suggestions.insert(
                    "data_preparation".to_string(),
                    vec![
                        "Ensure data is numeric and finite".to_string(),
                        "Consider outlier detection and removal".to_string(),
                        "Check for missing values".to_string(),
                    ],
                );
                suggestions.insert(
                    "performance".to_string(),
                    vec![
                        "Use SIMD-optimized functions for large datasets".to_string(),
                        "Consider parallel computation for correlation matrices".to_string(),
                    ],
                );
            }
            "regression" => {
                suggestions.insert(
                    "data_preparation".to_string(),
                    vec![
                        "Check for multicollinearity".to_string(),
                        "Normalize features if needed".to_string(),
                        "Consider feature selection".to_string(),
                    ],
                );
                suggestions.insert(
                    "model_selection".to_string(),
                    vec![
                        "Use regularization for high-dimensional data".to_string(),
                        "Consider robust regression for outliers".to_string(),
                    ],
                );
            }
            "hypothesis_testing" => {
                suggestions.insert(
                    "assumptions".to_string(),
                    vec![
                        "Check normality assumptions".to_string(),
                        "Verify independence of observations".to_string(),
                        "Consider non-parametric alternatives".to_string(),
                    ],
                );
                suggestions.insert(
                    "interpretation".to_string(),
                    vec![
                        "Adjust for multiple comparisons if needed".to_string(),
                        "Consider effect size in addition to p-values".to_string(),
                    ],
                );
            }
            _ => {
                suggestions.insert(
                    "general".to_string(),
                    vec![
                        "Validate input data quality".to_string(),
                        "Check function prerequisites".to_string(),
                    ],
                );
            }
        }

        suggestions
    }
}

/// Comprehensive error reporting with standardized messages
pub struct StandardizedErrorReporter;

impl StandardizedErrorReporter {
    /// Generate a comprehensive error report
    pub fn generate_report(error: &StatsError, context: Option<&str>) -> String {
        let mut report = String::new();

        // Main _error message
        report.push_str(&format!("❌ Error: {}\n\n", error));

        // Context information
        if let Some(ctx) = context {
            report.push_str(&format!("📍 Context: {}\n\n", ctx));
        }

        // Recovery suggestions
        let suggestions = RecoverySuggestions::get_suggestions(error);
        if !suggestions.is_empty() {
            report.push_str("💡 Suggested Solutions:\n");
            for (i, (suggestion, impact)) in suggestions.iter().enumerate() {
                let impact_icon = match impact {
                    PerformanceImpact::None => "⚡",
                    PerformanceImpact::Minimal => "🔋",
                    PerformanceImpact::Moderate => "⏱️",
                    PerformanceImpact::Significant => "⚠️",
                };
                report.push_str(&format!("   {}. {} {}\n", i + 1, impact_icon, suggestion));
            }
            report.push('\n');
        }

        // Performance impact legend
        report.push_str("Legend: ⚡ No impact, 🔋 Minimal, ⏱️ Moderate, ⚠️ Significant\n");

        report
    }
}

/// Enhanced error context for better debugging
#[derive(Debug, Clone)]
pub struct EnhancedErrorContext {
    /// The function where the error occurred
    pub function_name: String,
    /// The module where the error occurred
    pub module_name: String,
    /// Input data characteristics
    pub data_info: DataDiagnostics,
    /// System information
    pub system_info: SystemDiagnostics,
    /// Suggested recovery actions with priority
    pub recovery_actions: Vec<RecoveryAction>,
}

/// Data characteristics for error diagnosis
#[derive(Debug, Clone)]
pub struct DataDiagnostics {
    /// Data shape information
    pub shape: Vec<usize>,
    /// Data type information
    pub data_type: String,
    /// Statistical summary
    pub summary: StatsSummary,
    /// Quality issues detected
    pub quality_issues: Vec<DataQualityIssue>,
}

/// Statistical summary for error context
#[derive(Debug, Clone)]
pub struct StatsSummary {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub nan_count: usize,
    pub inf_count: usize,
    pub finite_count: usize,
}

/// Data quality issues
#[derive(Debug, Clone, PartialEq)]
pub enum DataQualityIssue {
    HasNaN,
    HasInfinite,
    HasNegative,
    HasZeros,
    Constant,
    HighSkewness,
    Outliers(usize),
    SmallSample(usize),
}

/// System diagnostics for error context
#[derive(Debug, Clone)]
pub struct SystemDiagnostics {
    /// Available memory (approximate)
    pub available_memory_mb: Option<usize>,
    /// CPU information
    pub cpu_info: String,
    /// SIMD capabilities
    pub simd_available: bool,
    /// Thread count
    pub thread_count: usize,
}

/// Recovery action with metadata
#[derive(Debug, Clone)]
pub struct RecoveryAction {
    /// Description of the action
    pub description: String,
    /// Priority (1 = highest, 5 = lowest)
    pub priority: u8,
    /// Expected performance impact
    pub performance_impact: PerformanceImpact,
    /// Code example (if applicable)
    pub code_example: Option<String>,
    /// Whether this action is automatic or manual
    pub automatic: bool,
}

/// Batch error handler for operations on multiple datasets
pub struct BatchErrorHandler {
    errors: Vec<(usize, StatsError, EnhancedErrorContext)>,
    warnings: Vec<(usize, String)>,
}

impl BatchErrorHandler {
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Add an error for a specific batch item
    pub fn add_error(&mut self, index: usize, error: StatsError, context: EnhancedErrorContext) {
        self.errors.push((index, error, context));
    }

    /// Add a warning for a specific batch item
    pub fn add_warning(&mut self, index: usize, warning: String) {
        self.warnings.push((index, warning));
    }

    /// Generate a comprehensive batch error report
    pub fn generate_batch_report(&self) -> String {
        let mut report = String::new();

        if !self.errors.is_empty() {
            report.push_str(&format!(
                "🚨 Batch Processing Errors ({} errors):\n\n",
                self.errors.len()
            ));

            for (i, (index, error, context)) in self.errors.iter().enumerate() {
                report.push_str(&format!("Error {} (Item {}):\n", i + 1, index));
                report.push_str(&format!("  ❌ {}\n", error));
                report.push_str(&format!(
                    "  📍 Function: {}::{}\n",
                    context.module_name, context.function_name
                ));
                report.push_str(&format!("  📊 Data: {:?}\n", context.data_info.shape));

                if !context.recovery_actions.is_empty() {
                    report.push_str("  💡 Suggested Actions:\n");
                    for action in &context.recovery_actions {
                        let priority_icon = match action.priority {
                            1 => "🔴",
                            2 => "🟡",
                            3 => "🟢",
                            _ => "⚪",
                        };
                        report.push_str(&format!("    {} {}\n", priority_icon, action.description));
                    }
                }
                report.push('\n');
            }
        }

        if !self.warnings.is_empty() {
            report.push_str(&format!(
                "⚠️  Batch Processing Warnings ({} warnings):\n\n",
                self.warnings.len()
            ));

            for (index, warning) in &self.warnings {
                report.push_str(&format!("  Item {}: {}\n", index, warning));
            }
        }

        report
    }

    /// Get summary statistics of errors
    pub fn get_error_summary(&self) -> HashMap<String, usize> {
        let mut summary = HashMap::new();

        for (_, error_, _) in &self.errors {
            let error_type = match error_ {
                StatsError::ComputationError(_) => "Computation",
                StatsError::DomainError(_) => "Domain",
                StatsError::DimensionMismatch(_) => "Dimension",
                StatsError::InvalidArgument(_) => "Invalid Argument",
                StatsError::NotImplementedError(_) => "Not Implemented",
                StatsError::ConvergenceError(_) => "Convergence",
                StatsError::CoreError(_) => "Core",
                StatsError::InsufficientData(_) => "Insufficient Data",
                StatsError::InvalidInput(_) => "Invalid Input",
                StatsError::NotImplemented(_) => "Not Implemented",
                StatsError::DistributionError(_) => "Distribution",
            };

            *summary.entry(error_type.to_string()).or_insert(0) += 1;
        }

        summary
    }
}

impl Default for BatchErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhanced error diagnostics
pub struct ErrorDiagnostics;

impl ErrorDiagnostics {
    /// Generate comprehensive diagnostics for array data
    pub fn diagnose_array_f64(data: &[f64], name: &str) -> DataDiagnostics {
        let mut quality_issues = Vec::new();
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut finite_values = Vec::new();
        let mut has_negative = false;
        let mut has_zeros = false;

        for &value in data {
            if value.is_nan() {
                nan_count += 1;
            } else if value.is_infinite() {
                inf_count += 1;
            } else {
                finite_values.push(value);
                if value < 0.0 {
                    has_negative = true;
                }
                if value == 0.0 {
                    has_zeros = true;
                }
            }
        }

        // Calculate basic statistics for finite values
        let (min, max, mean, std) = if !finite_values.is_empty() {
            let min = finite_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = finite_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean = finite_values.iter().sum::<f64>() / finite_values.len() as f64;
            let variance = finite_values
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / finite_values.len() as f64;
            let std = variance.sqrt();
            (Some(min), Some(max), Some(mean), Some(std))
        } else {
            (None, None, None, None)
        };

        // Detect quality issues
        if nan_count > 0 {
            quality_issues.push(DataQualityIssue::HasNaN);
        }
        if inf_count > 0 {
            quality_issues.push(DataQualityIssue::HasInfinite);
        }
        if has_negative {
            quality_issues.push(DataQualityIssue::HasNegative);
        }
        if has_zeros {
            quality_issues.push(DataQualityIssue::HasZeros);
        }
        if finite_values.len() < 2 {
            quality_issues.push(DataQualityIssue::SmallSample(finite_values.len()));
        }

        // Check for constant data
        if let (Some(min_val), Some(max_val)) = (min, max) {
            if (max_val - min_val).abs() < 1e-15 {
                quality_issues.push(DataQualityIssue::Constant);
            }
        }

        // Simple outlier detection (values beyond 3 std devs)
        if let (Some(mean_val), Some(std_val)) = (mean, std) {
            if std_val > 0.0 {
                let outlier_count = finite_values
                    .iter()
                    .filter(|&&x| (x - mean_val).abs() > 3.0 * std_val)
                    .count();
                if outlier_count > 0 {
                    quality_issues.push(DataQualityIssue::Outliers(outlier_count));
                }
            }
        }

        DataDiagnostics {
            shape: vec![data.len()],
            data_type: "f64".to_string(),
            summary: StatsSummary {
                min,
                max,
                mean,
                std,
                nan_count,
                inf_count,
                finite_count: finite_values.len(),
            },
            quality_issues,
        }
    }

    /// Generate system diagnostics
    pub fn get_system_diagnostics() -> SystemDiagnostics {
        use scirs2_core::simd_ops::PlatformCapabilities;

        let capabilities = PlatformCapabilities::detect();
        let thread_count = num_cpus::get();

        SystemDiagnostics {
            available_memory_mb: Self::get_available_memory_mb(),
            cpu_info: format!("Threads: {}", thread_count),
            simd_available: capabilities.simd_available,
            thread_count,
        }
    }

    fn get_available_memory_mb() -> Option<usize> {
        // Simple approximation - would use system APIs in production
        Some(8192) // Assume 8GB available
    }
}

/// Inter-module error consistency checker
pub struct InterModuleErrorChecker;

impl InterModuleErrorChecker {
    /// Check error consistency across modules
    pub fn validate_error_consistency(
        module_errors: &HashMap<String, Vec<StatsError>>,
    ) -> Vec<String> {
        let mut inconsistencies = Vec::new();

        // Check for similar error patterns across modules
        let mut error_patterns: HashMap<String, Vec<String>> = HashMap::new();

        for (module, errors) in module_errors {
            for error in errors {
                let pattern = Self::extract_error_pattern(error);
                error_patterns
                    .entry(pattern)
                    .or_default()
                    .push(module.clone());
            }
        }

        // Look for patterns that should be consistent but aren't
        for (pattern, modules) in error_patterns {
            if modules.len() > 1 {
                let unique_modules: std::collections::HashSet<_> = modules.into_iter().collect();
                if unique_modules.len() > 1 {
                    inconsistencies.push(format!(
                        "Error pattern '{}' appears inconsistently across modules: {:?}",
                        pattern, unique_modules
                    ));
                }
            }
        }

        inconsistencies
    }

    fn extract_error_pattern(error: &StatsError) -> String {
        match error {
            StatsError::DimensionMismatch(_) => "dimension_mismatch".to_string(),
            StatsError::InvalidArgument(msg) if msg.contains("empty") => "empty_array".to_string(),
            StatsError::InvalidArgument(msg) if msg.contains("NaN") => "nan_values".to_string(),
            StatsError::DomainError(msg) if msg.contains("positive") => "non_positive".to_string(),
            StatsError::DomainError(msg) if msg.contains("probability") => {
                "invalid_probability".to_string()
            }
            StatsError::ConvergenceError(_) => "convergence_failure".to_string(),
            StatsError::ComputationError(msg) if msg.contains("singular") => {
                "singular_matrix".to_string()
            }
            _ => "other".to_string(),
        }
    }
}

/// Auto-recovery system for common errors
pub struct AutoRecoverySystem;

impl AutoRecoverySystem {
    /// Attempt automatic recovery for common errors
    pub fn attempt_auto_recovery(
        error: &StatsError,
        context: &EnhancedErrorContext,
    ) -> Option<RecoveryAction> {
        match error {
            StatsError::InvalidArgument(msg) if msg.contains("NaN") => Some(RecoveryAction {
                description: "Automatically remove NaN values".to_string(),
                priority: 1,
                performance_impact: PerformanceImpact::Minimal,
                code_example: Some(
                    "let cleandata = data.iter().filter(|x| x.is_finite()).collect();".to_string(),
                ),
                automatic: true,
            }),
            StatsError::DimensionMismatch(_) => {
                Some(RecoveryAction {
                    description: "Attempt automatic dimension alignment".to_string(),
                    priority: 2,
                    performance_impact: PerformanceImpact::Minimal,
                    code_example: Some(
                        "let aligneddata = data.broadcast_to(targetshape);".to_string(),
                    ),
                    automatic: false, // Usually requires user input
                })
            }
            StatsError::ComputationError(msg) if msg.contains("singular") => Some(RecoveryAction {
                description: "Add regularization to handle singularity".to_string(),
                priority: 1,
                performance_impact: PerformanceImpact::Minimal,
                code_example: Some("let regularized = matrix + Array2::eye(n) * 1e-6;".to_string()),
                automatic: true,
            }),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = ErrorMessages::length_mismatch("x", 5, "y", 3);
        assert!(err.to_string().contains("Array length mismatch"));
        assert!(err.to_string().contains("same number of elements"));
    }

    #[test]
    fn test_error_validator() {
        let empty_data: &[f64] = &[];
        assert!(ErrorValidator::validate_array(empty_data, "test").is_err());

        let finitedata = [1.0, 2.0, 3.0];
        assert!(ErrorValidator::validate_finite_array(&finitedata, "test").is_ok());

        let nandata = [1.0, f64::NAN, 3.0];
        assert!(ErrorValidator::validate_finite_array(&nandata, "test").is_err());
    }

    #[test]
    fn test_recovery_suggestions() {
        let err = ErrorMessages::empty_array("data");
        let suggestions = RecoverySuggestions::get_suggestions(&err);
        assert!(!suggestions.is_empty());
    }

    #[test]
    fn test_enhanced_error_context() {
        let data = [1.0, 2.0, f64::NAN, 4.0];
        let diagnostics = ErrorDiagnostics::diagnose_array_f64(&data, "testdata");

        assert_eq!(diagnostics.shape, vec![4]);
        assert_eq!(diagnostics.summary.nan_count, 1);
        assert_eq!(diagnostics.summary.finite_count, 3);
        assert!(diagnostics
            .quality_issues
            .contains(&DataQualityIssue::HasNaN));
    }

    #[test]
    fn test_batch_error_handler() {
        let mut handler = BatchErrorHandler::new();

        let error = ErrorMessages::empty_array("test");
        let context = EnhancedErrorContext {
            function_name: "test_function".to_string(),
            module_name: "test_module".to_string(),
            data_info: ErrorDiagnostics::diagnose_array_f64(&[], "empty"),
            system_info: ErrorDiagnostics::get_system_diagnostics(),
            recovery_actions: vec![],
        };

        handler.add_error(0, error, context);
        handler.add_warning(1, "This is a test warning".to_string());

        let report = handler.generate_batch_report();
        assert!(report.contains("Batch Processing Errors"));
        assert!(report.contains("Batch Processing Warnings"));

        let summary = handler.get_error_summary();
        assert_eq!(summary.get("Invalid Argument"), Some(&1));
    }

    #[test]
    fn test_auto_recovery_system() {
        let error = ErrorMessages::nan_detected("test context");
        let context = EnhancedErrorContext {
            function_name: "test".to_string(),
            module_name: "test".to_string(),
            data_info: ErrorDiagnostics::diagnose_array_f64(&[f64::NAN], "test"),
            system_info: ErrorDiagnostics::get_system_diagnostics(),
            recovery_actions: vec![],
        };

        let recovery = AutoRecoverySystem::attempt_auto_recovery(&error, &context);
        assert!(recovery.is_some());
        assert!(recovery.unwrap().automatic);
    }
}
