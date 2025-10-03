//! Comprehensive API consistency validation framework
//!
//! This module provides tools and infrastructure for validating API consistency
//! across the entire scirs2-stats library. It includes:
//! - Function signature validation
//! - Parameter naming consistency checks
//! - Return type standardization validation
//! - Error handling consistency verification
//! - Documentation completeness analysis
//! - Performance characteristics validation
//! - Cross-module API compatibility checks
//! - SciPy compatibility verification

use crate::error::StatsResult;
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;

/// API consistency validation framework
pub struct APIConsistencyValidator {
    /// Configuration for validation
    pub config: ValidationConfig,
    /// Validation results
    pub results: ValidationResults,
    /// Function registry for cross-module validation
    pub function_registry: FunctionRegistry,
}

/// Configuration for API validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Enable parameter naming validation
    pub validate_parameter_names: bool,
    /// Enable return type validation
    pub validate_return_types: bool,
    /// Enable error handling validation
    pub validate_error_handling: bool,
    /// Enable documentation validation
    pub validate_documentation: bool,
    /// Enable performance validation
    pub validate_performance: bool,
    /// Enable SciPy compatibility validation
    pub validate_scipy_compatibility: bool,
    /// Strict mode (fail on any inconsistency)
    pub strict_mode: bool,
    /// Custom naming conventions
    pub naming_conventions: NamingConventions,
}

/// Naming conventions for API consistency
#[derive(Debug, Clone)]
pub struct NamingConventions {
    /// Standard parameter names for common concepts
    pub parameter_names: HashMap<String, Vec<String>>,
    /// Function name patterns
    pub function_patterns: Vec<FunctionPattern>,
    /// Module naming rules
    pub module_patterns: Vec<String>,
}

/// Function pattern for naming validation
#[derive(Debug, Clone)]
pub struct FunctionPattern {
    /// Pattern description
    pub description: String,
    /// Regular expression pattern
    pub pattern: String,
    /// Category of functions this applies to
    pub category: FunctionCategory,
}

/// Categories of functions for validation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionCategory {
    /// Descriptive statistics functions
    DescriptiveStats,
    /// Statistical tests
    StatisticalTests,
    /// Distribution functions
    Distributions,
    /// Regression functions
    Regression,
    /// Correlation functions
    Correlation,
    /// Utility functions
    Utilities,
    /// Advanced algorithms
    Advanced,
}

/// Results of API validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    /// Overall validation status
    pub overall_status: ValidationStatus,
    /// Individual check results
    pub check_results: Vec<ValidationCheck>,
    /// Inconsistencies found
    pub inconsistencies: Vec<APIInconsistency>,
    /// Warnings (non-critical issues)
    pub warnings: Vec<ValidationWarning>,
    /// Summary statistics
    pub summary: ValidationSummary,
}

/// Validation status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ValidationStatus {
    /// All checks passed
    Passed,
    /// Some warnings but no critical issues
    PassedWithWarnings,
    /// Critical inconsistencies found
    Failed,
    /// Validation not yet run
    NotRun,
}

/// Individual validation check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationCheck {
    /// Name of the check
    pub name: String,
    /// Category of the check
    pub category: CheckCategory,
    /// Status of this check
    pub status: ValidationStatus,
    /// Description of what was checked
    pub description: String,
    /// Details about the check result
    pub details: String,
    /// Severity level
    pub severity: Severity,
}

/// Categories of validation checks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CheckCategory {
    /// Parameter naming consistency
    ParameterNaming,
    /// Return type consistency
    ReturnTypes,
    /// Error handling consistency
    ErrorHandling,
    /// Documentation quality
    Documentation,
    /// Performance characteristics
    Performance,
    /// Cross-module compatibility
    CrossModule,
    /// SciPy compatibility
    ScipyCompatibility,
}

/// Severity levels for validation issues
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Severity {
    /// Critical issue that breaks consistency
    Critical,
    /// Major issue that should be addressed
    Major,
    /// Minor issue or suggestion
    Minor,
    /// Informational note
    Info,
}

/// API inconsistency detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct APIInconsistency {
    /// Type of inconsistency
    pub inconsistency_type: InconsistencyType,
    /// Functions/modules involved
    pub affected_functions: Vec<String>,
    /// Description of the issue
    pub description: String,
    /// Suggested fix
    pub suggested_fix: String,
    /// Severity level
    pub severity: Severity,
    /// Impact on users
    pub impact: String,
}

/// Types of API inconsistencies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InconsistencyType {
    /// Parameter naming inconsistency
    ParameterNaming,
    /// Inconsistent return types
    ReturnTypeInconsistency,
    /// Different error handling patterns
    ErrorHandlingInconsistency,
    /// Missing or inconsistent documentation
    DocumentationInconsistency,
    /// Performance characteristic differences
    PerformanceInconsistency,
    /// SciPy compatibility issues
    ScipyCompatibilityIssue,
}

/// Validation warning (non-critical issue)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    /// Warning message
    pub message: String,
    /// Category of warning
    pub category: CheckCategory,
    /// Affected function/module
    pub location: String,
    /// Suggestion for improvement
    pub suggestion: String,
}

/// Summary of validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    /// Total number of checks performed
    pub total_checks: usize,
    /// Number of passed checks
    pub passed_checks: usize,
    /// Number of failed checks
    pub failed_checks: usize,
    /// Number of warnings
    pub warning_count: usize,
    /// Number of critical issues
    pub critical_issues: usize,
    /// Overall consistency score (0-100)
    pub consistency_score: f64,
}

/// Function registry for cross-module validation
#[derive(Debug, Clone)]
pub struct FunctionRegistry {
    /// Registered functions by module
    pub functions_by_module: HashMap<String, Vec<FunctionSignature>>,
    /// Functions by category
    pub functions_by_category: HashMap<FunctionCategory, Vec<FunctionSignature>>,
    /// Parameter usage statistics
    pub parameter_usage: HashMap<String, ParameterUsage>,
}

/// Function signature information
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    /// Function name
    pub name: String,
    /// Module containing the function
    pub module: String,
    /// Function category
    pub category: FunctionCategory,
    /// Parameter information
    pub parameters: Vec<ParameterInfo>,
    /// Return type information
    pub return_type: ReturnTypeInfo,
    /// Error types that can be returned
    pub error_types: Vec<String>,
    /// Documentation status
    pub documentation_status: DocumentationStatus,
}

/// Parameter information
#[derive(Debug, Clone)]
pub struct ParameterInfo {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Whether parameter is optional
    pub optional: bool,
    /// Default value if any
    pub default_value: Option<String>,
    /// Parameter description
    pub description: String,
}

/// Return type information
#[derive(Debug, Clone)]
pub struct ReturnTypeInfo {
    /// Return type name
    pub type_name: String,
    /// Whether return is wrapped in Result
    pub is_result: bool,
    /// Generic type parameters
    pub generic_params: Vec<String>,
    /// Description of return value
    pub description: String,
}

/// Documentation status
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentationStatus {
    /// Complete documentation
    Complete,
    /// Partial documentation
    Partial,
    /// Missing documentation
    Missing,
    /// Documentation present but poor quality
    PoorQuality,
}

/// Parameter usage statistics
#[derive(Debug, Clone)]
pub struct ParameterUsage {
    /// Parameter name
    pub name: String,
    /// Number of functions using this parameter
    pub usage_count: usize,
    /// Different type signatures seen
    pub type_signatures: HashSet<String>,
    /// Modules where this parameter is used
    pub modules: HashSet<String>,
    /// Common alternative names
    pub alternative_names: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            validate_parameter_names: true,
            validate_return_types: true,
            validate_error_handling: true,
            validate_documentation: true,
            validate_performance: false, // Expensive, off by default
            validate_scipy_compatibility: true,
            strict_mode: false,
            naming_conventions: NamingConventions::default(),
        }
    }
}

impl Default for NamingConventions {
    fn default() -> Self {
        let mut parameter_names = HashMap::new();

        // Standard parameter names for common statistical concepts
        parameter_names.insert(
            "data".to_string(),
            vec!["data".to_string(), "x".to_string(), "array".to_string()],
        );
        parameter_names.insert(
            "axis".to_string(),
            vec!["axis".to_string(), "dim".to_string()],
        );
        parameter_names.insert(
            "degrees_of_freedom".to_string(),
            vec!["ddof".to_string(), "df".to_string()],
        );
        parameter_names.insert(
            "alpha".to_string(),
            vec!["alpha".to_string(), "significance".to_string()],
        );
        parameter_names.insert(
            "alternative".to_string(),
            vec!["alternative".to_string(), "alt".to_string()],
        );
        parameter_names.insert(
            "method".to_string(),
            vec!["method".to_string(), "algorithm".to_string()],
        );

        let function_patterns = vec![
            FunctionPattern {
                description: "Statistical test functions should end with 'test'".to_string(),
                pattern: r".*test$".to_string(),
                category: FunctionCategory::StatisticalTests,
            },
            FunctionPattern {
                description: "Correlation functions should contain 'corr'".to_string(),
                pattern: r".*corr.*".to_string(),
                category: FunctionCategory::Correlation,
            },
            FunctionPattern {
                description: "Distribution functions should be named after the distribution"
                    .to_string(),
                pattern: r"^[a-z_]+$".to_string(),
                category: FunctionCategory::Distributions,
            },
        ];

        Self {
            parameter_names,
            function_patterns,
            module_patterns: vec!["^[a-z_]+$".to_string()],
        }
    }
}

impl APIConsistencyValidator {
    /// Create new API consistency validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            results: ValidationResults::new(),
            function_registry: FunctionRegistry::new(),
        }
    }

    /// Run complete API consistency validation
    pub fn validate_all(&mut self) -> StatsResult<&ValidationResults> {
        // Reset results
        self.results = ValidationResults::new();

        // Run individual validation checks
        if self.config.validate_parameter_names {
            self.validate_parameter_naming()?;
        }

        if self.config.validate_return_types {
            self.validate_return_type_consistency()?;
        }

        if self.config.validate_error_handling {
            self.validate_error_handling_consistency()?;
        }

        if self.config.validate_documentation {
            self.validate_documentation_quality()?;
        }

        if self.config.validate_performance {
            self.validate_performance_characteristics()?;
        }

        if self.config.validate_scipy_compatibility {
            self.validate_scipy_compatibility()?;
        }

        // Compute summary
        self.compute_validation_summary();

        // Determine overall status
        self.results.overall_status = self.determine_overall_status();

        Ok(&self.results)
    }

    /// Validate parameter naming consistency
    fn validate_parameter_naming(&mut self) -> StatsResult<()> {
        let mut check = ValidationCheck {
            name: "Parameter Naming Consistency".to_string(),
            category: CheckCategory::ParameterNaming,
            status: ValidationStatus::NotRun,
            description: "Checking consistency of parameter names across functions".to_string(),
            details: String::new(),
            severity: Severity::Major,
        };

        // Analyze parameter usage patterns
        let mut parameter_analysis = HashMap::new();

        for (module, functions) in &self.function_registry.functions_by_module {
            for function in functions {
                for param in &function.parameters {
                    let entry = parameter_analysis
                        .entry(param.name.clone())
                        .or_insert_with(ParameterUsageAnalysis::new);

                    entry.add_usage(
                        module.clone(),
                        function.name.clone(),
                        param.param_type.clone(),
                    );
                }
            }
        }

        // Check for inconsistencies
        let mut inconsistencies_found = 0;
        let mut details = Vec::new();

        for (param_name, analysis) in &parameter_analysis {
            if analysis.type_variations.len() > 1 {
                // Same parameter name used with different types
                let inconsistency = APIInconsistency {
                    inconsistency_type: InconsistencyType::ParameterNaming,
                    affected_functions: analysis.functions.clone(),
                    description: format!(
                        "Parameter '{}' used with different types: {:?}",
                        param_name, analysis.type_variations
                    ),
                    suggested_fix: format!(
                        "Standardize parameter '{}' to use consistent type across all functions",
                        param_name
                    ),
                    severity: Severity::Major,
                    impact: "May cause confusion for users and complicate generic programming"
                        .to_string(),
                };

                self.results.inconsistencies.push(inconsistency);
                inconsistencies_found += 1;
            }

            // Check against naming conventions
            if let Some(standard_names) = self
                .config
                .naming_conventions
                .parameter_names
                .get(param_name)
            {
                if !standard_names.contains(param_name) {
                    let warning = ValidationWarning {
                        message: format!(
                            "Parameter '{}' doesn't follow naming convention. Consider: {:?}",
                            param_name, standard_names
                        ),
                        category: CheckCategory::ParameterNaming,
                        location: format!("Functions: {:?}", analysis.functions),
                        suggestion: format!("Use one of the standard names: {:?}", standard_names),
                    };

                    self.results.warnings.push(warning);
                }
            }
        }

        details.push(format!(
            "Analyzed {} unique parameter names",
            parameter_analysis.len()
        ));
        details.push(format!(
            "Found {} type inconsistencies",
            inconsistencies_found
        ));

        check.details = details.join("; ");
        check.status = if inconsistencies_found == 0 {
            ValidationStatus::Passed
        } else if self.config.strict_mode {
            ValidationStatus::Failed
        } else {
            ValidationStatus::PassedWithWarnings
        };

        self.results.check_results.push(check);
        Ok(())
    }

    /// Validate return type consistency
    fn validate_return_type_consistency(&mut self) -> StatsResult<()> {
        let mut check = ValidationCheck {
            name: "Return Type Consistency".to_string(),
            category: CheckCategory::ReturnTypes,
            status: ValidationStatus::NotRun,
            description: "Checking consistency of return types across similar functions"
                .to_string(),
            details: String::new(),
            severity: Severity::Major,
        };

        let mut return_type_patterns = HashMap::new();
        let mut inconsistencies_found = 0;

        // Group functions by category and analyze return types
        for (category, functions) in &self.function_registry.functions_by_category {
            let mut category_return_types = HashSet::new();

            for function in functions {
                category_return_types.insert(function.return_type.type_name.clone());
            }

            return_type_patterns.insert(category.clone(), category_return_types);

            // Check for excessive variation in return types within category
            if functions.len() > 3 && return_type_patterns[category].len() > functions.len() / 2 {
                let inconsistency = APIInconsistency {
                    inconsistency_type: InconsistencyType::ReturnTypeInconsistency,
                    affected_functions: functions.iter().map(|f| f.name.clone()).collect(),
                    description: format!(
                        "Functions in {:?} category have inconsistent return types: {:?}",
                        category, return_type_patterns[category]
                    ),
                    suggested_fix: "Standardize return types within functional categories"
                        .to_string(),
                    severity: Severity::Major,
                    impact: "Makes API harder to learn and use consistently".to_string(),
                };

                self.results.inconsistencies.push(inconsistency);
                inconsistencies_found += 1;
            }
        }

        // Check Result<T> usage consistency
        let mut result_usage_patterns = HashMap::new();
        for (module, functions) in &self.function_registry.functions_by_module {
            let result_count = functions.iter().filter(|f| f.return_type.is_result).count();
            let total_count = functions.len();

            result_usage_patterns.insert(module.clone(), (result_count, total_count));
        }

        check.details = format!(
            "Analyzed return types across {} categories, found {} inconsistencies",
            return_type_patterns.len(),
            inconsistencies_found
        );

        check.status = if inconsistencies_found == 0 {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        };

        self.results.check_results.push(check);
        Ok(())
    }

    /// Validate error handling consistency
    fn validate_error_handling_consistency(&mut self) -> StatsResult<()> {
        let mut check = ValidationCheck {
            name: "Error Handling Consistency".to_string(),
            category: CheckCategory::ErrorHandling,
            status: ValidationStatus::NotRun,
            description: "Checking consistency of error handling patterns".to_string(),
            details: String::new(),
            severity: Severity::Critical,
        };

        let mut error_patterns = HashMap::new();
        let mut inconsistencies_found = 0;

        // Analyze error handling patterns
        for (module, functions) in &self.function_registry.functions_by_module {
            for function in functions {
                // Check if function returns Result
                if !function.return_type.is_result
                    && function.category != FunctionCategory::Utilities
                {
                    let warning = ValidationWarning {
                        message: format!(
                            "Function '{}' in module '{}' doesn't return Result<T> but may fail",
                            function.name, module
                        ),
                        category: CheckCategory::ErrorHandling,
                        location: format!("{}::{}", module, function.name),
                        suggestion: "Consider returning Result<T> for fallible operations"
                            .to_string(),
                    };

                    self.results.warnings.push(warning);
                }

                // Collect error type patterns
                for error_type in &function.error_types {
                    error_patterns
                        .entry(error_type.clone())
                        .or_insert_with(Vec::new)
                        .push(format!("{}::{}", module, function.name));
                }
            }
        }

        // Check for consistent error types
        let expected_error_types = ["StatsError".to_string()];
        for (error_type, functions) in &error_patterns {
            if !expected_error_types.contains(error_type) {
                let inconsistency = APIInconsistency {
                    inconsistency_type: InconsistencyType::ErrorHandlingInconsistency,
                    affected_functions: functions.clone(),
                    description: format!(
                        "Non-standard error type '{}' used in functions: {:?}",
                        error_type, functions
                    ),
                    suggested_fix: "Use StatsError for consistent error handling".to_string(),
                    severity: Severity::Major,
                    impact: "Inconsistent error handling makes error recovery difficult"
                        .to_string(),
                };

                self.results.inconsistencies.push(inconsistency);
                inconsistencies_found += 1;
            }
        }

        check.details = format!(
            "Analyzed error handling in {} functions, found {} inconsistencies",
            self.function_registry
                .functions_by_module
                .values()
                .map(|funcs| funcs.len())
                .sum::<usize>(),
            inconsistencies_found
        );

        check.status = if inconsistencies_found == 0 {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        };

        self.results.check_results.push(check);
        Ok(())
    }

    /// Validate documentation quality
    fn validate_documentation_quality(&mut self) -> StatsResult<()> {
        let mut check = ValidationCheck {
            name: "Documentation Quality".to_string(),
            category: CheckCategory::Documentation,
            status: ValidationStatus::NotRun,
            description: "Checking completeness and quality of documentation".to_string(),
            details: String::new(),
            severity: Severity::Minor,
        };

        let mut doc_stats = DocumentationStats::new();
        let mut poorly_documented_functions = Vec::new();

        for (module, functions) in &self.function_registry.functions_by_module {
            for function in functions {
                doc_stats.total_functions += 1;

                match function.documentation_status {
                    DocumentationStatus::Complete => doc_stats.complete_docs += 1,
                    DocumentationStatus::Partial => doc_stats.partial_docs += 1,
                    DocumentationStatus::Missing => {
                        doc_stats.missing_docs += 1;
                        poorly_documented_functions.push(format!("{}::{}", module, function.name));
                    }
                    DocumentationStatus::PoorQuality => {
                        doc_stats.poor_quality_docs += 1;
                        poorly_documented_functions.push(format!("{}::{}", module, function.name));
                    }
                }
            }
        }

        // Check documentation completeness
        let completion_rate = doc_stats.complete_docs as f64 / doc_stats.total_functions as f64;

        if completion_rate < 0.8 {
            let inconsistency = APIInconsistency {
                inconsistency_type: InconsistencyType::DocumentationInconsistency,
                affected_functions: poorly_documented_functions,
                description: format!(
                    "Documentation completion rate is {:.1}%, below 80% threshold",
                    completion_rate * 100.0
                ),
                suggested_fix: "Add comprehensive documentation to all public functions"
                    .to_string(),
                severity: Severity::Minor,
                impact: "Poor documentation reduces library usability and adoption".to_string(),
            };

            self.results.inconsistencies.push(inconsistency);
        }

        check.details = format!(
            "Documentation completion: {:.1}% ({}/{} functions documented)",
            completion_rate * 100.0,
            doc_stats.complete_docs,
            doc_stats.total_functions
        );

        check.status = if completion_rate >= 0.8 {
            ValidationStatus::Passed
        } else {
            ValidationStatus::PassedWithWarnings
        };

        self.results.check_results.push(check);
        Ok(())
    }

    /// Validate performance characteristics
    fn validate_performance_characteristics(&mut self) -> StatsResult<()> {
        let mut check = ValidationCheck {
            name: "Performance Characteristics".to_string(),
            category: CheckCategory::Performance,
            status: ValidationStatus::NotRun,
            description: "Checking consistency of performance characteristics".to_string(),
            details: String::new(),
            severity: Severity::Minor,
        };

        // This would involve benchmarking similar functions and comparing performance
        // For now, we'll do a basic check of function complexity patterns

        let mut performance_issues = Vec::new();

        // Check for functions that should have SIMD variants
        for (module, functions) in &self.function_registry.functions_by_module {
            for function in functions {
                if function.category == FunctionCategory::DescriptiveStats
                    && !function.name.contains("simd")
                    && !function.name.contains("parallel")
                {
                    // Check if SIMD variant exists
                    let simd_variant_name = format!("{}_simd", function.name);
                    let has_simd_variant = functions.iter().any(|f| f.name == simd_variant_name);

                    if !has_simd_variant {
                        performance_issues.push(format!("{}::{}", module, function.name));
                    }
                }
            }
        }

        if !performance_issues.is_empty() {
            let warning = ValidationWarning {
                message: format!(
                    "{} functions might benefit from SIMD optimization",
                    performance_issues.len()
                ),
                category: CheckCategory::Performance,
                location: "Various modules".to_string(),
                suggestion:
                    "Consider implementing SIMD variants for performance-critical functions"
                        .to_string(),
            };

            self.results.warnings.push(warning);
        }

        check.details = format!(
            "Analyzed performance patterns, identified {} optimization opportunities",
            performance_issues.len()
        );

        check.status = ValidationStatus::Passed; // Performance issues are warnings, not failures

        self.results.check_results.push(check);
        Ok(())
    }

    /// Validate SciPy compatibility
    fn validate_scipy_compatibility(&mut self) -> StatsResult<()> {
        let mut check = ValidationCheck {
            name: "SciPy Compatibility".to_string(),
            category: CheckCategory::ScipyCompatibility,
            status: ValidationStatus::NotRun,
            description: "Checking compatibility with SciPy API patterns".to_string(),
            details: String::new(),
            severity: Severity::Major,
        };

        // Define expected SciPy-compatible function signatures
        let scipy_functions = self.get_expected_scipy_functions();
        let mut compatibility_issues = Vec::new();

        for (scipy_name, expected_sig) in &scipy_functions {
            // Check if we have this function
            let mut found = false;
            for functions in self.function_registry.functions_by_module.values() {
                for function in functions {
                    if function.name == *scipy_name {
                        found = true;

                        // Check parameter compatibility
                        if !self.check_parameter_compatibility(
                            &function.parameters,
                            &expected_sig.parameters,
                        ) {
                            compatibility_issues.push(format!(
                                "Function '{}' has incompatible parameters with SciPy",
                                scipy_name
                            ));
                        }
                        break;
                    }
                }
                if found {
                    break;
                }
            }

            if !found {
                compatibility_issues.push(format!(
                    "Missing SciPy-compatible function: '{}'",
                    scipy_name
                ));
            }
        }

        if !compatibility_issues.is_empty() {
            let inconsistency = APIInconsistency {
                inconsistency_type: InconsistencyType::ScipyCompatibilityIssue,
                affected_functions: compatibility_issues.clone(),
                description: format!(
                    "Found {} SciPy compatibility issues",
                    compatibility_issues.len()
                ),
                suggested_fix:
                    "Implement missing functions or adjust parameters for SciPy compatibility"
                        .to_string(),
                severity: Severity::Major,
                impact: "Reduces ease of migration from SciPy".to_string(),
            };

            self.results.inconsistencies.push(inconsistency);
        }

        check.details = format!(
            "Checked {} SciPy functions, found {} compatibility issues",
            scipy_functions.len(),
            compatibility_issues.len()
        );

        check.status = if compatibility_issues.is_empty() {
            ValidationStatus::Passed
        } else {
            ValidationStatus::Failed
        };

        self.results.check_results.push(check);
        Ok(())
    }

    /// Get expected SciPy-compatible function signatures
    fn get_expected_scipy_functions(&self) -> HashMap<String, FunctionSignature> {
        let mut functions = HashMap::new();

        // Add some key SciPy functions for compatibility checking
        functions.insert(
            "mean".to_string(),
            FunctionSignature {
                name: "mean".to_string(),
                module: "stats".to_string(),
                category: FunctionCategory::DescriptiveStats,
                parameters: vec![ParameterInfo {
                    name: "data".to_string(),
                    param_type: "ArrayView1<F>".to_string(),
                    optional: false,
                    default_value: None,
                    description: "Input data".to_string(),
                }],
                return_type: ReturnTypeInfo {
                    type_name: "StatsResult<F>".to_string(),
                    is_result: true,
                    generic_params: vec!["F".to_string()],
                    description: "Mean value".to_string(),
                },
                error_types: vec!["StatsError".to_string()],
                documentation_status: DocumentationStatus::Complete,
            },
        );

        functions.insert(
            "std".to_string(),
            FunctionSignature {
                name: "std".to_string(),
                module: "stats".to_string(),
                category: FunctionCategory::DescriptiveStats,
                parameters: vec![
                    ParameterInfo {
                        name: "data".to_string(),
                        param_type: "ArrayView1<F>".to_string(),
                        optional: false,
                        default_value: None,
                        description: "Input data".to_string(),
                    },
                    ParameterInfo {
                        name: "ddof".to_string(),
                        param_type: "usize".to_string(),
                        optional: true,
                        default_value: Some("0".to_string()),
                        description: "Delta degrees of freedom".to_string(),
                    },
                ],
                return_type: ReturnTypeInfo {
                    type_name: "StatsResult<F>".to_string(),
                    is_result: true,
                    generic_params: vec!["F".to_string()],
                    description: "Standard deviation".to_string(),
                },
                error_types: vec!["StatsError".to_string()],
                documentation_status: DocumentationStatus::Complete,
            },
        );

        // Add more SciPy functions as needed...

        functions
    }

    /// Check parameter compatibility between actual and expected signatures
    fn check_parameter_compatibility(
        &self,
        actual: &[ParameterInfo],
        expected: &[ParameterInfo],
    ) -> bool {
        if actual.len() != expected.len() {
            return false;
        }

        for (actual_param, expected_param) in actual.iter().zip(expected.iter()) {
            if actual_param.name != expected_param.name {
                return false;
            }

            // Basic type compatibility check (simplified)
            if actual_param.param_type != expected_param.param_type {
                return false;
            }
        }

        true
    }

    /// Compute validation summary
    fn compute_validation_summary(&mut self) {
        let total_checks = self.results.check_results.len();
        let passed_checks = self
            .results
            .check_results
            .iter()
            .filter(|c| c.status == ValidationStatus::Passed)
            .count();
        let failed_checks = self
            .results
            .check_results
            .iter()
            .filter(|c| c.status == ValidationStatus::Failed)
            .count();
        let warning_count = self.results.warnings.len();
        let critical_issues = self
            .results
            .inconsistencies
            .iter()
            .filter(|i| i.severity == Severity::Critical)
            .count();

        // Calculate consistency score
        let consistency_score = if total_checks > 0 {
            let base_score = (passed_checks as f64 / total_checks as f64) * 100.0;
            let warning_penalty = (warning_count as f64 * 2.0).min(20.0);
            let critical_penalty = (critical_issues as f64 * 10.0).min(50.0);

            (base_score - warning_penalty - critical_penalty).max(0.0)
        } else {
            0.0
        };

        self.results.summary = ValidationSummary {
            total_checks,
            passed_checks,
            failed_checks,
            warning_count,
            critical_issues,
            consistency_score,
        };
    }

    /// Determine overall validation status
    fn determine_overall_status(&self) -> ValidationStatus {
        let critical_issues = self
            .results
            .inconsistencies
            .iter()
            .any(|i| i.severity == Severity::Critical);

        let failed_checks = self
            .results
            .check_results
            .iter()
            .any(|c| c.status == ValidationStatus::Failed);

        if critical_issues || (failed_checks && self.config.strict_mode) {
            ValidationStatus::Failed
        } else if !self.results.warnings.is_empty() || failed_checks {
            ValidationStatus::PassedWithWarnings
        } else {
            ValidationStatus::Passed
        }
    }

    /// Register a function for validation
    pub fn register_function(&mut self, functionsig: FunctionSignature) {
        // Add to module registry
        self.function_registry
            .functions_by_module
            .entry(functionsig.module.clone())
            .or_default()
            .push(functionsig.clone());

        // Add to category registry
        self.function_registry
            .functions_by_category
            .entry(functionsig.category.clone())
            .or_default()
            .push(functionsig.clone());

        // Update parameter usage statistics
        for param in &functionsig.parameters {
            let usage = self
                .function_registry
                .parameter_usage
                .entry(param.name.clone())
                .or_insert_with(|| ParameterUsage {
                    name: param.name.clone(),
                    usage_count: 0,
                    type_signatures: HashSet::new(),
                    modules: HashSet::new(),
                    alternative_names: Vec::new(),
                });

            usage.usage_count += 1;
            usage.type_signatures.insert(param.param_type.clone());
            usage.modules.insert(functionsig.module.clone());
        }
    }

    /// Generate validation report
    pub fn generate_report(&self) -> ValidationReport {
        ValidationReport::new(&self.results)
    }
}

impl ValidationResults {
    fn new() -> Self {
        Self {
            overall_status: ValidationStatus::NotRun,
            check_results: Vec::new(),
            inconsistencies: Vec::new(),
            warnings: Vec::new(),
            summary: ValidationSummary {
                total_checks: 0,
                passed_checks: 0,
                failed_checks: 0,
                warning_count: 0,
                critical_issues: 0,
                consistency_score: 0.0,
            },
        }
    }
}

impl FunctionRegistry {
    fn new() -> Self {
        Self {
            functions_by_module: HashMap::new(),
            functions_by_category: HashMap::new(),
            parameter_usage: HashMap::new(),
        }
    }
}

/// Parameter usage analysis helper
#[derive(Debug)]
struct ParameterUsageAnalysis {
    functions: Vec<String>,
    modules: HashSet<String>,
    type_variations: HashSet<String>,
}

impl ParameterUsageAnalysis {
    fn new() -> Self {
        Self {
            functions: Vec::new(),
            modules: HashSet::new(),
            type_variations: HashSet::new(),
        }
    }

    fn add_usage(&mut self, module: String, function: String, paramtype: String) {
        self.functions.push(format!("{}::{}", module, function));
        self.modules.insert(module);
        self.type_variations.insert(paramtype);
    }
}

/// Documentation statistics helper
#[derive(Debug)]
struct DocumentationStats {
    total_functions: usize,
    complete_docs: usize,
    partial_docs: usize,
    missing_docs: usize,
    poor_quality_docs: usize,
}

impl DocumentationStats {
    fn new() -> Self {
        Self {
            total_functions: 0,
            complete_docs: 0,
            partial_docs: 0,
            missing_docs: 0,
            poor_quality_docs: 0,
        }
    }
}

/// Validation report generator
pub struct ValidationReport {
    results: ValidationResults,
}

impl ValidationReport {
    pub fn new(results: &ValidationResults) -> Self {
        Self {
            results: results.clone(),
        }
    }

    /// Generate markdown report
    pub fn to_markdown(&self) -> String {
        let mut report = String::new();

        report.push_str("# API Consistency Validation Report\n\n");

        // Overall status
        report.push_str(&format!(
            "**Overall Status:** {:?}\n",
            self.results.overall_status
        ));
        report.push_str(&format!(
            "**Consistency Score:** {:.1}%\n\n",
            self.results.summary.consistency_score
        ));

        // Summary
        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "- **Total Checks:** {}\n",
            self.results.summary.total_checks
        ));
        report.push_str(&format!(
            "- **Passed:** {}\n",
            self.results.summary.passed_checks
        ));
        report.push_str(&format!(
            "- **Failed:** {}\n",
            self.results.summary.failed_checks
        ));
        report.push_str(&format!(
            "- **Warnings:** {}\n",
            self.results.summary.warning_count
        ));
        report.push_str(&format!(
            "- **Critical Issues:** {}\n\n",
            self.results.summary.critical_issues
        ));

        // Individual checks
        report.push_str("## Check Results\n\n");
        for check in &self.results.check_results {
            report.push_str(&format!("### {}\n", check.name));
            report.push_str(&format!("**Status:** {:?}\n", check.status));
            report.push_str(&format!("**Category:** {:?}\n", check.category));
            report.push_str(&format!("**Description:** {}\n", check.description));
            report.push_str(&format!("**Details:** {}\n\n", check.details));
        }

        // Inconsistencies
        if !self.results.inconsistencies.is_empty() {
            report.push_str("## Inconsistencies Found\n\n");
            for (i, inconsistency) in self.results.inconsistencies.iter().enumerate() {
                report.push_str(&format!("### Issue {}\n", i + 1));
                report.push_str(&format!(
                    "**Type:** {:?}\n",
                    inconsistency.inconsistency_type
                ));
                report.push_str(&format!("**Severity:** {:?}\n", inconsistency.severity));
                report.push_str(&format!("**Description:** {}\n", inconsistency.description));
                report.push_str(&format!(
                    "**Suggested Fix:** {}\n",
                    inconsistency.suggested_fix
                ));
                report.push_str(&format!("**Impact:** {}\n\n", inconsistency.impact));
            }
        }

        // Warnings
        if !self.results.warnings.is_empty() {
            report.push_str("## Warnings\n\n");
            for warning in &self.results.warnings {
                report.push_str(&format!(
                    "- **{}:** {} ({})\n",
                    warning.location, warning.message, warning.suggestion
                ));
            }
        }

        report
    }

    /// Generate JSON report
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.results)
    }
}

/// Convenience function to validate API consistency
#[allow(dead_code)]
pub fn validate_api_consistency(
    config: Option<ValidationConfig>,
) -> StatsResult<ValidationResults> {
    let config = config.unwrap_or_default();
    let mut validator = APIConsistencyValidator::new(config);

    // In a real implementation, this would automatically discover and register
    // all functions in the library. For now, we'll return basic results.

    validator.validate_all()?;
    Ok(validator.results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!(config.validate_parameter_names);
        assert!(config.validate_return_types);
        assert!(config.validate_error_handling);
        assert!(!config.strict_mode);
    }

    #[test]
    fn test_function_registry() {
        let registry = FunctionRegistry::new();

        let functionsig = FunctionSignature {
            name: "mean".to_string(),
            module: "descriptive".to_string(),
            category: FunctionCategory::DescriptiveStats,
            parameters: vec![ParameterInfo {
                name: "data".to_string(),
                param_type: "ArrayView1<f64>".to_string(),
                optional: false,
                default_value: None,
                description: "Input data".to_string(),
            }],
            return_type: ReturnTypeInfo {
                type_name: "StatsResult<f64>".to_string(),
                is_result: true,
                generic_params: vec!["f64".to_string()],
                description: "Mean value".to_string(),
            },
            error_types: vec!["StatsError".to_string()],
            documentation_status: DocumentationStatus::Complete,
        };

        let mut validator = APIConsistencyValidator::new(ValidationConfig::default());
        validator.register_function(functionsig);

        assert_eq!(validator.function_registry.functions_by_module.len(), 1);
        assert_eq!(validator.function_registry.functions_by_category.len(), 1);
    }

    #[test]
    fn test_validation_report() {
        let results = ValidationResults::new();
        let report = ValidationReport::new(&results);
        let markdown = report.to_markdown();

        assert!(markdown.contains("API Consistency Validation Report"));
        assert!(markdown.contains("Overall Status"));
    }
}
