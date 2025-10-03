//! Enhanced error handling and recovery system
//!
//! This module provides comprehensive error handling with intelligent recovery
//! suggestions, detailed diagnostics, and context-aware error reporting.

use crate::error::StatsError;
use num_cpus;
use std::collections::HashMap;
use std::fmt;

/// Enhanced error with recovery suggestions
#[derive(Debug, Clone)]
pub struct EnhancedStatsError {
    /// Original error
    pub error: StatsError,
    /// Context information
    pub context: ErrorContext,
    /// Recovery suggestions
    pub recovery_suggestions: Vec<RecoverySuggestion>,
    /// Related documentation links
    pub documentation_links: Vec<String>,
    /// Example code snippets
    pub example_snippets: Vec<CodeSnippet>,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Performance impact assessment
    pub performance_impact: PerformanceImpact,
}

/// Error context information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Function name where error occurred
    pub function_name: String,
    /// Module path
    pub module_path: String,
    /// Input data characteristics
    pub data_characteristics: DataCharacteristics,
    /// System information
    pub system_info: SystemInfo,
    /// Computation state
    pub computation_state: ComputationState,
}

/// Data characteristics for error context
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    /// Data size information
    pub size_info: Option<SizeInfo>,
    /// Data type information
    pub type_info: String,
    /// Data range information
    pub range_info: Option<RangeInfo>,
    /// Missing data information
    pub missingdata_info: Option<MissingDataInfo>,
    /// Data distribution characteristics
    pub distribution_info: Option<DistributionInfo>,
}

/// Size information for arrays/matrices
#[derive(Debug, Clone)]
pub struct SizeInfo {
    /// Number of elements
    pub n_elements: usize,
    /// Shape (for multidimensional arrays)
    pub shape: Vec<usize>,
    /// Memory usage estimate
    pub memory_usage_mb: f64,
}

/// Range information for numerical data
#[derive(Debug, Clone)]
pub struct RangeInfo {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Has infinite values
    pub has_infinite: bool,
    /// Has NaN values
    pub has_nan: bool,
    /// Has zero values
    pub has_zero: bool,
}

/// Missing data information
#[derive(Debug, Clone)]
pub struct MissingDataInfo {
    /// Number of missing values
    pub count: usize,
    /// Percentage of missing values
    pub percentage: f64,
    /// Pattern of missingness
    pub pattern: MissingPattern,
}

/// Missing data patterns
#[derive(Debug, Clone, PartialEq)]
pub enum MissingPattern {
    /// Missing completely at random
    MCAR,
    /// Missing at random
    MAR,
    /// Missing not at random
    MNAR,
    /// Unknown pattern
    Unknown,
}

/// Distribution characteristics
#[derive(Debug, Clone)]
pub struct DistributionInfo {
    /// Estimated mean
    pub mean: Option<f64>,
    /// Estimated variance
    pub variance: Option<f64>,
    /// Estimated skewness
    pub skewness: Option<f64>,
    /// Estimated kurtosis
    pub kurtosis: Option<f64>,
    /// Suspected distribution family
    pub suspected_family: Option<String>,
}

/// System information
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Available memory
    pub available_memory_mb: Option<f64>,
    /// Number of CPU cores
    pub cpu_cores: Option<usize>,
    /// SIMD capabilities
    pub simd_capabilities: Vec<String>,
    /// Parallel processing availability
    pub parallel_available: bool,
}

/// Computation state
#[derive(Debug, Clone)]
pub struct ComputationState {
    /// Algorithm being used
    pub algorithm: String,
    /// Iteration number (if iterative)
    pub iteration: Option<usize>,
    /// Convergence status
    pub convergence_status: Option<ConvergenceStatus>,
    /// Current tolerance
    pub current_tolerance: Option<f64>,
    /// Intermediate results
    pub intermediate_results: HashMap<String, f64>,
}

/// Convergence status
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceStatus {
    /// Not started
    NotStarted,
    /// In progress
    InProgress,
    /// Converged successfully
    Converged,
    /// Failed to converge
    FailedToConverge,
    /// Diverged
    Diverged,
}

/// Recovery suggestion with specific actions
#[derive(Debug, Clone)]
pub struct RecoverySuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Description of the suggestion
    pub description: String,
    /// Specific action to take
    pub action: RecoveryAction,
    /// Expected outcome
    pub expected_outcome: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Prerequisites for this suggestion
    pub prerequisites: Vec<String>,
}

/// Types of recovery suggestions
#[derive(Debug, Clone, PartialEq)]
pub enum SuggestionType {
    /// Change algorithm parameters
    ParameterAdjustment,
    /// Switch to different algorithm
    AlgorithmChange,
    /// Preprocess data
    DataPreprocessing,
    /// Increase computational resources
    ResourceIncrease,
    /// Use approximation method
    Approximation,
    /// Check input validation
    InputValidation,
    /// Memory optimization
    MemoryOptimization,
    /// Numerical stability improvement
    NumericalStability,
}

/// Specific recovery actions
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Adjust parameter to specific value
    AdjustParameter {
        parameter_name: String,
        current_value: String,
        suggested_value: String,
        explanation: String,
    },
    /// Change algorithm
    ChangeAlgorithm {
        current_algorithm: String,
        suggested_algorithm: String,
        reasons: Vec<String>,
    },
    /// Preprocess data
    PreprocessData {
        preprocessing_steps: Vec<PreprocessingStep>,
    },
    /// Scale computation
    ScaleComputation {
        current_approach: String,
        suggested_approach: String,
        expected_improvement: String,
    },
    /// Validate inputs
    ValidateInputs {
        validation_checks: Vec<ValidationCheck>,
    },
    /// Simple data preprocessing (without detailed steps)
    SimplePreprocessData,
    /// Simple input validation (without detailed checks)
    SimpleValidateInputs,
    /// Adjust tolerance for convergence
    AdjustTolerance { new_tolerance: f64 },
    /// Increase maximum iterations
    IncreaseIterations { factor: f64 },
    /// Switch to different algorithm
    SwitchAlgorithm { new_algorithm: String },
    /// Enable parallel processing
    EnableParallelProcessing { num_threads: usize },
    /// Use chunked processing for large data
    UseChunkedProcessing { chunksize: usize },
    /// Apply regularization
    ApplyRegularization { regularization_strength: f64 },
    /// Reduce precision for speed
    ReducePrecision { new_precision: String },
    /// Use approximation methods
    UseApproximation { approximation_method: String },
}

/// Data preprocessing steps
#[derive(Debug, Clone)]
pub struct PreprocessingStep {
    /// Step name
    pub name: String,
    /// Description
    pub description: String,
    /// Code example
    pub code_example: String,
    /// Expected impact
    pub expected_impact: String,
}

/// Input validation checks
#[derive(Debug, Clone)]
pub struct ValidationCheck {
    /// Check name
    pub name: String,
    /// Condition to check
    pub condition: String,
    /// How to fix if failed
    pub fix_suggestion: String,
}

/// Code snippet for examples
#[derive(Debug, Clone)]
pub struct CodeSnippet {
    /// Title of the snippet
    pub title: String,
    /// Code content
    pub code: String,
    /// Language (usually "rust")
    pub language: String,
    /// Description
    pub description: String,
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum ErrorSeverity {
    /// Low severity - operation can often continue
    Low,
    /// Medium severity - may impact results
    Medium,
    /// High severity - computation cannot proceed
    High,
    /// Critical severity - system-level issue
    Critical,
}

/// Performance impact assessment
#[derive(Debug, Clone)]
pub struct PerformanceImpact {
    /// Expected memory usage change
    pub memory_impact: ImpactLevel,
    /// Expected computation time change
    pub time_impact: ImpactLevel,
    /// Expected accuracy impact
    pub accuracy_impact: ImpactLevel,
    /// Scalability concerns
    pub scalability_impact: ImpactLevel,
}

/// Impact levels for performance assessment
#[derive(Debug, Clone, PartialEq)]
pub enum ImpactLevel {
    /// No significant impact
    None,
    /// Minor impact
    Minor,
    /// Moderate impact
    Moderate,
    /// Major impact
    Major,
    /// Severe impact
    Severe,
}

/// Enhanced error handling system
pub struct ErrorRecoverySystem {
    /// Error history
    error_history: Vec<EnhancedStatsError>,
    /// Recovery success rates
    #[allow(dead_code)]
    recovery_success_rates: HashMap<String, f64>,
    /// System configuration
    config: ErrorRecoveryConfig,
}

/// Configuration for error recovery system
#[derive(Debug, Clone)]
pub struct ErrorRecoveryConfig {
    /// Maximum number of errors to keep in history
    pub max_historysize: usize,
    /// Enable detailed diagnostics
    pub detailed_diagnostics: bool,
    /// Enable automatic recovery suggestions
    pub auto_suggestions: bool,
    /// Include performance analysis
    pub performance_analysis: bool,
    /// Include code examples
    pub include_examples: bool,
}

impl Default for ErrorRecoveryConfig {
    fn default() -> Self {
        Self {
            max_historysize: 100,
            detailed_diagnostics: true,
            auto_suggestions: true,
            performance_analysis: true,
            include_examples: true,
        }
    }
}

impl ErrorRecoverySystem {
    /// Create new error recovery system
    pub fn new(config: ErrorRecoveryConfig) -> Self {
        Self {
            error_history: Vec::new(),
            recovery_success_rates: HashMap::new(),
            config,
        }
    }

    /// Enhance a basic error with context and recovery suggestions
    pub fn enhance_error(
        &mut self,
        error: StatsError,
        function_name: &str,
        module_path: &str,
    ) -> EnhancedStatsError {
        let context = self.build_error_context(function_name, module_path, &error);
        let recovery_suggestions = self.generate_recovery_suggestions(&error, &context);
        let documentation_links = self.generate_documentation_links(&error);
        let example_snippets = if self.config.include_examples {
            self.generate_example_snippets(&error, &recovery_suggestions)
        } else {
            Vec::new()
        };
        let severity = self.assess_error_severity(&error, &context);
        let performance_impact = if self.config.performance_analysis {
            self.assess_performance_impact(&error, &context)
        } else {
            PerformanceImpact {
                memory_impact: ImpactLevel::None,
                time_impact: ImpactLevel::None,
                accuracy_impact: ImpactLevel::None,
                scalability_impact: ImpactLevel::None,
            }
        };

        let enhanced_error = EnhancedStatsError {
            error,
            context,
            recovery_suggestions,
            documentation_links,
            example_snippets,
            severity,
            performance_impact,
        };

        // Add to history
        self.error_history.push(enhanced_error.clone());
        if self.error_history.len() > self.config.max_historysize {
            self.error_history.drain(0..1);
        }

        enhanced_error
    }

    /// Build error context
    fn build_error_context(
        &self,
        function_name: &str,
        module_path: &str,
        error: &StatsError,
    ) -> ErrorContext {
        let data_characteristics = self.inferdata_characteristics(error);
        let system_info = self.gather_system_info();
        let computation_state = self.infer_computation_state(error, function_name);

        ErrorContext {
            function_name: function_name.to_string(),
            module_path: module_path.to_string(),
            data_characteristics,
            system_info,
            computation_state,
        }
    }

    /// Infer data characteristics from error
    fn inferdata_characteristics(&self, error: &StatsError) -> DataCharacteristics {
        // This would analyze the _error to infer data properties
        DataCharacteristics {
            size_info: None,
            type_info: "unknown".to_string(),
            range_info: None,
            missingdata_info: None,
            distribution_info: None,
        }
    }

    /// Gather system information
    fn gather_system_info(&self) -> SystemInfo {
        SystemInfo {
            available_memory_mb: self.get_available_memory(),
            cpu_cores: Some(num_cpus::get()),
            simd_capabilities: self.detect_simd_capabilities(),
            parallel_available: true,
        }
    }

    /// Get available memory (simplified)
    fn get_available_memory(&self) -> Option<f64> {
        // This would use system APIs to get actual memory info
        None
    }

    /// Detect SIMD capabilities
    fn detect_simd_capabilities(&self) -> Vec<String> {
        let mut capabilities = Vec::new();

        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("sse") {
                capabilities.push("SSE".to_string());
            }
            if std::arch::is_x86_feature_detected!("sse2") {
                capabilities.push("SSE2".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx") {
                capabilities.push("AVX".to_string());
            }
            if std::arch::is_x86_feature_detected!("avx2") {
                capabilities.push("AVX2".to_string());
            }
        }

        capabilities
    }

    /// Infer computation state
    fn infer_computation_state(&self, error: &StatsError, functionname: &str) -> ComputationState {
        ComputationState {
            algorithm: functionname.to_string(),
            iteration: None,
            convergence_status: None,
            current_tolerance: None,
            intermediate_results: HashMap::new(),
        }
    }

    /// Generate recovery suggestions based on error type
    fn generate_recovery_suggestions(
        &self,
        error: &StatsError,
        context: &ErrorContext,
    ) -> Vec<RecoverySuggestion> {
        let mut suggestions = Vec::new();

        match error {
            StatsError::InvalidArgument(msg) => {
                suggestions.extend(self.generate_invalid_argument_suggestions(msg, context));
            }
            StatsError::DimensionMismatch(msg) => {
                suggestions.extend(self.generate_dimension_mismatch_suggestions(msg, context));
            }
            StatsError::ComputationError(msg) => {
                suggestions.extend(self.generate_computation_error_suggestions(msg, context));
            }
            StatsError::ConvergenceError(msg) => {
                suggestions.extend(self.generate_convergence_error_suggestions(msg, context));
            }
            _ => {
                suggestions.push(RecoverySuggestion {
                    suggestion_type: SuggestionType::InputValidation,
                    description: "Check input data and parameters".to_string(),
                    action: RecoveryAction::ValidateInputs {
                        validation_checks: vec![ValidationCheck {
                            name: "Data finite check".to_string(),
                            condition: "All values are finite (not NaN or infinite)".to_string(),
                            fix_suggestion: "Remove or replace NaN/infinite values".to_string(),
                        }],
                    },
                    expected_outcome: "Eliminate invalid input data".to_string(),
                    confidence: 0.7,
                    prerequisites: vec![],
                });
            }
        }

        suggestions
    }

    /// Generate suggestions for invalid argument errors
    fn generate_invalid_argument_suggestions(
        &self,
        msg: &str,
        _context: &ErrorContext,
    ) -> Vec<RecoverySuggestion> {
        vec![RecoverySuggestion {
            suggestion_type: SuggestionType::InputValidation,
            description: "Validate input parameters before calling the function".to_string(),
            action: RecoveryAction::ValidateInputs {
                validation_checks: vec![ValidationCheck {
                    name: "Parameter bounds check".to_string(),
                    condition: "Parameters are within valid ranges".to_string(),
                    fix_suggestion: "Adjust parameters to valid ranges".to_string(),
                }],
            },
            expected_outcome: "Function executes successfully with valid inputs".to_string(),
            confidence: 0.9,
            prerequisites: vec!["Input data available".to_string()],
        }]
    }

    /// Generate suggestions for dimension mismatch errors
    fn generate_dimension_mismatch_suggestions(
        &self,
        msg: &str,
        _context: &ErrorContext,
    ) -> Vec<RecoverySuggestion> {
        vec![RecoverySuggestion {
            suggestion_type: SuggestionType::DataPreprocessing,
            description: "Reshape or transpose arrays to match expected dimensions".to_string(),
            action: RecoveryAction::PreprocessData {
                preprocessing_steps: vec![PreprocessingStep {
                    name: "Array reshape".to_string(),
                    description: "Reshape arrays to compatible dimensions".to_string(),
                    code_example: "array.intoshape((new_rows, new_cols))".to_string(),
                    expected_impact: "Arrays will have compatible dimensions".to_string(),
                }],
            },
            expected_outcome: "Arrays have compatible dimensions for the operation".to_string(),
            confidence: 0.85,
            prerequisites: vec!["Data can be reshaped without loss".to_string()],
        }]
    }

    /// Generate suggestions for computation errors
    fn generate_computation_error_suggestions(
        &self,
        msg: &str,
        context: &ErrorContext,
    ) -> Vec<RecoverySuggestion> {
        let mut suggestions = Vec::new();

        if msg.contains("singular") || msg.contains("invert") {
            suggestions.push(RecoverySuggestion {
                suggestion_type: SuggestionType::NumericalStability,
                description: "Add regularization to improve numerical stability".to_string(),
                action: RecoveryAction::AdjustParameter {
                    parameter_name: "regularization".to_string(),
                    current_value: "0.0".to_string(),
                    suggested_value: "1e-6".to_string(),
                    explanation: "Small regularization prevents singular matrices".to_string(),
                },
                expected_outcome: "Matrix inversion becomes numerically stable".to_string(),
                confidence: 0.8,
                prerequisites: vec!["Matrix inversion required".to_string()],
            });
        }

        if msg.contains("memory") || msg.contains("allocation") {
            suggestions.push(RecoverySuggestion {
                suggestion_type: SuggestionType::MemoryOptimization,
                description: "Use chunked processing to reduce memory usage".to_string(),
                action: RecoveryAction::ScaleComputation {
                    current_approach: "Process entire dataset at once".to_string(),
                    suggested_approach: "Process data in smaller chunks".to_string(),
                    expected_improvement: "Reduced memory usage".to_string(),
                },
                expected_outcome: "Computation succeeds with available memory".to_string(),
                confidence: 0.75,
                prerequisites: vec!["Data can be processed in chunks".to_string()],
            });
        }

        suggestions
    }

    /// Generate suggestions for convergence errors
    fn generate_convergence_error_suggestions(
        &self,
        msg: &str,
        _context: &ErrorContext,
    ) -> Vec<RecoverySuggestion> {
        vec![RecoverySuggestion {
            suggestion_type: SuggestionType::ParameterAdjustment,
            description: "Increase maximum iterations or relax tolerance".to_string(),
            action: RecoveryAction::AdjustParameter {
                parameter_name: "max_iterations".to_string(),
                current_value: "unknown".to_string(),
                suggested_value: "increased value".to_string(),
                explanation: "More iterations allow algorithm to converge".to_string(),
            },
            expected_outcome: "Algorithm converges within the iteration limit".to_string(),
            confidence: 0.7,
            prerequisites: vec!["Algorithm is potentially convergent".to_string()],
        }]
    }

    /// Generate documentation links
    fn generate_documentation_links(&self, error: &StatsError) -> Vec<String> {
        let mut links = Vec::new();

        match error {
            StatsError::InvalidArgument(_) => {
                links.push(
                    "https://docs.rs/scirs2-stats/latest/scirs2_stats/index.html#input-validation"
                        .to_string(),
                );
            }
            StatsError::DimensionMismatch(_) => {
                links.push(
                    "https://docs.rs/scirs2-stats/latest/scirs2_stats/index.html#array-operations"
                        .to_string(),
                );
            }
            StatsError::ComputationError(_) => {
                links.push("https://docs.rs/scirs2-stats/latest/scirs2_stats/index.html#numerical-stability".to_string());
            }
            _ => {
                links.push(
                    "https://docs.rs/scirs2-stats/latest/scirs2_stats/index.html".to_string(),
                );
            }
        }

        links
    }

    /// Generate example code snippets
    fn generate_example_snippets(
        &self,
        error: &StatsError,
        suggestions: &[RecoverySuggestion],
    ) -> Vec<CodeSnippet> {
        let mut snippets = Vec::new();

        if !suggestions.is_empty() {
            match &suggestions[0].action {
                RecoveryAction::AdjustParameter {
                    parameter_name,
                    suggested_value,
                    ..
                } => {
                    snippets.push(CodeSnippet {
                        title: format!("Adjust {} parameter", parameter_name),
                        code: format!(
                            "// Set {} to {}\nlet {} = {};\n// Then retry the operation",
                            parameter_name, suggested_value, parameter_name, suggested_value
                        ),
                        language: "rust".to_string(),
                        description: "Parameter adjustment example".to_string(),
                    });
                }
                RecoveryAction::PreprocessData {
                    preprocessing_steps,
                } => {
                    if !preprocessing_steps.is_empty() {
                        snippets.push(CodeSnippet {
                            title: "Data preprocessing".to_string(),
                            code: preprocessing_steps[0].code_example.clone(),
                            language: "rust".to_string(),
                            description: preprocessing_steps[0].description.clone(),
                        });
                    }
                }
                _ => {}
            }
        }

        snippets
    }

    /// Assess error severity
    fn assess_error_severity(&self, error: &StatsError, context: &ErrorContext) -> ErrorSeverity {
        match error {
            StatsError::InvalidArgument(_) => ErrorSeverity::Medium,
            StatsError::DimensionMismatch(_) => ErrorSeverity::Medium,
            StatsError::ComputationError(_) => ErrorSeverity::High,
            StatsError::ConvergenceError(_) => ErrorSeverity::Medium,
            _ => ErrorSeverity::Low,
        }
    }

    /// Assess performance impact
    fn assess_performance_impact(
        &self,
        error: &StatsError,
        context: &ErrorContext,
    ) -> PerformanceImpact {
        match error {
            StatsError::ComputationError(msg) if msg.contains("memory") => PerformanceImpact {
                memory_impact: ImpactLevel::Major,
                time_impact: ImpactLevel::Moderate,
                accuracy_impact: ImpactLevel::None,
                scalability_impact: ImpactLevel::Major,
            },
            StatsError::ConvergenceError(_) => PerformanceImpact {
                memory_impact: ImpactLevel::Minor,
                time_impact: ImpactLevel::Major,
                accuracy_impact: ImpactLevel::Moderate,
                scalability_impact: ImpactLevel::Moderate,
            },
            _ => PerformanceImpact {
                memory_impact: ImpactLevel::None,
                time_impact: ImpactLevel::Minor,
                accuracy_impact: ImpactLevel::Minor,
                scalability_impact: ImpactLevel::None,
            },
        }
    }

    /// Get error history
    pub fn error_history(&self) -> &[EnhancedStatsError] {
        &self.error_history
    }

    /// Generate comprehensive error report
    pub fn generate_error_report(&self, enhancederror: &EnhancedStatsError) -> String {
        let mut report = String::new();

        report.push_str("# Error Report\n\n");
        report.push_str(&format!("**Error:** {}\n\n", enhancederror.error));
        report.push_str(&format!("**Severity:** {:?}\n\n", enhancederror.severity));
        report.push_str(&format!(
            "**Function:** {}\n",
            enhancederror.context.function_name
        ));
        report.push_str(&format!(
            "**Module:** {}\n\n",
            enhancederror.context.module_path
        ));

        report.push_str("## Recovery Suggestions\n\n");
        for (i, suggestion) in enhancederror.recovery_suggestions.iter().enumerate() {
            report.push_str(&format!(
                "{}. **{}** (Confidence: {:.0}%)\n",
                i + 1,
                suggestion.description,
                suggestion.confidence * 100.0
            ));
            report.push_str(&format!("   - {}\n", suggestion.expected_outcome));
        }

        if !enhancederror.example_snippets.is_empty() {
            report.push_str("\n## Example Code\n\n");
            for snippet in &enhancederror.example_snippets {
                report.push_str(&format!("### {}\n\n", snippet.title));
                report.push_str(&format!(
                    "```{}\n{}\n```\n\n",
                    snippet.language, snippet.code
                ));
            }
        }

        if !enhancederror.documentation_links.is_empty() {
            report.push_str("## Documentation\n\n");
            for link in &enhancederror.documentation_links {
                report.push_str(&format!("- [Documentation]({})\n", link));
            }
        }

        report
    }
}

impl fmt::Display for EnhancedStatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;

        if !self.recovery_suggestions.is_empty() {
            write!(f, "\n\nSuggestions:")?;
            for suggestion in &self.recovery_suggestions {
                write!(f, "\n  - {}", suggestion.description)?;
            }
        }

        Ok(())
    }
}

/// Global error recovery system instance
static mut GLOBAL_ERROR_RECOVERY: Option<ErrorRecoverySystem> = None;
static mut ERROR_RECOVERY_INITIALIZED: bool = false;

/// Initialize global error recovery system
#[allow(dead_code)]
pub fn initialize_error_recovery(config: Option<ErrorRecoveryConfig>) {
    unsafe {
        if !ERROR_RECOVERY_INITIALIZED {
            GLOBAL_ERROR_RECOVERY = Some(ErrorRecoverySystem::new(config.unwrap_or_default()));
            ERROR_RECOVERY_INITIALIZED = true;
        }
    }
}

/// Enhance error with recovery suggestions (global function)
#[allow(dead_code)]
pub fn enhance_error_with_recovery(
    error: StatsError,
    function_name: &str,
    module_path: &str,
) -> EnhancedStatsError {
    unsafe {
        if !ERROR_RECOVERY_INITIALIZED {
            initialize_error_recovery(None);
        }

        if let Some(ref mut system) = GLOBAL_ERROR_RECOVERY {
            system.enhance_error(error, function_name, module_path)
        } else {
            // Fallback if system not available
            EnhancedStatsError {
                error,
                context: ErrorContext {
                    function_name: function_name.to_string(),
                    module_path: module_path.to_string(),
                    data_characteristics: DataCharacteristics {
                        size_info: None,
                        type_info: "unknown".to_string(),
                        range_info: None,
                        missingdata_info: None,
                        distribution_info: None,
                    },
                    system_info: SystemInfo {
                        available_memory_mb: None,
                        cpu_cores: Some(num_cpus::get()),
                        simd_capabilities: vec![],
                        parallel_available: true,
                    },
                    computation_state: ComputationState {
                        algorithm: function_name.to_string(),
                        iteration: None,
                        convergence_status: None,
                        current_tolerance: None,
                        intermediate_results: HashMap::new(),
                    },
                },
                recovery_suggestions: vec![],
                documentation_links: vec![],
                example_snippets: vec![],
                severity: ErrorSeverity::Medium,
                performance_impact: PerformanceImpact {
                    memory_impact: ImpactLevel::None,
                    time_impact: ImpactLevel::None,
                    accuracy_impact: ImpactLevel::None,
                    scalability_impact: ImpactLevel::None,
                },
            }
        }
    }
}

/// Convenience macro for enhanced error handling
#[macro_export]
macro_rules! enhanced_error {
    ($error:expr) => {
        enhance_error_with_recovery($error, function_name!(), module_path!())
    };
}
