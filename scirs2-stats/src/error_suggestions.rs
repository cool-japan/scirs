//! Enhanced error suggestion system with context-aware recovery strategies
//!
//! This module provides an intelligent error suggestion system that analyzes
//! error patterns and provides detailed, actionable recovery suggestions.

use crate::error::StatsError;
use std::collections::HashMap;

/// Error suggestion engine that provides intelligent recovery suggestions
pub struct SuggestionEngine {
    /// Common error patterns and their solutions
    patterns: HashMap<String, Vec<Suggestion>>,
    /// Context-specific suggestions
    context_suggestions: HashMap<String, Vec<Suggestion>>,
}

/// A recovery suggestion with priority and detailed steps
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// Brief description of the suggestion
    pub title: String,
    /// Detailed steps to implement the suggestion
    pub steps: Vec<String>,
    /// Priority level (1-5, where 1 is highest)
    pub priority: u8,
    /// Example code if applicable
    pub example: Option<String>,
    /// Links to relevant documentation
    pub docs: Vec<String>,
}

impl Default for SuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl SuggestionEngine {
    /// Create a new suggestion engine with built-in patterns
    pub fn new() -> Self {
        let mut engine = Self {
            patterns: HashMap::new(),
            context_suggestions: HashMap::new(),
        };

        engine.initialize_patterns();
        engine
    }

    /// Initialize common error patterns and solutions
    fn initialize_patterns(&mut self) {
        // NaN value patterns
        self.patterns.insert(
            "nan".to_string(),
            vec![
                Suggestion {
                    title: "Remove NaN values".to_string(),
                    steps: vec![
                        "Filter out NaN values using is_nan() check".to_string(),
                        "Use array.iter().filter(|x| !x.is_nan())".to_string(),
                        "Consider using ndarray's mapv() for element-wise operations".to_string(),
                    ],
                    priority: 1,
                    example: Some(
                        r#"
// Remove NaN values
let cleandata: Vec<f64> = data.iter()
    .filter(|&&x| !x.is_nan())
    .copied()
    .collect();
                    "#
                        .to_string(),
                    ),
                    docs: vec!["data_cleaning".to_string()],
                },
                Suggestion {
                    title: "Impute missing values".to_string(),
                    steps: vec![
                        "Calculate mean/median of non-NaN values".to_string(),
                        "Replace NaN with calculated statistic".to_string(),
                        "Consider forward/backward fill for time series".to_string(),
                    ],
                    priority: 2,
                    example: Some(
                        r#"
// Impute with mean
let mean = data.iter()
    .filter(|&&x| !x.is_nan())
    .sum::<f64>() / valid_count as f64;
let imputed = data.mapv(|x| if x.is_nan() { mean } else { x });
                    "#
                        .to_string(),
                    ),
                    docs: vec!["imputation_methods".to_string()],
                },
            ],
        );

        // Empty array patterns
        self.patterns.insert(
            "empty".to_string(),
            vec![Suggestion {
                title: "Check data loading process".to_string(),
                steps: vec![
                    "Verify file path and permissions".to_string(),
                    "Check if filters are too restrictive".to_string(),
                    "Add logging to data loading steps".to_string(),
                    "Validate data source is not empty".to_string(),
                ],
                priority: 1,
                example: Some(
                    r#"
// Add validation after loading
let data = loaddata(path)?;
if data.is_empty() {
    eprintln!("Warning: Loaded data is empty from {}", path);
    return Err(StatsError::invalid_argument("No data loaded"));
}
                    "#
                    .to_string(),
                ),
                docs: vec!["data_loading".to_string()],
            }],
        );

        // Dimension mismatch patterns
        self.patterns.insert(
            "dimension".to_string(),
            vec![
                Suggestion {
                    title: "Reshape arrays to match".to_string(),
                    steps: vec![
                        "Check shapes with .shape() or .dim()".to_string(),
                        "Use reshape() to adjust dimensions".to_string(),
                        "Ensure broadcasting rules are followed".to_string(),
                    ],
                    priority: 1,
                    example: Some(
                        r#"
// Check and match dimensions
println!("Array A shape: {:?}", a.shape());
println!("Array B shape: {:?}", b.shape());

// Reshape if needed
let b_reshaped = b.reshape((a.shape()[0], 1));
                    "#
                        .to_string(),
                    ),
                    docs: vec!["array_broadcasting".to_string()],
                },
                Suggestion {
                    title: "Transpose if needed".to_string(),
                    steps: vec![
                        "Check if arrays need transposition".to_string(),
                        "Use .t() or .transpose() methods".to_string(),
                    ],
                    priority: 2,
                    example: Some(
                        r#"
// Transpose for matrix multiplication
let result = a.dot(&b.t());
                    "#
                        .to_string(),
                    ),
                    docs: vec!["linear_algebra".to_string()],
                },
            ],
        );

        // Convergence failure patterns
        self.patterns.insert(
            "converge".to_string(),
            vec![
                Suggestion {
                    title: "Adjust algorithm parameters".to_string(),
                    steps: vec![
                        "Increase maximum iterations".to_string(),
                        "Relax convergence tolerance".to_string(),
                        "Try different learning rates".to_string(),
                    ],
                    priority: 1,
                    example: Some(
                        r#"
// Adjust parameters
let config = OptimizationConfig {
    max_iter: 10000,  // Increased from default
    tolerance: 1e-6,  // Relaxed from 1e-8
    learning_rate: 0.01,  // Reduced for stability
};
                    "#
                        .to_string(),
                    ),
                    docs: vec!["optimization_parameters".to_string()],
                },
                Suggestion {
                    title: "Preprocess data for better conditioning".to_string(),
                    steps: vec![
                        "Standardize features to zero mean, unit variance".to_string(),
                        "Remove highly correlated features".to_string(),
                        "Apply regularization techniques".to_string(),
                    ],
                    priority: 2,
                    example: Some(
                        r#"
// Standardize data
let mean = data.mean().unwrap();
let std = data.std(1);
let standardized = (data - mean) / std;
                    "#
                        .to_string(),
                    ),
                    docs: vec!["data_preprocessing".to_string()],
                },
            ],
        );

        // Singular matrix patterns
        self.patterns.insert(
            "singular".to_string(),
            vec![
                Suggestion {
                    title: "Add regularization".to_string(),
                    steps: vec![
                        "Add small value to diagonal (ridge regularization)".to_string(),
                        "Use SVD for pseudo-inverse".to_string(),
                        "Consider dimensionality reduction".to_string(),
                    ],
                    priority: 1,
                    example: Some(
                        r#"
// Ridge regularization
let lambda = 1e-4;
let regularized = matrix + lambda * Array2::eye(matrix.nrows());
                    "#
                        .to_string(),
                    ),
                    docs: vec!["regularization".to_string()],
                },
                Suggestion {
                    title: "Check for linear dependencies".to_string(),
                    steps: vec![
                        "Calculate correlation matrix".to_string(),
                        "Remove highly correlated features (|r| > 0.95)".to_string(),
                        "Use PCA to identify redundant dimensions".to_string(),
                    ],
                    priority: 2,
                    example: Some(
                        r#"
// Check correlations
let corr_matrix = corrcoef(&data.t(), "pearson")?;
for i in 0..n_features {
    for j in i+1..n_features {
        if corr_matrix[(i,j)].abs() > 0.95 {
            println!("Features {} and {} are highly correlated", i, j);
        }
    }
}
                    "#
                        .to_string(),
                    ),
                    docs: vec!["multicollinearity".to_string()],
                },
            ],
        );

        // Overflow patterns
        self.patterns.insert(
            "overflow".to_string(),
            vec![
                Suggestion {
                    title: "Scale input data".to_string(),
                    steps: vec![
                        "Normalize to [0, 1] or [-1, 1] range".to_string(),
                        "Use log transformation for large values".to_string(),
                        "Apply feature scaling techniques".to_string(),
                    ],
                    priority: 1,
                    example: Some(
                        r#"
// Min-max scaling
let min = data.min().unwrap();
let max = data.max().unwrap();
let scaled = (data - min) / (max - min);

// Log transformation
let log_transformed = data.mapv(|x| x.ln());
                    "#
                        .to_string(),
                    ),
                    docs: vec!["feature_scaling".to_string()],
                },
                Suggestion {
                    title: "Use numerically stable algorithms".to_string(),
                    steps: vec![
                        "Use log-sum-exp trick for exponentials".to_string(),
                        "Prefer stable implementations (e.g., log1p)".to_string(),
                        "Work in log space when possible".to_string(),
                    ],
                    priority: 2,
                    example: Some(
                        r#"
// Log-sum-exp trick
#[allow(dead_code)]
fn log_sum_exp(values: &[f64]) -> f64 {
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum = values.iter().map(|&x| (x - max_val).exp()).sum::<f64>();
    max_val + sum.ln()
}
                    "#
                        .to_string(),
                    ),
                    docs: vec!["numerical_stability".to_string()],
                },
            ],
        );
    }

    /// Get suggestions for a specific error
    pub fn get_suggestions(&self, error: &StatsError) -> Vec<Suggestion> {
        let error_str = error.to_string().to_lowercase();
        let mut suggestions = Vec::new();

        // Check each pattern
        for (pattern, pattern_suggestions) in &self.patterns {
            if error_str.contains(pattern) {
                suggestions.extend_from_slice(pattern_suggestions);
            }
        }

        // Sort by priority
        suggestions.sort_by_key(|s| s.priority);
        suggestions
    }

    /// Add context-specific suggestions
    pub fn add_context_suggestions(&mut self, context: String, suggestions: Vec<Suggestion>) {
        self.context_suggestions.insert(context, suggestions);
    }

    /// Get suggestions for a specific context
    pub fn get_context_suggestions(&self, context: &str) -> Option<&Vec<Suggestion>> {
        self.context_suggestions.get(context)
    }
}

/// Enhanced error formatter with suggestions
pub struct ErrorFormatter {
    suggestion_engine: SuggestionEngine,
}

impl Default for ErrorFormatter {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorFormatter {
    /// Create a new error formatter
    pub fn new() -> Self {
        Self {
            suggestion_engine: SuggestionEngine::new(),
        }
    }

    /// Format an error with detailed suggestions
    pub fn format_error(&self, error: StatsError, context: Option<&str>) -> String {
        let mut output = format!("Error: {}\n", error);

        // Get automatic suggestions
        let mut suggestions = self.suggestion_engine.get_suggestions(&error);

        // Add context-specific suggestions if available
        if let Some(ctx) = context {
            if let Some(ctx_suggestions) = self.suggestion_engine.get_context_suggestions(ctx) {
                suggestions.extend_from_slice(ctx_suggestions);
            }
        }

        if !suggestions.is_empty() {
            output.push_str("\n📋 Suggested Solutions:\n");

            for (i, suggestion) in suggestions.iter().enumerate() {
                output.push_str(&format!(
                    "\n{}. {} (Priority: {})\n",
                    i + 1,
                    suggestion.title,
                    suggestion.priority
                ));

                output.push_str("   Steps:\n");
                for step in &suggestion.steps {
                    output.push_str(&format!("   • {}\n", step));
                }

                if let Some(example) = &suggestion.example {
                    output.push_str("\n   Example:\n");
                    for line in example.lines() {
                        output.push_str(&format!("   {}\n", line));
                    }
                }

                if !suggestion.docs.is_empty() {
                    output.push_str("\n   See also: ");
                    output.push_str(&suggestion.docs.join(", "));
                    output.push('\n');
                }
            }
        }

        output
    }
}

/// Quick error diagnosis tool
#[allow(dead_code)]
pub fn diagnose_error(error: &StatsError) -> DiagnosisReport {
    let error_str = error.to_string().to_lowercase();

    let error_type = if error_str.contains("dimension") {
        ErrorType::DimensionMismatch
    } else if error_str.contains("empty") {
        ErrorType::EmptyData
    } else if error_str.contains("nan") {
        ErrorType::InvalidValues
    } else if error_str.contains("converge") {
        ErrorType::ConvergenceFailure
    } else if error_str.contains("singular") {
        ErrorType::SingularMatrix
    } else if error_str.contains("overflow") {
        ErrorType::NumericalOverflow
    } else if error_str.contains("domain") {
        ErrorType::DomainError
    } else {
        ErrorType::Other
    };

    let severity = match error_type {
        ErrorType::NumericalOverflow | ErrorType::SingularMatrix => Severity::High,
        ErrorType::ConvergenceFailure | ErrorType::InvalidValues => Severity::Medium,
        _ => Severity::Low,
    };

    let likely_causes = match error_type {
        ErrorType::DimensionMismatch => vec![
            "Arrays have incompatible shapes".to_string(),
            "Missing transpose operation".to_string(),
            "Incorrect axis specification".to_string(),
        ],
        ErrorType::EmptyData => vec![
            "Data loading failed".to_string(),
            "Filters removed all data".to_string(),
            "Incorrect file path".to_string(),
        ],
        ErrorType::InvalidValues => vec![
            "Missing data not handled".to_string(),
            "Division by zero".to_string(),
            "Invalid mathematical operation".to_string(),
        ],
        ErrorType::ConvergenceFailure => vec![
            "Poor initial values".to_string(),
            "Ill-conditioned problem".to_string(),
            "Insufficient iterations".to_string(),
        ],
        ErrorType::SingularMatrix => vec![
            "Linear dependencies in data".to_string(),
            "Insufficient observations".to_string(),
            "Perfect multicollinearity".to_string(),
        ],
        ErrorType::NumericalOverflow => vec![
            "Values too large".to_string(),
            "Exponential growth".to_string(),
            "Insufficient precision".to_string(),
        ],
        ErrorType::DomainError => vec![
            "Invalid parameter values".to_string(),
            "Out of bounds input".to_string(),
            "Constraint violation".to_string(),
        ],
        ErrorType::Other => vec!["Unknown cause".to_string()],
    };

    DiagnosisReport {
        error_type,
        severity,
        likely_causes,
    }
}

/// Error diagnosis report
#[derive(Debug)]
pub struct DiagnosisReport {
    pub error_type: ErrorType,
    pub severity: Severity,
    pub likely_causes: Vec<String>,
}

/// Common error types
#[derive(Debug, PartialEq)]
pub enum ErrorType {
    DimensionMismatch,
    EmptyData,
    InvalidValues,
    ConvergenceFailure,
    SingularMatrix,
    NumericalOverflow,
    DomainError,
    Other,
}

/// Error severity levels
#[derive(Debug, PartialEq, PartialOrd)]
pub enum Severity {
    Low,
    Medium,
    High,
}

/// Helper macro for creating errors with suggestions
#[macro_export]
macro_rules! stats_error_with_suggestions {
    ($error_type:ident, $msg:expr, $($suggestion:expr),+) => {
        {
            let error = StatsError::$error_type($msg);
            let formatter = ErrorFormatter::new();
            let mut engine = SuggestionEngine::new();

            let suggestions = vec![
                $(
                    Suggestion {
                        title: $suggestion.to_string(),
                        steps: vec![],
                        priority: 1,
                        example: None,
                        docs: vec![],
                    },
                )+
            ];

            engine.add_context_suggestions("custom".to_string(), suggestions);
            eprintln!("{}", formatter.format_error(error, Some("custom")));
            error
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggestion_engine() {
        let engine = SuggestionEngine::new();

        // Test NaN error suggestions
        let nan_error = StatsError::invalid_argument("Found NaN values");
        let suggestions = engine.get_suggestions(&nan_error);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].priority, 1);
    }

    #[test]
    fn test_error_diagnosis() {
        let dim_error = StatsError::dimension_mismatch("Arrays must have same length");
        let diagnosis = diagnose_error(&dim_error);
        assert_eq!(diagnosis.error_type, ErrorType::DimensionMismatch);
        assert_eq!(diagnosis.severity, Severity::Low);
    }

    #[test]
    fn test_error_formatter() {
        let formatter = ErrorFormatter::new();
        let error = StatsError::invalid_argument("Array contains NaN values");
        let formatted = formatter.format_error(error, None);

        assert!(formatted.contains("Suggested Solutions"));
        assert!(formatted.contains("Remove NaN values"));
    }
}
