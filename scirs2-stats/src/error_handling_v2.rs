//! Enhanced error handling system for v1.0.0
//!
//! This module provides a unified error handling system with:
//! - Structured error codes
//! - Automatic suggestion generation
//! - Performance impact warnings
//! - Recovery strategies

use crate::error::StatsError;
use std::fmt;

/// Error codes for categorization and tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // Domain errors (1xxx)
    E1001, // Value out of domain
    E1002, // Negative value where positive required
    E1003, // Probability out of range
    E1004, // Invalid degrees of freedom

    // Dimension errors (2xxx)
    E2001, // Array dimension mismatch
    E2002, // Matrix not square
    E2003, // Insufficient data points
    E2004, // Empty input

    // Computation errors (3xxx)
    E3001, // Numerical overflow
    E3002, // Numerical underflow
    E3003, // Convergence failure
    E3004, // Singular matrix
    E3005, // NaN encountered
    E3006, // Infinity encountered

    // Algorithm errors (4xxx)
    E4001, // Maximum iterations exceeded
    E4002, // Tolerance not achieved
    E4003, // Invalid algorithm parameter

    // Memory errors (5xxx)
    E5001, // Allocation failure
    E5002, // Memory limit exceeded
}

impl ErrorCode {
    /// Get a human-readable description of the error code
    pub fn description(&self) -> &'static str {
        match self {
            ErrorCode::E1001 => "Value is outside the valid domain",
            ErrorCode::E1002 => "Negative value provided where positive required",
            ErrorCode::E1003 => "Probability value must be between 0 and 1",
            ErrorCode::E1004 => "Invalid degrees of freedom",

            ErrorCode::E2001 => "Array dimensions do not match",
            ErrorCode::E2002 => "Matrix must be square",
            ErrorCode::E2003 => "Insufficient data points for operation",
            ErrorCode::E2004 => "Empty input provided",

            ErrorCode::E3001 => "Numerical overflow occurred",
            ErrorCode::E3002 => "Numerical underflow occurred",
            ErrorCode::E3003 => "Algorithm failed to converge",
            ErrorCode::E3004 => "Matrix is singular or near-singular",
            ErrorCode::E3005 => "NaN (Not a Number) encountered",
            ErrorCode::E3006 => "Infinity encountered",

            ErrorCode::E4001 => "Maximum iterations exceeded",
            ErrorCode::E4002 => "Required tolerance not achieved",
            ErrorCode::E4003 => "Invalid algorithm parameter",

            ErrorCode::E5001 => "Memory allocation failed",
            ErrorCode::E5002 => "Memory limit exceeded",
        }
    }

    /// Get the severity level (1-5, where 1 is most severe)
    pub fn severity(&self) -> u8 {
        match self {
            ErrorCode::E3001 | ErrorCode::E3002 | ErrorCode::E5001 | ErrorCode::E5002 => 1,
            ErrorCode::E3003 | ErrorCode::E3004 => 2,
            ErrorCode::E1001 | ErrorCode::E1002 | ErrorCode::E1003 | ErrorCode::E1004 => 3,
            ErrorCode::E2001 | ErrorCode::E2002 | ErrorCode::E2003 | ErrorCode::E2004 => 3,
            ErrorCode::E3005 | ErrorCode::E3006 => 3,
            ErrorCode::E4001 | ErrorCode::E4002 | ErrorCode::E4003 => 4,
        }
    }
}

/// Enhanced error with code, context, and suggestions
#[derive(Debug)]
pub struct EnhancedError {
    /// The error code
    pub code: ErrorCode,
    /// The underlying stats error
    pub error: StatsError,
    /// Additional context
    pub context: ErrorContext,
    /// Recovery suggestions
    pub suggestions: Vec<RecoverySuggestion>,
    /// Performance impact if error is ignored
    pub performance_impact: Option<PerformanceImpact>,
}

/// Context information for errors
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// The operation being performed
    pub operation: String,
    /// Input parameters that caused the error
    pub parameters: Vec<(String, String)>,
    /// Call stack depth
    pub call_depth: usize,
    /// Time when error occurred
    pub timestamp: std::time::SystemTime,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            parameters: Vec::new(),
            call_depth: 0,
            timestamp: std::time::SystemTime::now(),
        }
    }

    pub fn with_parameter(mut self, name: impl Into<String>, value: impl fmt::Display) -> Self {
        self.parameters.push((name.into(), value.to_string()));
        self
    }
}

/// Recovery suggestion with actionable steps
#[derive(Debug, Clone)]
pub struct RecoverySuggestion {
    /// Brief title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Code example
    pub example: Option<String>,
    /// Estimated complexity (1-5)
    pub complexity: u8,
    /// Whether this fixes the root cause
    pub fixes_root_cause: bool,
}

/// Performance impact information
#[derive(Debug, Clone)]
pub struct PerformanceImpact {
    /// Expected slowdown factor
    pub slowdown_factor: f64,
    /// Memory overhead in bytes
    pub memory_overhead: Option<usize>,
    /// Description of impact
    pub description: String,
}

/// Builder for creating enhanced errors
pub struct ErrorBuilder {
    code: ErrorCode,
    context: ErrorContext,
    suggestions: Vec<RecoverySuggestion>,
    performance_impact: Option<PerformanceImpact>,
}

impl ErrorBuilder {
    pub fn new(code: ErrorCode, operation: impl Into<String>) -> Self {
        Self {
            code,
            context: ErrorContext::new(operation),
            suggestions: Vec::new(),
            performance_impact: None,
        }
    }

    pub fn parameter(mut self, name: impl Into<String>, value: impl fmt::Display) -> Self {
        self.context = self.context.with_parameter(name, value);
        self
    }

    pub fn suggestion(mut self, suggestion: RecoverySuggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    pub fn performance_impact(mut self, impact: PerformanceImpact) -> Self {
        self.performance_impact = Some(impact);
        self
    }

    pub fn build(self, error: StatsError) -> EnhancedError {
        let mut enhanced = EnhancedError {
            code: self.code,
            error,
            context: self.context,
            suggestions: self.suggestions,
            performance_impact: self.performance_impact,
        };

        // Add automatic suggestions based on error code
        enhanced.add_automatic_suggestions();
        enhanced
    }
}

impl EnhancedError {
    /// Add automatic suggestions based on the error code
    fn add_automatic_suggestions(&mut self) {
        match self.code {
            ErrorCode::E3005 => {
                if self.suggestions.is_empty() {
                    self.suggestions.push(RecoverySuggestion {
                        title: "Handle NaN values".to_string(),
                        description: "Filter out or replace NaN values before computation"
                            .to_string(),
                        example: Some("data.iter().filter(|x| !x.is_nan())".to_string()),
                        complexity: 2,
                        fixes_root_cause: true,
                    });
                }
            }
            ErrorCode::E2004 => {
                if self.suggestions.is_empty() {
                    self.suggestions.push(RecoverySuggestion {
                        title: "Check input data".to_string(),
                        description: "Ensure data is loaded correctly and not filtered out"
                            .to_string(),
                        example: Some(
                            "assert!(!data.is_empty(), \"Data cannot be empty\");".to_string(),
                        ),
                        complexity: 1,
                        fixes_root_cause: true,
                    });
                }
            }
            ErrorCode::E3003 => {
                if self.suggestions.is_empty() {
                    self.suggestions.push(RecoverySuggestion {
                        title: "Adjust convergence parameters".to_string(),
                        description: "Increase max iterations or relax tolerance".to_string(),
                        example: Some("options.max_iter(1000).tolerance(1e-6)".to_string()),
                        complexity: 2,
                        fixes_root_cause: false,
                    });
                }
            }
            _ => {}
        }
    }

    /// Format the error as a detailed report
    pub fn detailed_report(&self) -> String {
        let mut report = format!("Error {}: {}\n\n", self.code, self.code.description());

        report.push_str(&format!("Operation: {}\n", self.context.operation));

        if !self.context.parameters.is_empty() {
            report.push_str("Parameters:\n");
            for (name, value) in &self.context.parameters {
                report.push_str(&format!("  - {}: {}\n", name, value));
            }
            report.push('\n');
        }

        report.push_str(&format!("Details: {}\n\n", self.error));

        if !self.suggestions.is_empty() {
            report.push_str("Recovery Suggestions:\n");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                report.push_str(&format!(
                    "{}. {} (complexity: {}/5)\n   {}\n",
                    i + 1,
                    suggestion.title,
                    suggestion.complexity,
                    suggestion.description
                ));
                if let Some(example) = &suggestion.example {
                    report.push_str(&format!("   Example: {}\n", example));
                }
                report.push('\n');
            }
        }

        if let Some(impact) = &self.performance_impact {
            report.push_str(&format!(
                "Performance Impact if ignored:\n  - Slowdown: {}x\n  - {}\n",
                impact.slowdown_factor, impact.description
            ));
        }

        report
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Helper macros for creating standardized errors
#[macro_export]
macro_rules! stats_error {
    ($code:expr, $op:expr, $msg:expr) => {
        ErrorBuilder::new($code, $op)
            .build(StatsError::computation($msg))
    };

    ($code:expr, $op:expr, $msg:expr, $($param:expr => $value:expr),+) => {
        ErrorBuilder::new($code, $op)
            $(.parameter($param, $value))+
            .build(StatsError::computation($msg))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_builder() {
        let error = ErrorBuilder::new(ErrorCode::E3005, "mean calculation")
            .parameter("array_length", 100)
            .parameter("nan_count", 5)
            .suggestion(RecoverySuggestion {
                title: "Remove NaN values".to_string(),
                description: "Filter array before calculation".to_string(),
                example: None,
                complexity: 2,
                fixes_root_cause: true,
            })
            .build(StatsError::computation("NaN values in input"));

        assert_eq!(error.code, ErrorCode::E3005);
        assert_eq!(error.context.operation, "mean calculation");
        assert_eq!(error.context.parameters.len(), 2);
        assert!(!error.suggestions.is_empty());
    }

    #[test]
    fn test_error_code_severity() {
        assert_eq!(ErrorCode::E3001.severity(), 1); // Overflow is severe
        assert_eq!(ErrorCode::E1001.severity(), 3); // Domain error is moderate
        assert_eq!(ErrorCode::E4001.severity(), 4); // Max iterations is less severe
    }
}
