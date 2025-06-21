//! Error mapping utilities for SciRS2 module integration
//!
//! This module provides consistent error handling and mapping between
//! different SciRS2 modules, allowing for unified error reporting
//! and recovery strategies across the ecosystem.

use super::IntegrationError;
use std::collections::HashMap;

/// Error mapping registry for cross-module error handling
pub struct ErrorMapper {
    /// Registered error mappings
    mappings: HashMap<String, Box<dyn ErrorMapping>>,
    /// Error context stack
    context_stack: Vec<ErrorContext>,
    /// Error recovery strategies
    recovery_strategies: HashMap<String, Box<dyn ErrorRecovery>>,
}

impl ErrorMapper {
    /// Create new error mapper
    pub fn new() -> Self {
        let mut mapper = Self {
            mappings: HashMap::new(),
            context_stack: Vec::new(),
            recovery_strategies: HashMap::new(),
        };

        // Register built-in mappings
        mapper.register_builtin_mappings();
        mapper.register_builtin_recovery_strategies();

        mapper
    }

    /// Register an error mapping
    pub fn register_mapping<M: ErrorMapping + 'static>(
        &mut self,
        source_module: String,
        mapping: M,
    ) {
        self.mappings.insert(source_module, Box::new(mapping));
    }

    /// Register error recovery strategy
    pub fn register_recovery<R: ErrorRecovery + 'static>(
        &mut self,
        error_type: String,
        recovery: R,
    ) {
        self.recovery_strategies
            .insert(error_type, Box::new(recovery));
    }

    /// Map error from source module to integration error
    pub fn map_error(
        &self,
        source_module: &str,
        source_error: &dyn std::error::Error,
    ) -> IntegrationError {
        if let Some(mapping) = self.mappings.get(source_module) {
            mapping.map_error(source_error)
        } else {
            IntegrationError::ModuleCompatibility(format!(
                "Unmapped error from {}: {}",
                source_module, source_error
            ))
        }
    }

    /// Push error context
    pub fn push_context(&mut self, context: ErrorContext) {
        self.context_stack.push(context);
    }

    /// Pop error context
    pub fn pop_context(&mut self) -> Option<ErrorContext> {
        self.context_stack.pop()
    }

    /// Get current error context
    pub fn current_context(&self) -> Option<&ErrorContext> {
        self.context_stack.last()
    }

    /// Attempt error recovery
    pub fn attempt_recovery(
        &self,
        error: &IntegrationError,
    ) -> Result<RecoveryAction, IntegrationError> {
        let error_type = self.classify_error(error);

        if let Some(recovery) = self.recovery_strategies.get(&error_type) {
            recovery.attempt_recovery(error)
        } else {
            Err(IntegrationError::ModuleCompatibility(format!(
                "No recovery strategy for error type: {}",
                error_type
            )))
        }
    }

    /// Create enriched error with context
    pub fn enrich_error(&self, error: IntegrationError) -> EnrichedError {
        let suggestions = self.generate_suggestions(&error);
        EnrichedError {
            original_error: error,
            context_stack: self.context_stack.clone(),
            module_trace: self.build_module_trace(),
            suggestions,
            related_errors: Vec::new(),
        }
    }

    /// Aggregate multiple errors into a single report
    pub fn aggregate_errors(&self, errors: Vec<IntegrationError>) -> AggregatedError {
        let mut by_category = HashMap::new();
        let mut by_module = HashMap::new();

        for (index, error) in errors.iter().enumerate() {
            let category = self.classify_error(error);
            by_category
                .entry(category.clone())
                .or_insert_with(Vec::new)
                .push(index);

            if let Some(module) = self.extract_module_from_error(error) {
                by_module.entry(module).or_insert_with(Vec::new).push(index);
            }
        }

        let summary = self.generate_error_summary_from_indices(&errors, &by_category);

        AggregatedError {
            errors,
            by_category,
            by_module,
            summary,
        }
    }

    /// Generate error report
    pub fn generate_report(&self, error: &IntegrationError) -> ErrorReport {
        let enriched = self.enrich_error(error.clone());

        ErrorReport {
            error_id: self.generate_error_id(&enriched),
            timestamp: std::time::SystemTime::now(),
            error_type: self.classify_error(error),
            severity: self.assess_severity(error),
            enriched_error: enriched,
            recovery_suggestions: self.generate_recovery_suggestions(error),
            related_documentation: self.find_related_documentation(error),
        }
    }

    // Helper methods
    fn register_builtin_mappings(&mut self) {
        // Register neural module error mapping
        self.mappings
            .insert("scirs2-neural".to_string(), Box::new(NeuralErrorMapping));

        // Register optimization module error mapping
        self.mappings
            .insert("scirs2-optim".to_string(), Box::new(OptimErrorMapping));

        // Register linear algebra module error mapping
        self.mappings
            .insert("scirs2-linalg".to_string(), Box::new(LinalgErrorMapping));

        // Register core module error mapping
        self.mappings
            .insert("scirs2-core".to_string(), Box::new(CoreErrorMapping));
    }

    fn register_builtin_recovery_strategies(&mut self) {
        // Register tensor conversion recovery
        self.recovery_strategies.insert(
            "tensor_conversion".to_string(),
            Box::new(TensorConversionRecovery),
        );

        // Register module compatibility recovery
        self.recovery_strategies.insert(
            "module_compatibility".to_string(),
            Box::new(CompatibilityRecovery),
        );

        // Register configuration recovery
        self.recovery_strategies
            .insert("configuration".to_string(), Box::new(ConfigurationRecovery));
    }

    fn classify_error(&self, error: &IntegrationError) -> String {
        match error {
            IntegrationError::TensorConversion(_) => "tensor_conversion".to_string(),
            IntegrationError::ModuleCompatibility(_) => "module_compatibility".to_string(),
            IntegrationError::ConfigMismatch(_) => "configuration".to_string(),
            IntegrationError::VersionIncompatibility(_) => "version_compatibility".to_string(),
            IntegrationError::ApiBoundary(_) => "api_boundary".to_string(),
        }
    }

    fn assess_severity(&self, error: &IntegrationError) -> ErrorSeverity {
        match error {
            IntegrationError::TensorConversion(_) => ErrorSeverity::Medium,
            IntegrationError::ModuleCompatibility(_) => ErrorSeverity::High,
            IntegrationError::ConfigMismatch(_) => ErrorSeverity::Low,
            IntegrationError::VersionIncompatibility(_) => ErrorSeverity::High,
            IntegrationError::ApiBoundary(_) => ErrorSeverity::Medium,
        }
    }

    fn extract_module_from_error(&self, error: &IntegrationError) -> Option<String> {
        // Extract module name from error message or context
        match error {
            IntegrationError::ModuleCompatibility(msg)
            | IntegrationError::TensorConversion(msg)
            | IntegrationError::ConfigMismatch(msg)
            | IntegrationError::VersionIncompatibility(msg)
            | IntegrationError::ApiBoundary(msg) => {
                // Simple extraction based on known module names
                if msg.contains("scirs2-neural") {
                    Some("scirs2-neural".to_string())
                } else if msg.contains("scirs2-optim") {
                    Some("scirs2-optim".to_string())
                } else if msg.contains("scirs2-linalg") {
                    Some("scirs2-linalg".to_string())
                } else {
                    None
                }
            }
        }
    }

    fn build_module_trace(&self) -> Vec<String> {
        self.context_stack
            .iter()
            .map(|ctx| ctx.module_name.clone())
            .collect()
    }

    fn generate_suggestions(&self, error: &IntegrationError) -> Vec<String> {
        match error {
            IntegrationError::TensorConversion(_) => vec![
                "Check tensor shapes and data types".to_string(),
                "Ensure compatible precision levels".to_string(),
                "Verify memory layout compatibility".to_string(),
            ],
            IntegrationError::ModuleCompatibility(_) => vec![
                "Check module versions".to_string(),
                "Verify required features are enabled".to_string(),
                "Update module dependencies".to_string(),
            ],
            IntegrationError::ConfigMismatch(_) => vec![
                "Check configuration file syntax".to_string(),
                "Verify environment variables".to_string(),
                "Reset to default configuration".to_string(),
            ],
            IntegrationError::VersionIncompatibility(_) => vec![
                "Update to compatible versions".to_string(),
                "Check version compatibility matrix".to_string(),
                "Use version pinning in dependencies".to_string(),
            ],
            IntegrationError::ApiBoundary(_) => vec![
                "Check API documentation".to_string(),
                "Verify function signatures".to_string(),
                "Update integration code".to_string(),
            ],
        }
    }

    #[allow(dead_code)]
    fn generate_error_summary(
        &self,
        by_category: &HashMap<String, Vec<&IntegrationError>>,
    ) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Found {} error categories:\n", by_category.len()));

        for (category, errors) in by_category {
            summary.push_str(&format!("  {}: {} errors\n", category, errors.len()));
        }

        summary
    }

    fn generate_error_summary_from_indices(
        &self,
        _errors: &[IntegrationError],
        by_category: &HashMap<String, Vec<usize>>,
    ) -> String {
        let mut summary = String::new();
        summary.push_str(&format!("Found {} error categories:\n", by_category.len()));

        for (category, error_indices) in by_category {
            summary.push_str(&format!("  {}: {} errors\n", category, error_indices.len()));
        }

        summary
    }

    fn generate_error_id(&self, _enriched: &EnrichedError) -> String {
        // Generate unique error ID
        format!(
            "ERR_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        )
    }

    fn generate_recovery_suggestions(&self, error: &IntegrationError) -> Vec<RecoverySuggestion> {
        let base_suggestions = self.generate_suggestions(error);

        base_suggestions
            .into_iter()
            .enumerate()
            .map(|(i, suggestion)| RecoverySuggestion {
                priority: if i == 0 {
                    Priority::High
                } else {
                    Priority::Medium
                },
                action: suggestion,
                estimated_success_rate: 0.7 - (i as f64 * 0.1),
                requires_restart: false,
            })
            .collect()
    }

    fn find_related_documentation(&self, error: &IntegrationError) -> Vec<DocumentationLink> {
        match error {
            IntegrationError::TensorConversion(_) => vec![DocumentationLink {
                title: "Tensor Conversion Guide".to_string(),
                url: "https://scirs2.dev/docs/tensor-conversion".to_string(),
                section: Some("Basic Conversion".to_string()),
            }],
            IntegrationError::ModuleCompatibility(_) => vec![DocumentationLink {
                title: "Module Compatibility Matrix".to_string(),
                url: "https://scirs2.dev/docs/compatibility".to_string(),
                section: None,
            }],
            _ => Vec::new(),
        }
    }
}

impl Default for ErrorMapper {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for error mapping between modules
pub trait ErrorMapping: Send {
    fn map_error(&self, source_error: &dyn std::error::Error) -> IntegrationError;
}

/// Trait for error recovery strategies
pub trait ErrorRecovery: Send {
    fn attempt_recovery(
        &self,
        error: &IntegrationError,
    ) -> Result<RecoveryAction, IntegrationError>;
}

/// Error context for tracking error origins
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub module_name: String,
    pub function_name: Option<String>,
    pub operation: Option<String>,
    pub additional_info: HashMap<String, String>,
}

impl ErrorContext {
    /// Create new error context
    pub fn new(module_name: String) -> Self {
        Self {
            module_name,
            function_name: None,
            operation: None,
            additional_info: HashMap::new(),
        }
    }

    /// Add function name
    pub fn with_function(mut self, function_name: String) -> Self {
        self.function_name = Some(function_name);
        self
    }

    /// Add operation name
    pub fn with_operation(mut self, operation: String) -> Self {
        self.operation = Some(operation);
        self
    }

    /// Add additional information
    pub fn with_info(mut self, key: String, value: String) -> Self {
        self.additional_info.insert(key, value);
        self
    }
}

/// Enriched error with additional context
#[derive(Debug, Clone)]
pub struct EnrichedError {
    pub original_error: IntegrationError,
    pub context_stack: Vec<ErrorContext>,
    pub module_trace: Vec<String>,
    pub suggestions: Vec<String>,
    pub related_errors: Vec<IntegrationError>,
}

/// Aggregated error report
#[derive(Debug)]
pub struct AggregatedError {
    pub errors: Vec<IntegrationError>,
    pub by_category: HashMap<String, Vec<usize>>,
    pub by_module: HashMap<String, Vec<usize>>,
    pub summary: String,
}

/// Comprehensive error report
#[derive(Debug)]
pub struct ErrorReport {
    pub error_id: String,
    pub timestamp: std::time::SystemTime,
    pub error_type: String,
    pub severity: ErrorSeverity,
    pub enriched_error: EnrichedError,
    pub recovery_suggestions: Vec<RecoverySuggestion>,
    pub related_documentation: Vec<DocumentationLink>,
}

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Recovery actions
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Retry,
    RetryWithConfig(HashMap<String, String>),
    Fallback(String),
    ManualIntervention(String),
    Abort,
}

/// Recovery suggestion
#[derive(Debug, Clone)]
pub struct RecoverySuggestion {
    pub priority: Priority,
    pub action: String,
    pub estimated_success_rate: f64,
    pub requires_restart: bool,
}

/// Priority levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

/// Documentation link
#[derive(Debug, Clone)]
pub struct DocumentationLink {
    pub title: String,
    pub url: String,
    pub section: Option<String>,
}

// Built-in error mappings

/// Neural module error mapping
struct NeuralErrorMapping;

impl ErrorMapping for NeuralErrorMapping {
    fn map_error(&self, source_error: &dyn std::error::Error) -> IntegrationError {
        let error_str = source_error.to_string().to_lowercase();

        if error_str.contains("tensor") || error_str.contains("shape") {
            IntegrationError::TensorConversion(format!(
                "Neural module tensor error: {}",
                source_error
            ))
        } else if error_str.contains("gradient") {
            IntegrationError::ApiBoundary(format!("Neural module gradient error: {}", source_error))
        } else {
            IntegrationError::ModuleCompatibility(format!("Neural module error: {}", source_error))
        }
    }
}

/// Optimization module error mapping
struct OptimErrorMapping;

impl ErrorMapping for OptimErrorMapping {
    fn map_error(&self, source_error: &dyn std::error::Error) -> IntegrationError {
        let error_str = source_error.to_string().to_lowercase();

        if error_str.contains("parameter") || error_str.contains("optimizer") {
            IntegrationError::ConfigMismatch(format!(
                "Optimizer configuration error: {}",
                source_error
            ))
        } else if error_str.contains("learning_rate") {
            IntegrationError::ConfigMismatch(format!("Learning rate error: {}", source_error))
        } else {
            IntegrationError::ModuleCompatibility(format!(
                "Optimization module error: {}",
                source_error
            ))
        }
    }
}

/// Linear algebra module error mapping
struct LinalgErrorMapping;

impl ErrorMapping for LinalgErrorMapping {
    fn map_error(&self, source_error: &dyn std::error::Error) -> IntegrationError {
        let error_str = source_error.to_string().to_lowercase();

        if error_str.contains("matrix") || error_str.contains("dimension") {
            IntegrationError::TensorConversion(format!("Matrix dimension error: {}", source_error))
        } else if error_str.contains("singular") || error_str.contains("decomposition") {
            IntegrationError::ApiBoundary(format!(
                "Linear algebra operation error: {}",
                source_error
            ))
        } else {
            IntegrationError::ModuleCompatibility(format!(
                "Linear algebra module error: {}",
                source_error
            ))
        }
    }
}

/// Core module error mapping
struct CoreErrorMapping;

impl ErrorMapping for CoreErrorMapping {
    fn map_error(&self, source_error: &dyn std::error::Error) -> IntegrationError {
        let error_str = source_error.to_string().to_lowercase();

        if error_str.contains("config") {
            IntegrationError::ConfigMismatch(format!("Core configuration error: {}", source_error))
        } else if error_str.contains("type") || error_str.contains("conversion") {
            IntegrationError::TensorConversion(format!(
                "Core type conversion error: {}",
                source_error
            ))
        } else {
            IntegrationError::ModuleCompatibility(format!("Core module error: {}", source_error))
        }
    }
}

// Built-in recovery strategies

/// Tensor conversion recovery strategy
struct TensorConversionRecovery;

impl ErrorRecovery for TensorConversionRecovery {
    fn attempt_recovery(
        &self,
        error: &IntegrationError,
    ) -> Result<RecoveryAction, IntegrationError> {
        match error {
            IntegrationError::TensorConversion(msg) => {
                if msg.contains("shape") {
                    Ok(RecoveryAction::RetryWithConfig(
                        [("auto_reshape".to_string(), "true".to_string())].into(),
                    ))
                } else if msg.contains("precision") {
                    Ok(RecoveryAction::RetryWithConfig(
                        [("auto_convert_precision".to_string(), "true".to_string())].into(),
                    ))
                } else {
                    Ok(RecoveryAction::Fallback(
                        "Use manual conversion".to_string(),
                    ))
                }
            }
            _ => Err(IntegrationError::ModuleCompatibility(
                "Cannot recover from non-tensor-conversion error".to_string(),
            )),
        }
    }
}

/// Module compatibility recovery strategy
struct CompatibilityRecovery;

impl ErrorRecovery for CompatibilityRecovery {
    fn attempt_recovery(
        &self,
        error: &IntegrationError,
    ) -> Result<RecoveryAction, IntegrationError> {
        match error {
            IntegrationError::ModuleCompatibility(_) => Ok(RecoveryAction::ManualIntervention(
                "Check module versions and update dependencies".to_string(),
            )),
            IntegrationError::VersionIncompatibility(_) => Ok(RecoveryAction::ManualIntervention(
                "Update to compatible module versions".to_string(),
            )),
            _ => Err(IntegrationError::ModuleCompatibility(
                "Cannot recover from non-compatibility error".to_string(),
            )),
        }
    }
}

/// Configuration recovery strategy
struct ConfigurationRecovery;

impl ErrorRecovery for ConfigurationRecovery {
    fn attempt_recovery(
        &self,
        error: &IntegrationError,
    ) -> Result<RecoveryAction, IntegrationError> {
        match error {
            IntegrationError::ConfigMismatch(_) => Ok(RecoveryAction::RetryWithConfig(
                [("use_defaults".to_string(), "true".to_string())].into(),
            )),
            _ => Err(IntegrationError::ConfigMismatch(
                "Cannot recover from non-configuration error".to_string(),
            )),
        }
    }
}

/// Global error mapper instance
static GLOBAL_ERROR_MAPPER: std::sync::OnceLock<std::sync::Mutex<ErrorMapper>> =
    std::sync::OnceLock::new();

/// Initialize global error mapper
pub fn init_error_mapper() -> &'static std::sync::Mutex<ErrorMapper> {
    GLOBAL_ERROR_MAPPER.get_or_init(|| std::sync::Mutex::new(ErrorMapper::new()))
}

/// Map error using global mapper
pub fn map_module_error(
    source_module: &str,
    source_error: &dyn std::error::Error,
) -> IntegrationError {
    let mapper = init_error_mapper();
    if let Ok(mapper_guard) = mapper.lock() {
        mapper_guard.map_error(source_module, source_error)
    } else {
        IntegrationError::ModuleCompatibility(format!(
            "Failed to acquire error mapper lock for {}: {}",
            source_module, source_error
        ))
    }
}

/// Push error context
pub fn push_error_context(context: ErrorContext) -> Result<(), IntegrationError> {
    let mapper = init_error_mapper();
    let mut mapper_guard = mapper.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire error mapper lock".to_string())
    })?;
    mapper_guard.push_context(context);
    Ok(())
}

/// Pop error context
pub fn pop_error_context() -> Result<Option<ErrorContext>, IntegrationError> {
    let mapper = init_error_mapper();
    let mut mapper_guard = mapper.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire error mapper lock".to_string())
    })?;
    Ok(mapper_guard.pop_context())
}

/// Generate error report
pub fn generate_error_report(error: &IntegrationError) -> Result<ErrorReport, IntegrationError> {
    let mapper = init_error_mapper();
    let mapper_guard = mapper.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire error mapper lock".to_string())
    })?;
    Ok(mapper_guard.generate_report(error))
}

/// Attempt error recovery
pub fn attempt_error_recovery(
    error: &IntegrationError,
) -> Result<RecoveryAction, IntegrationError> {
    let mapper = init_error_mapper();
    let mapper_guard = mapper.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire error mapper lock".to_string())
    })?;
    mapper_guard.attempt_recovery(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_mapper_creation() {
        let mapper = ErrorMapper::new();
        assert_eq!(mapper.mappings.len(), 4); // neural, optim, linalg, core
        assert_eq!(mapper.recovery_strategies.len(), 3);
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_module".to_string())
            .with_function("test_function".to_string())
            .with_operation("test_operation".to_string())
            .with_info("key".to_string(), "value".to_string());

        assert_eq!(context.module_name, "test_module");
        assert_eq!(context.function_name, Some("test_function".to_string()));
        assert_eq!(context.operation, Some("test_operation".to_string()));
        assert_eq!(
            context.additional_info.get("key"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_neural_error_mapping() {
        let mapping = NeuralErrorMapping;

        // Create a dummy error
        let dummy_error = IntegrationError::TensorConversion("tensor shape mismatch".to_string());
        let mapped = mapping.map_error(&dummy_error);

        assert!(matches!(mapped, IntegrationError::TensorConversion(_)));
    }

    #[test]
    fn test_error_classification() {
        let mapper = ErrorMapper::new();

        let tensor_error = IntegrationError::TensorConversion("test".to_string());
        assert_eq!(mapper.classify_error(&tensor_error), "tensor_conversion");

        let compat_error = IntegrationError::ModuleCompatibility("test".to_string());
        assert_eq!(mapper.classify_error(&compat_error), "module_compatibility");
    }

    #[test]
    fn test_error_severity() {
        let mapper = ErrorMapper::new();

        let config_error = IntegrationError::ConfigMismatch("test".to_string());
        assert_eq!(mapper.assess_severity(&config_error), ErrorSeverity::Low);

        let compat_error = IntegrationError::ModuleCompatibility("test".to_string());
        assert_eq!(mapper.assess_severity(&compat_error), ErrorSeverity::High);
    }

    #[test]
    fn test_recovery_strategy() {
        let recovery = TensorConversionRecovery;

        let shape_error = IntegrationError::TensorConversion("shape mismatch".to_string());
        let result = recovery.attempt_recovery(&shape_error).unwrap();

        assert!(matches!(result, RecoveryAction::RetryWithConfig(_)));
    }

    #[test]
    fn test_context_stack() {
        let mut mapper = ErrorMapper::new();

        let context1 = ErrorContext::new("module1".to_string());
        let context2 = ErrorContext::new("module2".to_string());

        mapper.push_context(context1);
        mapper.push_context(context2);

        assert_eq!(mapper.context_stack.len(), 2);
        assert_eq!(mapper.current_context().unwrap().module_name, "module2");

        let popped = mapper.pop_context().unwrap();
        assert_eq!(popped.module_name, "module2");
        assert_eq!(mapper.context_stack.len(), 1);
    }

    #[test]
    fn test_global_error_mapping() {
        let dummy_error = IntegrationError::TensorConversion("test error".to_string());
        let mapped = map_module_error("scirs2-neural", &dummy_error);

        // Should successfully map without panicking
        assert!(matches!(mapped, IntegrationError::TensorConversion(_)));
    }

    #[test]
    fn test_global_context_management() {
        let context = ErrorContext::new("test_module".to_string());

        push_error_context(context).unwrap();
        let popped = pop_error_context().unwrap();

        assert!(popped.is_some());
        assert_eq!(popped.unwrap().module_name, "test_module");
    }
}
