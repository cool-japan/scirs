//! Integration utilities for working with other SciRS2 modules
//!
//! This module provides seamless integration capabilities with other modules
//! in the SciRS2 ecosystem, including tensor conversion utilities, shared
//! configuration systems, and consistent API patterns.

use crate::AutogradError;

pub mod config;
pub mod core;
pub mod error_mapping;
pub mod linalg;
pub mod neural;
pub mod optim;
pub mod tensor_conversion;

/// Integration error types for cross-module compatibility
#[derive(Debug, Clone, thiserror::Error)]
pub enum IntegrationError {
    #[error("Tensor conversion error: {0}")]
    TensorConversion(String),
    #[error("Module compatibility error: {0}")]
    ModuleCompatibility(String),
    #[error("Configuration mismatch: {0}")]
    ConfigMismatch(String),
    #[error("Version incompatibility: {0}")]
    VersionIncompatibility(String),
    #[error("API boundary error: {0}")]
    ApiBoundary(String),
}

impl From<IntegrationError> for AutogradError {
    fn from(err: IntegrationError) -> Self {
        AutogradError::IntegrationError(err.to_string())
    }
}

/// Trait for SciRS2 module integration capabilities
///
/// This trait provides basic integration patterns that work with autograd's
/// computational graph design. Complex tensor operations should use tensor_ops
/// functions rather than direct data access.
pub trait SciRS2Integration {
    /// Module identifier
    fn module_name() -> &'static str;

    /// Module version for compatibility checking
    fn module_version() -> &'static str;

    /// Check compatibility with the autograd module
    fn check_compatibility() -> Result<(), IntegrationError>;
}

/// Shared configuration for cross-module compatibility
#[derive(Debug, Clone)]
pub struct IntegrationConfig {
    /// Enable automatic tensor conversion
    pub auto_convert_tensors: bool,
    /// Strict compatibility checking
    pub strict_compatibility: bool,
    /// Default floating point precision
    pub default_precision: PrecisionLevel,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
    /// Error handling mode
    pub error_mode: ErrorMode,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            auto_convert_tensors: true,
            strict_compatibility: false,
            default_precision: PrecisionLevel::Float32,
            memory_strategy: MemoryStrategy::Shared,
            error_mode: ErrorMode::Propagate,
        }
    }
}

/// Precision levels for cross-module operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrecisionLevel {
    Float32,
    Float64,
    Mixed,
    Adaptive,
}

/// Memory management strategies for integration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryStrategy {
    /// Share memory between modules when possible
    Shared,
    /// Copy data between modules
    Copy,
    /// Use memory mapping for large tensors
    MemoryMapped,
    /// Adaptive strategy based on tensor size
    Adaptive,
}

/// Error handling modes for integration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorMode {
    /// Propagate errors immediately
    Propagate,
    /// Collect errors and handle batch-wise
    Collect,
    /// Attempt recovery with fallback strategies
    Recovery,
}

/// Global integration registry for SciRS2 modules
pub struct IntegrationRegistry {
    modules: std::collections::HashMap<String, ModuleInfo>,
    config: IntegrationConfig,
}

impl IntegrationRegistry {
    /// Create a new integration registry
    pub fn new() -> Self {
        Self {
            modules: std::collections::HashMap::new(),
            config: IntegrationConfig::default(),
        }
    }

    /// Register a module with the integration system
    pub fn register_module(&mut self, info: ModuleInfo) -> Result<(), IntegrationError> {
        // Check for version conflicts
        if let Some(existing) = self.modules.get(&info.name) {
            if existing.version != info.version {
                return Err(IntegrationError::VersionIncompatibility(format!(
                    "Module {} version mismatch: {} vs {}",
                    info.name, existing.version, info.version
                )));
            }
        }

        self.modules.insert(info.name.clone(), info);
        Ok(())
    }

    /// Get module information
    pub fn get_module(&self, name: &str) -> Option<&ModuleInfo> {
        self.modules.get(name)
    }

    /// Check compatibility between modules
    pub fn check_module_compatibility(
        &self,
        module1: &str,
        module2: &str,
    ) -> Result<bool, IntegrationError> {
        let mod1 = self.modules.get(module1).ok_or_else(|| {
            IntegrationError::ModuleCompatibility(format!("Module {} not found", module1))
        })?;
        let mod2 = self.modules.get(module2).ok_or_else(|| {
            IntegrationError::ModuleCompatibility(format!("Module {} not found", module2))
        })?;

        // Simple version compatibility check
        Ok(self.are_versions_compatible(&mod1.version, &mod2.version))
    }

    /// Update global configuration
    pub fn update_config(&mut self, config: IntegrationConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn config(&self) -> &IntegrationConfig {
        &self.config
    }

    /// List all registered modules
    pub fn list_modules(&self) -> Vec<&ModuleInfo> {
        self.modules.values().collect()
    }

    fn are_versions_compatible(&self, version1: &str, version2: &str) -> bool {
        // Simplified version compatibility check
        // In practice, this would implement semantic versioning rules
        let v1_parts: Vec<&str> = version1.split('.').collect();
        let v2_parts: Vec<&str> = version2.split('.').collect();

        if v1_parts.len() >= 2 && v2_parts.len() >= 2 {
            // Check major.minor compatibility
            v1_parts[0] == v2_parts[0] && v1_parts[1] == v2_parts[1]
        } else {
            version1 == version2
        }
    }
}

impl Default for IntegrationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Information about a registered SciRS2 module
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    pub name: String,
    pub version: String,
    pub features: Vec<String>,
    pub dependencies: Vec<String>,
    pub api_version: String,
}

impl ModuleInfo {
    /// Create new module information
    pub fn new(name: String, version: String) -> Self {
        Self {
            name,
            version,
            features: Vec::new(),
            dependencies: Vec::new(),
            api_version: "1.0".to_string(),
        }
    }

    /// Add a feature to the module
    pub fn with_feature(mut self, feature: String) -> Self {
        self.features.push(feature);
        self
    }

    /// Add a dependency to the module
    pub fn with_dependency(mut self, dependency: String) -> Self {
        self.dependencies.push(dependency);
        self
    }

    /// Set API version
    pub fn with_api_version(mut self, api_version: String) -> Self {
        self.api_version = api_version;
        self
    }
}

/// Global integration registry instance
static GLOBAL_REGISTRY: std::sync::OnceLock<std::sync::Mutex<IntegrationRegistry>> =
    std::sync::OnceLock::new();

/// Initialize the global integration registry
pub fn init_integration_registry() -> &'static std::sync::Mutex<IntegrationRegistry> {
    GLOBAL_REGISTRY.get_or_init(|| {
        let mut registry = IntegrationRegistry::new();

        // Register autograd module
        let autograd_info =
            ModuleInfo::new("scirs2-autograd".to_string(), "0.1.0-alpha.6".to_string())
                .with_feature("automatic_differentiation".to_string())
                .with_feature("computation_graphs".to_string())
                .with_feature("gradient_computation".to_string())
                .with_api_version("1.0".to_string());

        registry
            .register_module(autograd_info)
            .expect("Failed to register autograd module");

        std::sync::Mutex::new(registry)
    })
}

/// Register a module with the global registry
pub fn register_module(info: ModuleInfo) -> Result<(), IntegrationError> {
    let registry = init_integration_registry();
    let mut registry_guard = registry.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire registry lock".to_string())
    })?;
    registry_guard.register_module(info)
}

/// Get module information from the global registry
pub fn get_module_info(name: &str) -> Result<Option<ModuleInfo>, IntegrationError> {
    let registry = init_integration_registry();
    let registry_guard = registry.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire registry lock".to_string())
    })?;
    Ok(registry_guard.get_module(name).cloned())
}

/// Check compatibility between two modules
pub fn check_compatibility(module1: &str, module2: &str) -> Result<bool, IntegrationError> {
    let registry = init_integration_registry();
    let registry_guard = registry.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire registry lock".to_string())
    })?;
    registry_guard.check_module_compatibility(module1, module2)
}

/// Update global integration configuration
pub fn update_global_config(config: IntegrationConfig) -> Result<(), IntegrationError> {
    let registry = init_integration_registry();
    let mut registry_guard = registry.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire registry lock".to_string())
    })?;
    registry_guard.update_config(config);
    Ok(())
}

/// Get current global integration configuration
pub fn get_global_config() -> Result<IntegrationConfig, IntegrationError> {
    let registry = init_integration_registry();
    let registry_guard = registry.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire registry lock".to_string())
    })?;
    Ok(registry_guard.config().clone())
}

/// List all registered modules
pub fn list_registered_modules() -> Result<Vec<ModuleInfo>, IntegrationError> {
    let registry = init_integration_registry();
    let registry_guard = registry.lock().map_err(|_| {
        IntegrationError::ModuleCompatibility("Failed to acquire registry lock".to_string())
    })?;
    Ok(registry_guard.list_modules().into_iter().cloned().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integration_config_default() {
        let config = IntegrationConfig::default();
        assert!(config.auto_convert_tensors);
        assert!(!config.strict_compatibility);
        assert_eq!(config.default_precision, PrecisionLevel::Float32);
        assert_eq!(config.memory_strategy, MemoryStrategy::Shared);
        assert_eq!(config.error_mode, ErrorMode::Propagate);
    }

    #[test]
    fn test_module_info_creation() {
        let info = ModuleInfo::new("test-module".to_string(), "1.0.0".to_string())
            .with_feature("feature1".to_string())
            .with_dependency("dep1".to_string())
            .with_api_version("2.0".to_string());

        assert_eq!(info.name, "test-module");
        assert_eq!(info.version, "1.0.0");
        assert_eq!(info.features, vec!["feature1"]);
        assert_eq!(info.dependencies, vec!["dep1"]);
        assert_eq!(info.api_version, "2.0");
    }

    #[test]
    fn test_integration_registry() {
        let mut registry = IntegrationRegistry::new();

        let info = ModuleInfo::new("test-module".to_string(), "1.0.0".to_string());
        registry.register_module(info.clone()).unwrap();

        let retrieved = registry.get_module("test-module").unwrap();
        assert_eq!(retrieved.name, "test-module");
        assert_eq!(retrieved.version, "1.0.0");
    }

    #[test]
    fn test_version_compatibility() {
        let registry = IntegrationRegistry::new();
        assert!(registry.are_versions_compatible("1.0.0", "1.0.1"));
        assert!(registry.are_versions_compatible("1.1.0", "1.1.5"));
        assert!(!registry.are_versions_compatible("1.0.0", "2.0.0"));
        assert!(!registry.are_versions_compatible("1.0.0", "1.1.0"));
    }

    #[test]
    fn test_global_registry() {
        // Test that we can get module info
        let info = get_module_info("scirs2-autograd").unwrap();
        assert!(info.is_some());

        let autograd_info = info.unwrap();
        assert_eq!(autograd_info.name, "scirs2-autograd");
        assert!(autograd_info
            .features
            .contains(&"automatic_differentiation".to_string()));
    }
}
