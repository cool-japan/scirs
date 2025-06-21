//! Configuration integration utilities for SciRS2 modules
//!
//! This module provides a unified configuration system that allows
//! consistent settings across different SciRS2 modules, with support
//! for environment variables, configuration files, and runtime updates.

use super::{IntegrationConfig, IntegrationError, MemoryStrategy, PrecisionLevel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Global configuration manager for SciRS2 integration
pub struct ConfigManager {
    /// Current configuration
    config: GlobalConfig,
    /// Configuration sources
    sources: Vec<ConfigSource>,
    /// Configuration cache
    cache: HashMap<String, ConfigValue>,
    /// Watch for configuration changes
    watch_enabled: bool,
}

impl ConfigManager {
    /// Create new configuration manager
    pub fn new() -> Self {
        Self {
            config: GlobalConfig::default(),
            sources: Vec::new(),
            cache: HashMap::new(),
            watch_enabled: false,
        }
    }

    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(&mut self, path: P) -> Result<(), IntegrationError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            IntegrationError::ConfigMismatch(format!("Failed to read config file: {}", e))
        })?;

        let file_config: FileConfig = toml::from_str(&content).map_err(|e| {
            IntegrationError::ConfigMismatch(format!("Failed to parse config file: {}", e))
        })?;

        // Merge file configuration
        self.merge_file_config(file_config)?;

        // Add source
        self.sources
            .push(ConfigSource::File(path.as_ref().to_path_buf()));

        Ok(())
    }

    /// Load configuration from environment variables
    pub fn load_from_env(&mut self) -> Result<(), IntegrationError> {
        let mut env_config = HashMap::new();

        // Check for SciRS2-specific environment variables
        for (key, value) in std::env::vars() {
            if key.starts_with("SCIRS2_") {
                let config_key = key.strip_prefix("SCIRS2_").unwrap().to_lowercase();
                env_config.insert(config_key, value);
            }
        }

        self.merge_env_config(env_config)?;
        self.sources.push(ConfigSource::Environment);

        Ok(())
    }

    /// Set configuration value
    pub fn set<T: Into<ConfigValue>>(&mut self, key: &str, value: T) {
        self.cache.insert(key.to_string(), value.into());
        self.apply_cached_values();
    }

    /// Get configuration value
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.cache.get(key)
    }

    /// Get configuration value with default
    pub fn get_or_default<T: From<ConfigValue> + Default>(&self, key: &str) -> T {
        self.cache
            .get(key)
            .cloned()
            .map(T::from)
            .unwrap_or_default()
    }

    /// Get integration configuration
    pub fn integration_config(&self) -> &IntegrationConfig {
        &self.config.integration
    }

    /// Update integration configuration
    pub fn update_integration_config(&mut self, config: IntegrationConfig) {
        self.config.integration = config;
    }

    /// Get module-specific configuration
    pub fn module_config(&self, module_name: &str) -> Option<&ModuleConfig> {
        self.config.modules.get(module_name)
    }

    /// Set module-specific configuration
    pub fn set_module_config(&mut self, module_name: String, config: ModuleConfig) {
        self.config.modules.insert(module_name, config);
    }

    /// Enable configuration watching
    pub fn enable_watch(&mut self) -> Result<(), IntegrationError> {
        self.watch_enabled = true;
        // In practice, would set up file system watchers
        Ok(())
    }

    /// Validate configuration consistency
    pub fn validate(&self) -> Result<(), IntegrationError> {
        // Check integration config consistency
        if self.config.integration.strict_compatibility {
            // Verify all modules are compatible
            for (module_name, module_config) in &self.config.modules {
                if !module_config.enabled {
                    continue;
                }

                // Check version compatibility
                if let Some(required_version) = &module_config.required_version {
                    // Validate version compatibility
                    self.validate_module_version(module_name, required_version)?;
                }
            }
        }

        // Check for conflicting settings
        self.check_conflicting_settings()?;

        Ok(())
    }

    /// Export configuration to file
    pub fn export_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), IntegrationError> {
        let file_config = self.to_file_config();
        let content = toml::to_string_pretty(&file_config).map_err(|e| {
            IntegrationError::ConfigMismatch(format!("Failed to serialize config: {}", e))
        })?;

        std::fs::write(path.as_ref(), content).map_err(|e| {
            IntegrationError::ConfigMismatch(format!("Failed to write config file: {}", e))
        })?;

        Ok(())
    }

    /// Reset to default configuration
    pub fn reset_to_defaults(&mut self) {
        self.config = GlobalConfig::default();
        self.cache.clear();
    }

    /// Get configuration summary
    pub fn summary(&self) -> ConfigSummary {
        ConfigSummary {
            total_modules: self.config.modules.len(),
            enabled_modules: self.config.modules.values().filter(|m| m.enabled).count(),
            precision_level: self.config.integration.default_precision,
            memory_strategy: self.config.integration.memory_strategy,
            sources: self.sources.clone(),
            cache_size: self.cache.len(),
        }
    }

    // Helper methods
    fn merge_file_config(&mut self, file_config: FileConfig) -> Result<(), IntegrationError> {
        // Merge integration settings
        if let Some(integration) = file_config.integration {
            self.config.integration = self.merge_integration_config(integration)?;
        }

        // Merge module configurations
        for (name, module_config) in file_config.modules.unwrap_or_default() {
            self.config.modules.insert(name, module_config);
        }

        Ok(())
    }

    fn merge_env_config(
        &mut self,
        env_config: HashMap<String, String>,
    ) -> Result<(), IntegrationError> {
        for (key, value) in env_config {
            match key.as_str() {
                "auto_convert_tensors" => {
                    let val = value.parse::<bool>().map_err(|_| {
                        IntegrationError::ConfigMismatch(format!(
                            "Invalid boolean value for {}: {}",
                            key, value
                        ))
                    })?;
                    self.config.integration.auto_convert_tensors = val;
                }
                "strict_compatibility" => {
                    let val = value.parse::<bool>().map_err(|_| {
                        IntegrationError::ConfigMismatch(format!(
                            "Invalid boolean value for {}: {}",
                            key, value
                        ))
                    })?;
                    self.config.integration.strict_compatibility = val;
                }
                "default_precision" => {
                    self.config.integration.default_precision = match value.as_str() {
                        "float32" => PrecisionLevel::Float32,
                        "float64" => PrecisionLevel::Float64,
                        "mixed" => PrecisionLevel::Mixed,
                        "adaptive" => PrecisionLevel::Adaptive,
                        _ => {
                            return Err(IntegrationError::ConfigMismatch(format!(
                                "Invalid precision level: {}",
                                value
                            )))
                        }
                    };
                }
                "memory_strategy" => {
                    self.config.integration.memory_strategy = match value.as_str() {
                        "shared" => MemoryStrategy::Shared,
                        "copy" => MemoryStrategy::Copy,
                        "memory_mapped" => MemoryStrategy::MemoryMapped,
                        "adaptive" => MemoryStrategy::Adaptive,
                        _ => {
                            return Err(IntegrationError::ConfigMismatch(format!(
                                "Invalid memory strategy: {}",
                                value
                            )))
                        }
                    };
                }
                _ => {
                    // Store as cache value for module-specific settings
                    self.cache.insert(key, ConfigValue::String(value));
                }
            }
        }

        Ok(())
    }

    fn merge_integration_config(
        &self,
        file_integration: FileIntegrationConfig,
    ) -> Result<IntegrationConfig, IntegrationError> {
        let mut config = self.config.integration.clone();

        if let Some(val) = file_integration.auto_convert_tensors {
            config.auto_convert_tensors = val;
        }

        if let Some(val) = file_integration.strict_compatibility {
            config.strict_compatibility = val;
        }

        if let Some(precision) = file_integration.default_precision {
            config.default_precision = match precision.as_str() {
                "float32" => PrecisionLevel::Float32,
                "float64" => PrecisionLevel::Float64,
                "mixed" => PrecisionLevel::Mixed,
                "adaptive" => PrecisionLevel::Adaptive,
                _ => {
                    return Err(IntegrationError::ConfigMismatch(format!(
                        "Invalid precision level: {}",
                        precision
                    )))
                }
            };
        }

        if let Some(strategy) = file_integration.memory_strategy {
            config.memory_strategy = match strategy.as_str() {
                "shared" => MemoryStrategy::Shared,
                "copy" => MemoryStrategy::Copy,
                "memory_mapped" => MemoryStrategy::MemoryMapped,
                "adaptive" => MemoryStrategy::Adaptive,
                _ => {
                    return Err(IntegrationError::ConfigMismatch(format!(
                        "Invalid memory strategy: {}",
                        strategy
                    )))
                }
            };
        }

        Ok(config)
    }

    fn apply_cached_values(&mut self) {
        // Apply cached values to main configuration
        for (key, value) in &self.cache {
            match key.as_str() {
                "auto_convert_tensors" => {
                    if let ConfigValue::Bool(val) = value {
                        self.config.integration.auto_convert_tensors = *val;
                    }
                }
                "strict_compatibility" => {
                    if let ConfigValue::Bool(val) = value {
                        self.config.integration.strict_compatibility = *val;
                    }
                }
                _ => {} // Module-specific or unknown settings
            }
        }
    }

    fn validate_module_version(
        &self,
        _module_name: &str,
        _version: &str,
    ) -> Result<(), IntegrationError> {
        // Simplified version validation
        Ok(())
    }

    fn check_conflicting_settings(&self) -> Result<(), IntegrationError> {
        // Check for conflicting configuration settings
        if self.config.integration.memory_strategy == MemoryStrategy::Shared
            && !self.config.integration.auto_convert_tensors
        {
            return Err(IntegrationError::ConfigMismatch(
                "Shared memory strategy requires auto tensor conversion".to_string(),
            ));
        }

        Ok(())
    }

    fn to_file_config(&self) -> FileConfig {
        let integration = FileIntegrationConfig {
            auto_convert_tensors: Some(self.config.integration.auto_convert_tensors),
            strict_compatibility: Some(self.config.integration.strict_compatibility),
            default_precision: Some(
                format!("{:?}", self.config.integration.default_precision).to_lowercase(),
            ),
            memory_strategy: Some(
                format!("{:?}", self.config.integration.memory_strategy).to_lowercase(),
            ),
            error_mode: Some(format!("{:?}", self.config.integration.error_mode).to_lowercase()),
        };

        FileConfig {
            integration: Some(integration),
            modules: Some(self.config.modules.clone()),
        }
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Global configuration structure
#[derive(Debug, Clone)]
pub struct GlobalConfig {
    /// Integration configuration
    pub integration: IntegrationConfig,
    /// Module-specific configurations
    pub modules: HashMap<String, ModuleConfig>,
    /// Performance configuration
    pub performance: PerformanceConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        let mut modules = HashMap::new();

        // Add default configurations for known modules
        modules.insert("scirs2-neural".to_string(), ModuleConfig::default_neural());
        modules.insert("scirs2-optim".to_string(), ModuleConfig::default_optim());
        modules.insert("scirs2-linalg".to_string(), ModuleConfig::default_linalg());

        Self {
            integration: IntegrationConfig::default(),
            modules,
            performance: PerformanceConfig::default(),
            logging: LoggingConfig::default(),
        }
    }
}

/// Configuration for individual modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleConfig {
    /// Whether the module is enabled
    pub enabled: bool,
    /// Required module version
    pub required_version: Option<String>,
    /// Module-specific settings
    pub settings: HashMap<String, ConfigValue>,
    /// Feature flags
    pub features: Vec<String>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
}

impl ModuleConfig {
    /// Create new module config
    pub fn new() -> Self {
        Self {
            enabled: true,
            required_version: None,
            settings: HashMap::new(),
            features: Vec::new(),
            resource_limits: ResourceLimits::default(),
        }
    }

    /// Default configuration for neural module
    pub fn default_neural() -> Self {
        let mut config = Self::new();
        config.features = vec![
            "automatic_differentiation".to_string(),
            "gradient_checkpointing".to_string(),
            "mixed_precision".to_string(),
        ];
        config
            .settings
            .insert("batch_size".to_string(), ConfigValue::Int(32));
        config
            .settings
            .insert("learning_rate".to_string(), ConfigValue::Float(0.001));
        config
    }

    /// Default configuration for optimization module
    pub fn default_optim() -> Self {
        let mut config = Self::new();
        config.features = vec![
            "adaptive_optimizers".to_string(),
            "learning_rate_scheduling".to_string(),
            "gradient_clipping".to_string(),
        ];
        config.settings.insert(
            "default_optimizer".to_string(),
            ConfigValue::String("adam".to_string()),
        );
        config
            .settings
            .insert("weight_decay".to_string(), ConfigValue::Float(1e-4));
        config
    }

    /// Default configuration for linear algebra module
    pub fn default_linalg() -> Self {
        let mut config = Self::new();
        config.features = vec![
            "blas_acceleration".to_string(),
            "gpu_support".to_string(),
            "numerical_stability".to_string(),
        ];
        config
            .settings
            .insert("use_blas".to_string(), ConfigValue::Bool(true));
        config
            .settings
            .insert("pivot_threshold".to_string(), ConfigValue::Float(1e-3));
        config
    }
}

impl Default for ModuleConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource limits for modules
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes
    pub max_memory: Option<usize>,
    /// Maximum number of threads
    pub max_threads: Option<usize>,
    /// Maximum computation time in seconds
    pub max_compute_time: Option<f64>,
    /// GPU memory limit in bytes
    pub max_gpu_memory: Option<usize>,
}

/// Performance configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Number of threads for parallel operations
    pub num_threads: Option<usize>,
    /// Cache size for computations
    pub cache_size: usize,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            num_threads: None,       // Use system default
            cache_size: 1024 * 1024, // 1MB
            enable_gpu: false,       // Disabled by default
        }
    }
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Enable module-specific logging
    pub module_logging: bool,
    /// Log file path
    pub log_file: Option<String>,
    /// Enable performance logging
    pub performance_logging: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            module_logging: true,
            log_file: None,
            performance_logging: false,
        }
    }
}

/// Log levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Configuration values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Array(Vec<ConfigValue>),
    Object(HashMap<String, ConfigValue>),
}

impl From<bool> for ConfigValue {
    fn from(value: bool) -> Self {
        ConfigValue::Bool(value)
    }
}

impl From<i64> for ConfigValue {
    fn from(value: i64) -> Self {
        ConfigValue::Int(value)
    }
}

impl From<f64> for ConfigValue {
    fn from(value: f64) -> Self {
        ConfigValue::Float(value)
    }
}

impl From<String> for ConfigValue {
    fn from(value: String) -> Self {
        ConfigValue::String(value)
    }
}

impl From<&str> for ConfigValue {
    fn from(value: &str) -> Self {
        ConfigValue::String(value.to_string())
    }
}

impl Default for ConfigValue {
    fn default() -> Self {
        ConfigValue::String(String::new())
    }
}

/// Configuration sources
#[derive(Debug, Clone)]
pub enum ConfigSource {
    File(std::path::PathBuf),
    Environment,
    Runtime,
    Default,
}

/// Configuration summary
#[derive(Debug, Clone)]
pub struct ConfigSummary {
    pub total_modules: usize,
    pub enabled_modules: usize,
    pub precision_level: PrecisionLevel,
    pub memory_strategy: MemoryStrategy,
    pub sources: Vec<ConfigSource>,
    pub cache_size: usize,
}

/// File-based configuration format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileConfig {
    pub integration: Option<FileIntegrationConfig>,
    pub modules: Option<HashMap<String, ModuleConfig>>,
}

/// Integration configuration in file format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileIntegrationConfig {
    pub auto_convert_tensors: Option<bool>,
    pub strict_compatibility: Option<bool>,
    pub default_precision: Option<String>,
    pub memory_strategy: Option<String>,
    pub error_mode: Option<String>,
}

/// Global configuration manager instance
static GLOBAL_CONFIG_MANAGER: std::sync::OnceLock<std::sync::Mutex<ConfigManager>> =
    std::sync::OnceLock::new();

/// Initialize global configuration manager
pub fn init_config_manager() -> &'static std::sync::Mutex<ConfigManager> {
    GLOBAL_CONFIG_MANAGER.get_or_init(|| {
        let mut manager = ConfigManager::new();

        // Try to load from environment
        let _ = manager.load_from_env();

        // Try to load from default config file
        if let Ok(config_path) = std::env::var("SCIRS2_CONFIG_PATH") {
            let _ = manager.load_from_file(config_path);
        }

        std::sync::Mutex::new(manager)
    })
}

/// Get global configuration value
pub fn get_config_value(key: &str) -> Result<Option<ConfigValue>, IntegrationError> {
    let manager = init_config_manager();
    let manager_guard = manager.lock().map_err(|_| {
        IntegrationError::ConfigMismatch("Failed to acquire config lock".to_string())
    })?;
    Ok(manager_guard.get(key).cloned())
}

/// Set global configuration value
pub fn set_config_value<T: Into<ConfigValue>>(key: &str, value: T) -> Result<(), IntegrationError> {
    let manager = init_config_manager();
    let mut manager_guard = manager.lock().map_err(|_| {
        IntegrationError::ConfigMismatch("Failed to acquire config lock".to_string())
    })?;
    manager_guard.set(key, value);
    Ok(())
}

/// Get module configuration
pub fn get_module_config(module_name: &str) -> Result<Option<ModuleConfig>, IntegrationError> {
    let manager = init_config_manager();
    let manager_guard = manager.lock().map_err(|_| {
        IntegrationError::ConfigMismatch("Failed to acquire config lock".to_string())
    })?;
    Ok(manager_guard.module_config(module_name).cloned())
}

/// Update global integration configuration
pub fn update_integration_config(config: IntegrationConfig) -> Result<(), IntegrationError> {
    let manager = init_config_manager();
    let mut manager_guard = manager.lock().map_err(|_| {
        IntegrationError::ConfigMismatch("Failed to acquire config lock".to_string())
    })?;
    manager_guard.update_integration_config(config);
    Ok(())
}

/// Load configuration from file
pub fn load_config_from_file<P: AsRef<Path>>(path: P) -> Result<(), IntegrationError> {
    let manager = init_config_manager();
    let mut manager_guard = manager.lock().map_err(|_| {
        IntegrationError::ConfigMismatch("Failed to acquire config lock".to_string())
    })?;
    manager_guard.load_from_file(path)
}

/// Export configuration to file
pub fn export_config_to_file<P: AsRef<Path>>(path: P) -> Result<(), IntegrationError> {
    let manager = init_config_manager();
    let manager_guard = manager.lock().map_err(|_| {
        IntegrationError::ConfigMismatch("Failed to acquire config lock".to_string())
    })?;
    manager_guard.export_to_file(path)
}

/// Get configuration summary
pub fn get_config_summary() -> Result<ConfigSummary, IntegrationError> {
    let manager = init_config_manager();
    let manager_guard = manager.lock().map_err(|_| {
        IntegrationError::ConfigMismatch("Failed to acquire config lock".to_string())
    })?;
    Ok(manager_guard.summary())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_config_manager_creation() {
        let manager = ConfigManager::new();
        assert_eq!(manager.config.modules.len(), 3); // neural, optim, linalg
    }

    #[test]
    fn test_config_value_types() {
        let bool_val = ConfigValue::Bool(true);
        let int_val = ConfigValue::Int(42);
        let float_val = ConfigValue::Float(std::f64::consts::PI);
        let string_val = ConfigValue::String("test".to_string());

        assert!(matches!(bool_val, ConfigValue::Bool(true)));
        assert!(matches!(int_val, ConfigValue::Int(42)));
        assert!(
            matches!(float_val, ConfigValue::Float(f) if (f - std::f64::consts::PI).abs() < 1e-10)
        );
        assert!(matches!(string_val, ConfigValue::String(ref s) if s == "test"));
    }

    #[test]
    fn test_module_config() {
        let neural_config = ModuleConfig::default_neural();
        assert!(neural_config.enabled);
        assert!(neural_config
            .features
            .contains(&"automatic_differentiation".to_string()));
        assert!(neural_config.settings.contains_key("batch_size"));
    }

    #[test]
    fn test_config_value_conversions() {
        let bool_val: ConfigValue = true.into();
        let int_val: ConfigValue = 42i64.into();
        let float_val: ConfigValue = std::f64::consts::PI.into();
        let string_val: ConfigValue = "test".into();

        assert!(matches!(bool_val, ConfigValue::Bool(true)));
        assert!(matches!(int_val, ConfigValue::Int(42)));
        assert!(
            matches!(float_val, ConfigValue::Float(f) if (f - std::f64::consts::PI).abs() < 1e-10)
        );
        assert!(matches!(string_val, ConfigValue::String(ref s) if s == "test"));
    }

    #[test]
    fn test_resource_limits() {
        let limits = ResourceLimits {
            max_memory: Some(1024 * 1024 * 1024), // 1GB
            max_threads: Some(8),
            max_compute_time: Some(60.0),            // 60 seconds
            max_gpu_memory: Some(512 * 1024 * 1024), // 512MB
        };

        assert_eq!(limits.max_memory, Some(1024 * 1024 * 1024));
        assert_eq!(limits.max_threads, Some(8));
        assert_eq!(limits.max_compute_time, Some(60.0));
        assert_eq!(limits.max_gpu_memory, Some(512 * 1024 * 1024));
    }

    #[test]
    fn test_global_config_default() {
        let config = GlobalConfig::default();
        assert!(config.modules.contains_key("scirs2-neural"));
        assert!(config.modules.contains_key("scirs2-optim"));
        assert!(config.modules.contains_key("scirs2-linalg"));
        assert!(config.performance.enable_simd);
    }

    #[test]
    fn test_config_manager_set_get() {
        let mut manager = ConfigManager::new();

        manager.set("test_key", ConfigValue::String("test_value".to_string()));
        let retrieved = manager.get("test_key");

        assert!(retrieved.is_some());
        if let Some(ConfigValue::String(val)) = retrieved {
            assert_eq!(val, "test_value");
        } else {
            panic!("Expected string value");
        }
    }

    #[test]
    fn test_env_config_merge() {
        let mut manager = ConfigManager::new();
        let mut env_vars = HashMap::new();
        env_vars.insert("auto_convert_tensors".to_string(), "false".to_string());
        env_vars.insert("strict_compatibility".to_string(), "true".to_string());

        manager.merge_env_config(env_vars).unwrap();

        assert!(!manager.config.integration.auto_convert_tensors);
        assert!(manager.config.integration.strict_compatibility);
    }

    #[test]
    fn test_config_validation() {
        let manager = ConfigManager::new();

        // Default config should be valid
        assert!(manager.validate().is_ok());
    }
}
