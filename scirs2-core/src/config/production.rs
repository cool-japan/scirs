//! # Production-Level Configuration Management
//!
//! This module provides comprehensive configuration management for production deployments,
//! including environment-specific settings, validation, hot reloading, and feature flags.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::sync::RwLock;
use std::time::SystemTime;

/// Configuration source types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigSource {
    /// Environment variables
    Environment,
    /// Configuration file (JSON, YAML, TOML)
    File(String),
    /// Command line arguments
    CommandLine,
    /// Default values
    Default,
    /// Runtime override
    Override,
}

/// Configuration value with metadata
#[derive(Debug, Clone)]
pub struct ConfigValue {
    /// The actual value
    pub value: String,
    /// Source of the configuration value
    pub source: ConfigSource,
    /// Timestamp when the value was set
    pub timestamp: SystemTime,
    /// Whether the value is sensitive (will be masked in logs)
    pub is_sensitive: bool,
    /// Description of what this configuration does
    pub description: Option<String>,
}

/// Configuration entry with validation
#[derive(Debug, Clone)]
pub struct ConfigEntry {
    /// Configuration key
    pub key: String,
    /// Current value
    pub value: ConfigValue,
    /// Validation function name
    pub validator: Option<String>,
    /// Whether this configuration can be changed at runtime
    pub hot_reloadable: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Environment variable name (if different from key)
    pub env_var: Option<String>,
}

/// Environment type for configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Environment {
    /// Development environment
    Development,
    /// Testing environment
    Testing,
    /// Staging environment
    Staging,
    /// Production environment
    Production,
    /// Custom environment
    Custom(String),
}

impl fmt::Display for Environment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Environment::Development => write!(f, "development"),
            Environment::Testing => write!(f, "testing"),
            Environment::Staging => write!(f, "staging"),
            Environment::Production => write!(f, "production"),
            Environment::Custom(name) => write!(f, "{}", name),
        }
    }
}

impl std::str::FromStr for Environment {
    type Err = std::convert::Infallible;

    /// Parse environment from string
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let env = match s.to_lowercase().as_str() {
            "dev" | "development" => Environment::Development,
            "test" | "testing" => Environment::Testing,
            "stage" | "staging" => Environment::Staging,
            "prod" | "production" => Environment::Production,
            name => Environment::Custom(name.to_string()),
        };
        Ok(env)
    }
}

/// Feature flag configuration
#[derive(Debug, Clone)]
pub struct FeatureFlag {
    /// Feature name
    pub name: String,
    /// Whether the feature is enabled
    pub enabled: bool,
    /// Rollout percentage (0-100)
    pub rollout_percentage: f64,
    /// Target environments
    pub environments: Vec<Environment>,
    /// Target user groups
    pub user_groups: Vec<String>,
    /// Description of the feature
    pub description: Option<String>,
    /// Timestamp when the flag was last modified
    pub last_modified: SystemTime,
}

/// Configuration validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
}

/// Trait for configuration validators
pub trait ConfigValidator: Send + Sync {
    /// Validate a configuration value
    fn validate(&self, value: &str) -> ValidationResult;

    /// Get the name of this validator
    fn name(&self) -> &str;

    /// Get help text for this validator
    fn help(&self) -> &str;
}

/// Built-in validators
/// Validates positive integers
pub struct PositiveIntValidator;

impl ConfigValidator for PositiveIntValidator {
    fn validate(&self, value: &str) -> ValidationResult {
        match value.parse::<i64>() {
            Ok(n) if n > 0 => ValidationResult {
                is_valid: true,
                errors: Vec::new(),
                warnings: Vec::new(),
            },
            Ok(n) => ValidationResult {
                is_valid: false,
                errors: vec![format!("Value must be positive, got {}", n)],
                warnings: Vec::new(),
            },
            Err(_) => ValidationResult {
                is_valid: false,
                errors: vec![format!("Invalid integer: {}", value)],
                warnings: Vec::new(),
            },
        }
    }

    fn name(&self) -> &str {
        "positive_int"
    }

    fn help(&self) -> &str {
        "Must be a positive integer greater than 0"
    }
}

/// Validates port numbers
pub struct PortValidator;

impl ConfigValidator for PortValidator {
    fn validate(&self, value: &str) -> ValidationResult {
        match value.parse::<u16>() {
            Ok(0) => ValidationResult {
                is_valid: false,
                errors: vec!["Port 0 is not allowed".to_string()],
                warnings: Vec::new(),
            },
            Ok(port) => ValidationResult {
                is_valid: true,
                errors: Vec::new(),
                warnings: if port < 1024 {
                    vec!["Port number is below 1024 (requires root privileges)".to_string()]
                } else {
                    Vec::new()
                },
            },
            Err(_) => ValidationResult {
                is_valid: false,
                errors: vec![format!("Invalid port number: {}", value)],
                warnings: Vec::new(),
            },
        }
    }

    fn name(&self) -> &str {
        "port"
    }

    fn help(&self) -> &str {
        "Must be a valid port number (1-65535)"
    }
}

/// Validates URL format
pub struct UrlValidator;

impl ConfigValidator for UrlValidator {
    fn validate(&self, value: &str) -> ValidationResult {
        // Simple URL validation (in production, use a proper URL parsing library)
        if value.starts_with("http://") || value.starts_with("https://") {
            ValidationResult {
                is_valid: true,
                errors: Vec::new(),
                warnings: if value.starts_with("http://") {
                    vec!["Using HTTP instead of HTTPS may be insecure".to_string()]
                } else {
                    Vec::new()
                },
            }
        } else {
            ValidationResult {
                is_valid: false,
                errors: vec![format!("Invalid URL format: {}", value)],
                warnings: Vec::new(),
            }
        }
    }

    fn name(&self) -> &str {
        "url"
    }

    fn help(&self) -> &str {
        "Must be a valid HTTP or HTTPS URL"
    }
}

/// Production configuration manager
pub struct ProductionConfig {
    /// Configuration entries
    entries: RwLock<HashMap<String, ConfigEntry>>,
    /// Feature flags
    feature_flags: RwLock<HashMap<String, FeatureFlag>>,
    /// Available validators
    validators: HashMap<String, Box<dyn ConfigValidator>>,
    /// Current environment
    environment: Environment,
    /// Configuration file watchers
    file_watchers: RwLock<HashMap<String, SystemTime>>,
    /// Hot reload enabled
    hot_reload_enabled: bool,
}

impl ProductionConfig {
    /// Create a new production configuration manager
    pub fn new() -> Self {
        let mut validators: HashMap<String, Box<dyn ConfigValidator>> = HashMap::new();
        validators.insert("positive_int".to_string(), Box::new(PositiveIntValidator));
        validators.insert("port".to_string(), Box::new(PortValidator));
        validators.insert("url".to_string(), Box::new(UrlValidator));

        // Determine environment from ENV var
        let env_str = env::var("SCIRS_ENV").unwrap_or_else(|_| "development".to_string());
        let environment = Environment::from_str(&env_str).unwrap_or(Environment::Development);

        Self {
            entries: RwLock::new(HashMap::new()),
            feature_flags: RwLock::new(HashMap::new()),
            validators,
            environment,
            file_watchers: RwLock::new(HashMap::new()),
            hot_reload_enabled: true,
        }
    }

    /// Load configuration from environment variables
    pub fn load_from_env(&self) -> CoreResult<()> {
        let mut entries = self.entries.write().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        // Load all environment variables with SCIRS_ prefix
        for (key, value) in env::vars() {
            if key.starts_with("SCIRS_") {
                let config_key = key.strip_prefix("SCIRS_").unwrap().to_lowercase();

                let config_value = ConfigValue {
                    value: value.clone(),
                    source: ConfigSource::Environment,
                    timestamp: SystemTime::now(),
                    is_sensitive: self.is_sensitive_key(&config_key),
                    description: None,
                };

                let entry = ConfigEntry {
                    key: config_key.clone(),
                    value: config_value,
                    validator: None,
                    hot_reloadable: false, // Env vars are not hot reloadable
                    default_value: None,
                    env_var: Some(key),
                };

                entries.insert(config_key, entry);
            }
        }

        Ok(())
    }

    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(&self, path: P) -> CoreResult<()> {
        let path = path.as_ref();
        let content = fs::read_to_string(path).map_err(|e| {
            CoreError::ConfigError(ErrorContext::new(format!(
                "Failed to read config file {}: {}",
                path.display(),
                e
            )))
        })?;

        // Track file modification time for hot reloading
        if let Ok(metadata) = fs::metadata(path) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(mut watchers) = self.file_watchers.write() {
                    watchers.insert(path.to_string_lossy().to_string(), modified);
                }
            }
        }

        // Parse configuration based on file extension
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("json") => self.parse_json_config(&content),
            Some("yaml" | "yml") => self.parse_yaml_config(&content),
            Some("toml") => self.parse_toml_config(&content),
            _ => Err(CoreError::ConfigError(ErrorContext::new(format!(
                "Unsupported config file format: {}",
                path.display()
            )))),
        }
    }

    /// Parse JSON configuration
    fn parse_json_config(&self, _content: &str) -> CoreResult<()> {
        // In a real implementation, use serde_json
        Err(CoreError::ConfigError(ErrorContext::new(
            "JSON parsing not implemented in this example",
        )))
    }

    /// Parse YAML configuration
    fn parse_yaml_config(&self, _content: &str) -> CoreResult<()> {
        // In a real implementation, use serde_yaml
        Err(CoreError::ConfigError(ErrorContext::new(
            "YAML parsing not implemented in this example",
        )))
    }

    /// Parse TOML configuration
    fn parse_toml_config(&self, _content: &str) -> CoreResult<()> {
        // In a real implementation, use toml crate
        Err(CoreError::ConfigError(ErrorContext::new(
            "TOML parsing not implemented in this example",
        )))
    }

    /// Set a configuration value
    pub fn set<S: Into<String>>(&self, key: S, value: S) -> CoreResult<()> {
        let key = key.into();
        let value = value.into();

        // Validate the value if a validator is specified
        if let Some(entry) = self.get_entry(&key)? {
            if let Some(validator_name) = &entry.validator {
                if let Some(validator) = self.validators.get(validator_name) {
                    let validation_result = validator.validate(&value);
                    if !validation_result.is_valid {
                        return Err(CoreError::ConfigError(ErrorContext::new(format!(
                            "Validation failed for {}: {}",
                            key,
                            validation_result.errors.join(", ")
                        ))));
                    }
                }
            }
        }

        let mut entries = self.entries.write().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        let config_value = ConfigValue {
            value,
            source: ConfigSource::Override,
            timestamp: SystemTime::now(),
            is_sensitive: self.is_sensitive_key(&key),
            description: None,
        };

        if let Some(entry) = entries.get_mut(&key) {
            entry.value = config_value;
        } else {
            let entry = ConfigEntry {
                key: key.clone(),
                value: config_value,
                validator: None,
                hot_reloadable: true,
                default_value: None,
                env_var: None,
            };
            entries.insert(key, entry);
        }

        Ok(())
    }

    /// Get a configuration value
    pub fn get(&self, key: &str) -> CoreResult<Option<String>> {
        let entries = self.entries.read().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        Ok(entries.get(key).map(|entry| entry.value.value.clone()))
    }

    /// Get a configuration entry with metadata
    pub fn get_entry(&self, key: &str) -> CoreResult<Option<ConfigEntry>> {
        let entries = self.entries.read().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        Ok(entries.get(key).cloned())
    }

    /// Get a configuration value with type conversion
    pub fn get_typed<T>(&self, key: &str) -> CoreResult<Option<T>>
    where
        T: std::str::FromStr,
        T::Err: fmt::Display,
    {
        if let Some(value_str) = self.get(key)? {
            match value_str.parse::<T>() {
                Ok(value) => Ok(Some(value)),
                Err(e) => Err(CoreError::ConfigError(ErrorContext::new(format!(
                    "Failed to parse config value '{}' for key '{}': {}",
                    value_str, key, e
                )))),
            }
        } else {
            Ok(None)
        }
    }

    /// Get a configuration value with default
    pub fn get_or_default<T>(&self, key: &str, default: T) -> CoreResult<T>
    where
        T: std::str::FromStr + Clone,
        T::Err: fmt::Display,
    {
        match self.get_typed::<T>(key)? {
            Some(value) => Ok(value),
            None => Ok(default),
        }
    }

    /// Register a configuration entry with validation
    pub fn register_config(
        &self,
        key: String,
        default_value: Option<String>,
        validator: Option<String>,
        hot_reloadable: bool,
        description: Option<String>,
    ) -> CoreResult<()> {
        let mut entries = self.entries.write().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        // Check if already exists
        if entries.contains_key(&key) {
            return Ok(()); // Already registered
        }

        let value = if let Some(default) = &default_value {
            ConfigValue {
                value: default.clone(),
                source: ConfigSource::Default,
                timestamp: SystemTime::now(),
                is_sensitive: self.is_sensitive_key(&key),
                description: description.clone(),
            }
        } else {
            ConfigValue {
                value: String::new(),
                source: ConfigSource::Default,
                timestamp: SystemTime::now(),
                is_sensitive: self.is_sensitive_key(&key),
                description: description.clone(),
            }
        };

        let entry = ConfigEntry {
            key: key.clone(),
            value,
            validator,
            hot_reloadable,
            default_value,
            env_var: Some(format!("SCIRS_{}", key.to_uppercase())),
        };

        entries.insert(key, entry);
        Ok(())
    }

    /// Check if a configuration key is sensitive
    fn is_sensitive_key(&self, key: &str) -> bool {
        let sensitive_patterns = ["password", "secret", "key", "token", "credential", "auth"];
        let key_lower = key.to_lowercase();
        sensitive_patterns
            .iter()
            .any(|pattern| key_lower.contains(pattern))
    }

    /// Get current environment
    pub const fn environment(&self) -> &Environment {
        &self.environment
    }

    /// Set feature flag
    pub fn set_feature_flag(
        &self,
        name: String,
        enabled: bool,
        rollout_percentage: f64,
    ) -> CoreResult<()> {
        let mut flags = self.feature_flags.write().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire feature flags lock"))
        })?;

        let flag = FeatureFlag {
            name: name.clone(),
            enabled,
            rollout_percentage: rollout_percentage.clamp(0.0, 100.0),
            environments: vec![self.environment.clone()],
            user_groups: Vec::new(),
            description: None,
            last_modified: SystemTime::now(),
        };

        flags.insert(name, flag);
        Ok(())
    }

    /// Check if feature is enabled
    pub fn is_feature_enabled(&self, name: &str) -> bool {
        if let Ok(flags) = self.feature_flags.read() {
            if let Some(flag) = flags.get(name) {
                return flag.enabled
                    && flag.environments.contains(&self.environment)
                    && self.check_rollout_percentage(flag.rollout_percentage);
            }
        }
        false
    }

    /// Check rollout percentage (simplified implementation)
    fn check_rollout_percentage(&self, percentage: f64) -> bool {
        // In a real implementation, this would use consistent hashing
        // based on user ID or session ID
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        SystemTime::now().hash(&mut hasher);
        let hash = hasher.finish();

        (hash % 100) < percentage as u64
    }

    /// Validate all configuration
    pub fn validate_all(&self) -> CoreResult<HashMap<String, ValidationResult>> {
        let entries = self.entries.read().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        let mut results = HashMap::new();

        for (key, entry) in entries.iter() {
            if let Some(validator_name) = &entry.validator {
                if let Some(validator) = self.validators.get(validator_name) {
                    let result = validator.validate(&entry.value.value);
                    results.insert(key.clone(), result);
                }
            }
        }

        Ok(results)
    }

    /// Get configuration summary for monitoring
    pub fn get_summary(&self) -> CoreResult<ConfigSummary> {
        let entries = self.entries.read().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire config lock"))
        })?;

        let flags = self.feature_flags.read().map_err(|_| {
            CoreError::ConfigError(ErrorContext::new("Failed to acquire feature flags lock"))
        })?;

        let total_configs = entries.len();
        let env_configs = entries
            .values()
            .filter(|e| e.value.source == ConfigSource::Environment)
            .count();
        let file_configs = entries
            .values()
            .filter(|e| matches!(e.value.source, ConfigSource::File(_)))
            .count();
        let override_configs = entries
            .values()
            .filter(|e| e.value.source == ConfigSource::Override)
            .count();
        let sensitive_configs = entries.values().filter(|e| e.value.is_sensitive).count();

        let total_flags = flags.len();
        let enabled_flags = flags.values().filter(|f| f.enabled).count();

        Ok(ConfigSummary {
            environment: self.environment.clone(),
            total_configs,
            env_configs,
            file_configs,
            override_configs,
            sensitive_configs,
            total_flags,
            enabled_flags,
            hot_reload_enabled: self.hot_reload_enabled,
        })
    }

    /// Reload configuration from files (hot reload)
    pub fn reload(&self) -> CoreResult<Vec<String>> {
        if !self.hot_reload_enabled {
            return Err(CoreError::ConfigError(ErrorContext::new(
                "Hot reload is disabled",
            )));
        }

        let mut reloaded_files = Vec::new();

        if let Ok(watchers) = self.file_watchers.read() {
            for (file_path, last_modified) in watchers.iter() {
                if let Ok(metadata) = fs::metadata(file_path) {
                    if let Ok(current_modified) = metadata.modified() {
                        if current_modified > *last_modified {
                            // File has been modified, reload it
                            if let Err(e) = self.load_from_file(file_path) {
                                eprintln!("Failed to reload config file {}: {}", file_path, e);
                            } else {
                                reloaded_files.push(file_path.clone());
                            }
                        }
                    }
                }
            }
        }

        Ok(reloaded_files)
    }
}

impl Default for ProductionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration summary for monitoring
#[derive(Debug, Clone)]
pub struct ConfigSummary {
    pub environment: Environment,
    pub total_configs: usize,
    pub env_configs: usize,
    pub file_configs: usize,
    pub override_configs: usize,
    pub sensitive_configs: usize,
    pub total_flags: usize,
    pub enabled_flags: usize,
    pub hot_reload_enabled: bool,
}

impl fmt::Display for ConfigSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Configuration Summary:")?;
        writeln!(f, "  Environment: {}", self.environment)?;
        writeln!(f, "  Total configurations: {}", self.total_configs)?;
        writeln!(f, "    From environment: {}", self.env_configs)?;
        writeln!(f, "    From files: {}", self.file_configs)?;
        writeln!(f, "    Runtime overrides: {}", self.override_configs)?;
        writeln!(f, "    Sensitive configs: {}", self.sensitive_configs)?;
        writeln!(f, "  Feature flags:")?;
        writeln!(f, "    Total: {}", self.total_flags)?;
        writeln!(f, "    Enabled: {}", self.enabled_flags)?;
        writeln!(
            f,
            "  Hot reload: {}",
            if self.hot_reload_enabled {
                "enabled"
            } else {
                "disabled"
            }
        )?;
        Ok(())
    }
}

/// Global configuration instance
static GLOBAL_CONFIG: std::sync::LazyLock<ProductionConfig> = std::sync::LazyLock::new(|| {
    let config = ProductionConfig::new();

    // Load from environment on startup
    if let Err(e) = config.load_from_env() {
        eprintln!(
            "Warning: Failed to load configuration from environment: {}",
            e
        );
    }

    // Register common configurations
    let _ = config.register_config(
        "log_level".to_string(),
        Some("info".to_string()),
        None,
        true,
        Some("Logging level (trace, debug, info, warn, error)".to_string()),
    );

    let _ = config.register_config(
        "max_memory_mb".to_string(),
        Some("1024".to_string()),
        Some("positive_int".to_string()),
        true,
        Some("Maximum memory usage in megabytes".to_string()),
    );

    let _ = config.register_config(
        "worker_threads".to_string(),
        Some("4".to_string()),
        Some("positive_int".to_string()),
        false,
        Some("Number of worker threads".to_string()),
    );

    config
});

/// Get the global configuration instance
pub fn global_config() -> &'static ProductionConfig {
    &GLOBAL_CONFIG
}

/// Configuration convenience macros
/// Get a configuration value with type conversion
#[macro_export]
macro_rules! config_get {
    ($key:expr) => {
        $crate::config::production::global_config().get($key)
    };
    ($key:expr, $type:ty) => {
        $crate::config::production::global_config().get_typed::<$type>($key)
    };
    ($key:expr, $type:ty, $default:expr) => {
        $crate::config::production::global_config().get_or_default::<$type>($key, $default)
    };
}

/// Set a configuration value
#[macro_export]
macro_rules! config_set {
    ($key:expr, $value:expr) => {
        $crate::config::production::global_config().set($key, $value)
    };
}

/// Check if a feature is enabled
#[macro_export]
macro_rules! feature_enabled {
    ($feature:expr) => {
        $crate::config::production::global_config().is_feature_enabled($feature)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_parsing() {
        assert_eq!(
            Environment::from_str("development"),
            Ok(Environment::Development)
        );
        assert_eq!(Environment::from_str("dev"), Ok(Environment::Development));
        assert_eq!(
            Environment::from_str("production"),
            Ok(Environment::Production)
        );
        assert_eq!(
            Environment::from_str("custom"),
            Ok(Environment::Custom("custom".to_string()))
        );
    }

    #[test]
    fn test_validators() {
        let validator = PositiveIntValidator;

        let result = validator.validate("42");
        assert!(result.is_valid);

        let result = validator.validate("-5");
        assert!(!result.is_valid);

        let result = validator.validate("not_a_number");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_config_operations() {
        let config = ProductionConfig::new();

        // Test setting and getting
        config.set("test_key", "test_value").unwrap();
        assert_eq!(
            config.get("test_key").unwrap(),
            Some("test_value".to_string())
        );

        // Test typed get
        config.set("test_number", "42").unwrap();
        assert_eq!(config.get_typed::<i32>("test_number").unwrap(), Some(42));

        // Test default
        assert_eq!(config.get_or_default("missing_key", 100i32).unwrap(), 100);
    }

    #[test]
    fn test_feature_flags() {
        let config = ProductionConfig::new();

        config
            .set_feature_flag("test_feature".to_string(), true, 100.0)
            .unwrap();
        assert!(config.is_feature_enabled("test_feature"));

        config
            .set_feature_flag("disabled_feature".to_string(), false, 100.0)
            .unwrap();
        assert!(!config.is_feature_enabled("disabled_feature"));
    }

    #[test]
    fn test_sensitive_key_detection() {
        let config = ProductionConfig::new();

        assert!(config.is_sensitive_key("api_password"));
        assert!(config.is_sensitive_key("SECRET_KEY"));
        assert!(config.is_sensitive_key("auth_token"));
        assert!(!config.is_sensitive_key("log_level"));
        assert!(!config.is_sensitive_key("max_connections"));
    }

    #[test]
    fn test_config_registration() {
        let config = ProductionConfig::new();

        config
            .register_config(
                "test_config".to_string(),
                Some("default_value".to_string()),
                Some("positive_int".to_string()),
                true,
                Some("Test configuration".to_string()),
            )
            .unwrap();

        let entry = config.get_entry("test_config").unwrap().unwrap();
        assert_eq!(entry.default_value, Some("default_value".to_string()));
        assert_eq!(entry.validator, Some("positive_int".to_string()));
        assert!(entry.hot_reloadable);
    }
}
