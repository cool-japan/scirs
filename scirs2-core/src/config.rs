//! Configuration system for ``SciRS2``
//!
//! This module provides a centralized configuration system for ``SciRS2``, allowing
//! users to customize the behavior of various algorithms and functions.

use std::collections::HashMap;
use std::env;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};

/// Default precision for floating-point comparisons
pub const DEFAULT_FLOAT_EPS: f64 = 1e-10;

/// Default number of threads to use for parallel operations
pub const DEFAULT_NUM_THREADS: usize = 4;

/// Default maximum iterations for iterative algorithms
pub const DEFAULT_MAX_ITERATIONS: usize = 1000;

/// Default memory limit for operations (in bytes)
pub const DEFAULT_MEMORY_LIMIT: usize = 1_073_741_824; // 1 GB

/// Global configuration for ``SciRS2``
///
/// This struct is not intended to be instantiated directly.
/// Instead, use the `get_config` and `set_config` functions to access and modify the configuration.
#[derive(Debug, Clone)]
pub struct Config {
    /// Configuration values
    values: HashMap<String, ConfigValue>,
    /// Environment prefix for ``SciRS2`` environment variables
    env_prefix: String,
}

/// Configuration value types
#[derive(Debug, Clone)]
pub enum ConfigValue {
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i64),
    /// Unsigned integer value
    UInt(u64),
    /// Floating-point value
    Float(f64),
    /// String value
    String(String),
}

impl fmt::Display for ConfigValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigValue::Bool(b) => write!(f, "{b}"),
            ConfigValue::Int(i) => write!(f, "{i}"),
            ConfigValue::UInt(u) => write!(f, "{u}"),
            ConfigValue::Float(fl) => write!(f, "{fl}"),
            ConfigValue::String(s) => write!(f, "{s}"),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        let mut config = Self {
            values: HashMap::new(),
            env_prefix: "SCIRS_".to_string(),
        };

        // Set default values
        config.set_default("float_eps", ConfigValue::Float(DEFAULT_FLOAT_EPS));
        config.set_default("num_threads", ConfigValue::UInt(DEFAULT_NUM_THREADS as u64));
        config.set_default(
            "max_iterations",
            ConfigValue::UInt(DEFAULT_MAX_ITERATIONS as u64),
        );
        config.set_default(
            "memory_limit",
            ConfigValue::UInt(DEFAULT_MEMORY_LIMIT as u64),
        );
        config.set_default("parallel_enabled", ConfigValue::Bool(true));
        config.set_default("debug_mode", ConfigValue::Bool(false));
        config.set_default("suppress_warnings", ConfigValue::Bool(false));

        // Load values from environment variables
        config.load_from_env();

        config
    }
}

impl Config {
    /// Create a new configuration with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a configuration value
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ConfigValue> {
        self.values.get(key)
    }

    /// Set a configuration value
    pub fn set(&mut self, key: &str, value: ConfigValue) {
        self.values.insert(key.to_string(), value);
    }

    /// Set a default configuration value (only if the key doesn't exist)
    fn set_default(&mut self, key: &str, value: ConfigValue) {
        self.values.entry(key.to_string()).or_insert(value);
    }

    /// Get a boolean configuration value
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ConfigError` if the key is not found or if the value is not a boolean.
    pub fn get_bool(&self, key: &str) -> CoreResult<bool> {
        match self.get(key) {
            Some(ConfigValue::Bool(b)) => Ok(*b),
            Some(value) => Err(CoreError::ConfigError(
                ErrorContext::new(format!(
                    "Expected boolean value for key '{key}', got: {value}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )),
            None => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Configuration key '{key}' not found"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }

    /// Get an integer configuration value
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ConfigError` if the key is not found or if the value is not an integer.
    pub fn get_int(&self, key: &str) -> CoreResult<i64> {
        match self.get(key) {
            Some(ConfigValue::Int(i)) => Ok(*i),
            Some(ConfigValue::UInt(u)) if *u <= i64::MAX as u64 => Ok(*u as i64),
            Some(value) => Err(CoreError::ConfigError(
                ErrorContext::new(format!(
                    "Expected integer value for key '{key}', got: {value}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )),
            None => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Configuration key '{key}' not found"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }

    /// Get an unsigned integer configuration value
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ConfigError` if the key is not found or if the value is not an unsigned integer.
    pub fn get_uint(&self, key: &str) -> CoreResult<u64> {
        match self.get(key) {
            Some(ConfigValue::UInt(u)) => Ok(*u),
            Some(ConfigValue::Int(i)) if *i >= 0 => Ok(*i as u64),
            Some(value) => Err(CoreError::ConfigError(
                ErrorContext::new(format!(
                    "Expected unsigned integer value for key '{key}', got: {value}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )),
            None => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Configuration key '{key}' not found"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }

    /// Get a floating-point configuration value
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ConfigError` if the key is not found or if the value cannot be converted to a float.
    pub fn get_float(&self, key: &str) -> CoreResult<f64> {
        match self.get(key) {
            Some(ConfigValue::Float(f)) => Ok(*f),
            Some(ConfigValue::Int(i)) => Ok(*i as f64),
            Some(ConfigValue::UInt(u)) => Ok(*u as f64),
            Some(value) => Err(CoreError::ConfigError(
                ErrorContext::new(format!(
                    "Expected float value for key '{key}', got: {value}"
                ))
                .with_location(ErrorLocation::new(file!(), line!())),
            )),
            None => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Configuration key '{key}' not found"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }

    /// Get a string configuration value
    ///
    /// # Errors
    ///
    /// Returns `CoreError::ConfigError` if the key is not found.
    pub fn get_string(&self, key: &str) -> CoreResult<String> {
        match self.get(key) {
            Some(ConfigValue::String(s)) => Ok(s.clone()),
            Some(value) => Ok(value.to_string()),
            None => Err(CoreError::ConfigError(
                ErrorContext::new(format!("Configuration key '{key}' not found"))
                    .with_location(ErrorLocation::new(file!(), line!())),
            )),
        }
    }

    /// Load configuration values from environment variables
    fn load_from_env(&mut self) {
        // Find all environment variables with the configured prefix
        for (key, value) in env::vars() {
            if key.starts_with(&self.env_prefix) {
                let config_key = key[self.env_prefix.len()..].to_lowercase();

                // Try to parse the value as different types
                if let Ok(bool_val) = value.parse::<bool>() {
                    self.set(&config_key, ConfigValue::Bool(bool_val));
                } else if let Ok(int_val) = value.parse::<i64>() {
                    self.set(&config_key, ConfigValue::Int(int_val));
                } else if let Ok(uint_val) = value.parse::<u64>() {
                    self.set(&config_key, ConfigValue::UInt(uint_val));
                } else if let Ok(float_val) = value.parse::<f64>() {
                    self.set(&config_key, ConfigValue::Float(float_val));
                } else {
                    self.set(&config_key, ConfigValue::String(value));
                }
            }
        }
    }
}

static GLOBAL_CONFIG: std::sync::LazyLock<RwLock<Config>> =
    std::sync::LazyLock::new(|| RwLock::new(Config::default()));

thread_local! {
    static THREAD_LOCAL_CONFIG: Arc<Mutex<Option<Config>>> = Arc::new(Mutex::new(None));
}

/// Get the current configuration
///
/// This function first checks for a thread-local configuration, and falls back to the global configuration.
#[must_use]
pub fn get_config() -> Config {
    // Try to get thread-local config first
    let thread_local = THREAD_LOCAL_CONFIG.with(|config| {
        let config_lock = config.lock().unwrap();
        config_lock.clone()
    });

    // If thread-local config exists, use it, otherwise use global config
    match thread_local {
        Some(config) => config,
        None => GLOBAL_CONFIG.read().unwrap().clone(),
    }
}

/// Set the global configuration
pub fn set_global_config(config: Config) {
    let mut global_config = GLOBAL_CONFIG.write().unwrap();
    *global_config = config;
}

/// Set a thread-local configuration
pub fn set_thread_local_config(config: Config) {
    THREAD_LOCAL_CONFIG.with(|thread_config| {
        let mut config_lock = thread_config.lock().unwrap();
        *config_lock = Some(config);
    });
}

/// Clear the thread-local configuration
pub fn clear_thread_local_config() {
    THREAD_LOCAL_CONFIG.with(|thread_config| {
        let mut config_lock = thread_config.lock().unwrap();
        *config_lock = None;
    });
}

/// Set a global configuration value
pub fn set_config_value(key: &str, value: ConfigValue) {
    let mut global_config = GLOBAL_CONFIG.write().unwrap();
    global_config.set(key, value);
}

/// Get a global configuration value
#[must_use]
pub fn get_config_value(key: &str) -> Option<ConfigValue> {
    let config = get_config();
    config.get(key).cloned()
}

/// Set a thread-local configuration value
pub fn set_thread_local_config_value(key: &str, value: ConfigValue) {
    THREAD_LOCAL_CONFIG.with(|thread_config| {
        let mut config_lock = thread_config.lock().unwrap();

        // Create a new config from global if it doesn't exist
        if config_lock.is_none() {
            let global_config = GLOBAL_CONFIG.read().unwrap().clone();
            *config_lock = Some(global_config);
        }

        // Now set the value
        if let Some(config) = config_lock.as_mut() {
            config.set(key, value);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();

        assert!(matches!(
            config.get("float_eps"),
            Some(ConfigValue::Float(_))
        ));
        assert!(matches!(
            config.get("num_threads"),
            Some(ConfigValue::UInt(_))
        ));
        assert!(matches!(
            config.get("max_iterations"),
            Some(ConfigValue::UInt(_))
        ));
        assert!(matches!(
            config.get("memory_limit"),
            Some(ConfigValue::UInt(_))
        ));
    }

    #[test]
    fn test_config_get_methods() {
        let mut config = Config::default();

        config.set("test_bool", ConfigValue::Bool(true));
        config.set("test_int", ConfigValue::Int(42));
        config.set("test_uint", ConfigValue::UInt(100));
        config.set("test_float", ConfigValue::Float(3.5));
        config.set("test_string", ConfigValue::String("hello".to_string()));

        assert!(config.get_bool("test_bool").unwrap());
        assert_eq!(config.get_int("test_int").unwrap(), 42);
        assert_eq!(config.get_uint("test_uint").unwrap(), 100);
        assert_eq!(config.get_float("test_float").unwrap(), 3.5);
        assert_eq!(config.get_string("test_string").unwrap(), "hello");

        // Test type conversions
        assert_eq!(config.get_float("test_int").unwrap(), 42.0);
        assert_eq!(config.get_int("test_uint").unwrap(), 100);

        // Test error cases
        assert!(config.get_bool("nonexistent").is_err());
        assert!(config.get_bool("test_int").is_err());
    }

    #[test]
    fn test_global_config() {
        let original = get_config();
        let mut new_config = Config::default();
        new_config.set("test_value", ConfigValue::String("global".to_string()));

        set_global_config(new_config);

        let config = get_config();
        assert_eq!(config.get_string("test_value").unwrap(), "global");

        // Restore original config
        set_global_config(original);

        // Clean up by removing the test key
        let mut final_config = GLOBAL_CONFIG.write().unwrap();
        final_config.values.remove("test_value");
    }

    #[test]
    fn test_thread_local_config() {
        let test_key = "test_thread_key";
        let original = get_config();

        {
            // First set a global value
            let mut global_config = GLOBAL_CONFIG.write().unwrap();
            global_config.set(test_key, ConfigValue::String("global".to_string()));
        }

        // Set a thread-local value
        let mut thread_config = Config::default();
        thread_config.set(test_key, ConfigValue::String("thread-local".to_string()));
        set_thread_local_config(thread_config);

        // Thread-local should take precedence
        let config = get_config();
        assert_eq!(config.get_string(test_key).unwrap(), "thread-local");

        // Clear thread-local config
        clear_thread_local_config();

        // Need to verify thread-local is gone
        let thread_result = THREAD_LOCAL_CONFIG.with(|config| {
            let locked = config.lock().unwrap();
            locked.is_none()
        });
        assert!(thread_result, "Thread-local config should be cleared");

        // Restore original config
        set_global_config(original);

        // Clean up by removing the test key
        let mut final_config = GLOBAL_CONFIG.write().unwrap();
        final_config.values.remove(test_key);
    }
}

/// Production-level configuration management with validation, hot reloading, and feature flags
pub mod production;
