//! Configuration structures for security auditing
//!
//! This module provides configuration management for the security auditing framework,
//! including audit settings, test parameters, and validation controls.

use std::time::Duration;

/// Configuration for security auditing
#[derive(Debug, Clone)]
pub struct SecurityAuditConfig {
    /// Enable input validation checks
    pub enable_input_validation: bool,
    /// Enable privacy guarantee analysis
    pub enable_privacy_analysis: bool,
    /// Enable memory safety checks
    pub enable_memory_safety: bool,
    /// Enable numerical stability analysis
    pub enable_numerical_analysis: bool,
    /// Enable access control verification
    pub enable_access_control: bool,
    /// Enable cryptographic security checks
    pub enable_crypto_analysis: bool,
    /// Maximum test iterations for vulnerability detection
    pub max_test_iterations: usize,
    /// Timeout for individual security tests
    pub test_timeout: Duration,
    /// Detailed logging of security events
    pub detailed_logging: bool,
    /// Generate security recommendations
    pub generate_recommendations: bool,
}

impl Default for SecurityAuditConfig {
    fn default() -> Self {
        Self {
            enable_input_validation: true,
            enable_privacy_analysis: true,
            enable_memory_safety: true,
            enable_numerical_analysis: true,
            enable_access_control: true,
            enable_crypto_analysis: true,
            max_test_iterations: 1000,
            test_timeout: Duration::from_secs(30),
            detailed_logging: true,
            generate_recommendations: true,
        }
    }
}

impl SecurityAuditConfig {
    /// Create a new security audit configuration with custom settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a lightweight configuration for basic security checks
    pub fn lightweight() -> Self {
        Self {
            enable_input_validation: true,
            enable_privacy_analysis: false,
            enable_memory_safety: true,
            enable_numerical_analysis: false,
            enable_access_control: false,
            enable_crypto_analysis: false,
            max_test_iterations: 100,
            test_timeout: Duration::from_secs(5),
            detailed_logging: false,
            generate_recommendations: false,
        }
    }

    /// Create a comprehensive configuration for thorough security analysis
    pub fn comprehensive() -> Self {
        Self {
            enable_input_validation: true,
            enable_privacy_analysis: true,
            enable_memory_safety: true,
            enable_numerical_analysis: true,
            enable_access_control: true,
            enable_crypto_analysis: true,
            max_test_iterations: 10000,
            test_timeout: Duration::from_secs(300),
            detailed_logging: true,
            generate_recommendations: true,
        }
    }

    /// Enable all security analysis modules
    pub fn enable_all(&mut self) -> &mut Self {
        self.enable_input_validation = true;
        self.enable_privacy_analysis = true;
        self.enable_memory_safety = true;
        self.enable_numerical_analysis = true;
        self.enable_access_control = true;
        self.enable_crypto_analysis = true;
        self
    }

    /// Disable all security analysis modules
    pub fn disable_all(&mut self) -> &mut Self {
        self.enable_input_validation = false;
        self.enable_privacy_analysis = false;
        self.enable_memory_safety = false;
        self.enable_numerical_analysis = false;
        self.enable_access_control = false;
        self.enable_crypto_analysis = false;
        self
    }

    /// Configure for development environment (faster, less comprehensive)
    pub fn development(&mut self) -> &mut Self {
        self.max_test_iterations = 50;
        self.test_timeout = Duration::from_secs(1);
        self.detailed_logging = false;
        self
    }

    /// Configure for production environment (comprehensive, detailed)
    pub fn production(&mut self) -> &mut Self {
        self.max_test_iterations = 5000;
        self.test_timeout = Duration::from_secs(60);
        self.detailed_logging = true;
        self.generate_recommendations = true;
        self
    }

    /// Validate the configuration settings
    pub fn validate(&self) -> Result<(), String> {
        if self.max_test_iterations == 0 {
            return Err("max_test_iterations must be greater than 0".to_string());
        }

        if self.test_timeout.is_zero() {
            return Err("test_timeout must be greater than 0".to_string());
        }

        if !self.any_analysis_enabled() {
            return Err("At least one security analysis module must be enabled".to_string());
        }

        Ok(())
    }

    /// Check if any security analysis is enabled
    pub fn any_analysis_enabled(&self) -> bool {
        self.enable_input_validation
            || self.enable_privacy_analysis
            || self.enable_memory_safety
            || self.enable_numerical_analysis
            || self.enable_access_control
            || self.enable_crypto_analysis
    }

    /// Get the estimated analysis time based on configuration
    pub fn estimated_analysis_time(&self) -> Duration {
        let mut base_time = Duration::from_secs(1);

        if self.enable_input_validation {
            base_time += Duration::from_secs(5);
        }
        if self.enable_privacy_analysis {
            base_time += Duration::from_secs(10);
        }
        if self.enable_memory_safety {
            base_time += Duration::from_secs(3);
        }
        if self.enable_numerical_analysis {
            base_time += Duration::from_secs(8);
        }
        if self.enable_access_control {
            base_time += Duration::from_secs(2);
        }
        if self.enable_crypto_analysis {
            base_time += Duration::from_secs(15);
        }

        // Scale based on test iterations
        let iteration_factor = (self.max_test_iterations as f64 / 1000.0).max(0.1);
        Duration::from_secs((base_time.as_secs() as f64 * iteration_factor) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SecurityAuditConfig::default();
        assert!(config.enable_input_validation);
        assert!(config.enable_privacy_analysis);
        assert!(config.any_analysis_enabled());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_lightweight_config() {
        let config = SecurityAuditConfig::lightweight();
        assert!(config.enable_input_validation);
        assert!(!config.enable_privacy_analysis);
        assert_eq!(config.max_test_iterations, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_comprehensive_config() {
        let config = SecurityAuditConfig::comprehensive();
        assert!(config.enable_input_validation);
        assert!(config.enable_privacy_analysis);
        assert!(config.enable_crypto_analysis);
        assert_eq!(config.max_test_iterations, 10000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = SecurityAuditConfig::default();
        config.max_test_iterations = 0;
        assert!(config.validate().is_err());

        config.max_test_iterations = 100;
        config.test_timeout = Duration::from_secs(0);
        assert!(config.validate().is_err());

        config.test_timeout = Duration::from_secs(1);
        config.disable_all();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_estimated_analysis_time() {
        let lightweight = SecurityAuditConfig::lightweight();
        let comprehensive = SecurityAuditConfig::comprehensive();

        assert!(comprehensive.estimated_analysis_time() > lightweight.estimated_analysis_time());
    }
}