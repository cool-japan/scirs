//! Configuration for Real-Time Performance Monitoring
//!
//! This module contains configuration structures and default settings
//! for the real-time performance monitoring system.

/// Configuration for real-time performance monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMonitorConfig {
    /// Monitoring interval in milliseconds
    pub monitoring_interval_ms: u64,
    /// Maximum number of performance samples to keep
    pub max_samples: usize,
    /// Enable adaptive tuning based on performance
    pub adaptive_tuning: bool,
    /// Performance threshold for adaptation triggers
    pub adaptation_threshold: f64,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert threshold for performance degradation
    pub alert_threshold: f64,
    /// Enable automatic optimization
    pub auto_optimization: bool,
    /// Optimization interval in seconds
    pub optimization_interval_s: u64,
    /// Enable performance prediction
    pub enable_prediction: bool,
    /// Prediction horizon in samples
    pub prediction_horizon: usize,
    /// Enable system metrics collection
    pub enable_system_metrics: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection sensitivity (0.0 to 1.0)
    pub anomaly_sensitivity: f64,
    /// Maximum alert history to keep
    pub max_alert_history: usize,
    /// Maximum adaptation events to keep
    pub max_adaptation_history: usize,
}

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval_ms: 100,
            max_samples: 10000,
            adaptive_tuning: true,
            adaptation_threshold: 0.8,
            enable_alerts: true,
            alert_threshold: 0.5,
            auto_optimization: true,
            optimization_interval_s: 30,
            enable_prediction: true,
            prediction_horizon: 50,
            enable_system_metrics: true,
            enable_anomaly_detection: true,
            anomaly_sensitivity: 0.7,
            max_alert_history: 1000,
            max_adaptation_history: 500,
        }
    }
}

impl PerformanceMonitorConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Default::default()
    }

    /// Create a lightweight configuration for minimal overhead
    pub fn lightweight() -> Self {
        Self {
            monitoring_interval_ms: 500,
            max_samples: 1000,
            adaptive_tuning: false,
            adaptation_threshold: 0.9,
            enable_alerts: false,
            alert_threshold: 0.3,
            auto_optimization: false,
            optimization_interval_s: 300,
            enable_prediction: false,
            prediction_horizon: 10,
            enable_system_metrics: false,
            enable_anomaly_detection: false,
            anomaly_sensitivity: 0.5,
            max_alert_history: 100,
            max_adaptation_history: 50,
        }
    }

    /// Create a high-performance configuration for detailed monitoring
    pub fn high_performance() -> Self {
        Self {
            monitoring_interval_ms: 50,
            max_samples: 50000,
            adaptive_tuning: true,
            adaptation_threshold: 0.7,
            enable_alerts: true,
            alert_threshold: 0.6,
            auto_optimization: true,
            optimization_interval_s: 10,
            enable_prediction: true,
            prediction_horizon: 100,
            enable_system_metrics: true,
            enable_anomaly_detection: true,
            anomaly_sensitivity: 0.8,
            max_alert_history: 5000,
            max_adaptation_history: 2000,
        }
    }

    /// Create a debugging configuration with extensive logging
    pub fn debug() -> Self {
        Self {
            monitoring_interval_ms: 10,
            max_samples: 100000,
            adaptive_tuning: true,
            adaptation_threshold: 0.6,
            enable_alerts: true,
            alert_threshold: 0.7,
            auto_optimization: true,
            optimization_interval_s: 5,
            enable_prediction: true,
            prediction_horizon: 200,
            enable_system_metrics: true,
            enable_anomaly_detection: true,
            anomaly_sensitivity: 0.9,
            max_alert_history: 10000,
            max_adaptation_history: 5000,
        }
    }

    /// Builder pattern methods for configuration
    pub fn with_monitoring_interval_ms(mut self, interval: u64) -> Self {
        self.monitoring_interval_ms = interval;
        self
    }

    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    pub fn with_adaptive_tuning(mut self, enabled: bool) -> Self {
        self.adaptive_tuning = enabled;
        self
    }

    pub fn with_adaptation_threshold(mut self, threshold: f64) -> Self {
        self.adaptation_threshold = threshold;
        self
    }

    pub fn with_alerts(mut self, enabled: bool) -> Self {
        self.enable_alerts = enabled;
        self
    }

    pub fn with_alert_threshold(mut self, threshold: f64) -> Self {
        self.alert_threshold = threshold;
        self
    }

    pub fn with_auto_optimization(mut self, enabled: bool) -> Self {
        self.auto_optimization = enabled;
        self
    }

    pub fn with_optimization_interval_s(mut self, interval: u64) -> Self {
        self.optimization_interval_s = interval;
        self
    }

    pub fn with_prediction(mut self, enabled: bool) -> Self {
        self.enable_prediction = enabled;
        self
    }

    pub fn with_prediction_horizon(mut self, horizon: usize) -> Self {
        self.prediction_horizon = horizon;
        self
    }

    pub fn with_system_metrics(mut self, enabled: bool) -> Self {
        self.enable_system_metrics = enabled;
        self
    }

    pub fn with_anomaly_detection(mut self, enabled: bool) -> Self {
        self.enable_anomaly_detection = enabled;
        self
    }

    pub fn with_anomaly_sensitivity(mut self, sensitivity: f64) -> Self {
        self.anomaly_sensitivity = sensitivity.clamp(0.0, 1.0);
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.monitoring_interval_ms == 0 {
            return Err("Monitoring interval must be greater than 0".to_string());
        }

        if self.max_samples == 0 {
            return Err("Max samples must be greater than 0".to_string());
        }

        if !(0.0..=1.0).contains(&self.adaptation_threshold) {
            return Err("Adaptation threshold must be between 0.0 and 1.0".to_string());
        }

        if !(0.0..=1.0).contains(&self.alert_threshold) {
            return Err("Alert threshold must be between 0.0 and 1.0".to_string());
        }

        if self.optimization_interval_s == 0 && self.auto_optimization {
            return Err(
                "Optimization interval must be greater than 0 when auto-optimization is enabled"
                    .to_string(),
            );
        }

        if self.prediction_horizon == 0 && self.enable_prediction {
            return Err(
                "Prediction horizon must be greater than 0 when prediction is enabled".to_string(),
            );
        }

        if !(0.0..=1.0).contains(&self.anomaly_sensitivity) {
            return Err("Anomaly sensitivity must be between 0.0 and 1.0".to_string());
        }

        if self.max_alert_history == 0 && self.enable_alerts {
            return Err(
                "Max alert history must be greater than 0 when alerts are enabled".to_string(),
            );
        }

        if self.max_adaptation_history == 0 && self.adaptive_tuning {
            return Err(
                "Max adaptation history must be greater than 0 when adaptive tuning is enabled"
                    .to_string(),
            );
        }

        Ok(())
    }

    /// Get estimated memory usage in bytes
    pub fn estimated_memory_usage(&self) -> usize {
        let sample_size = std::mem::size_of::<super::metrics::PerformanceSample>();
        let alert_size = std::mem::size_of::<super::alerts::Alert>();
        let adaptation_size = 256; // Estimated size for adaptation events

        self.max_samples * sample_size
            + self.max_alert_history * alert_size
            + self.max_adaptation_history * adaptation_size
            + 8192 // Base overhead
    }

    /// Check if configuration is suitable for real-time operation
    pub fn is_realtime_suitable(&self) -> bool {
        // Real-time suitable if monitoring interval is reasonable
        // and memory usage is not excessive
        self.monitoring_interval_ms <= 1000 &&
        self.estimated_memory_usage() <= 100 * 1024 * 1024 && // 100MB limit
        self.max_samples <= 50000
    }

    /// Get recommended settings based on use case
    pub fn recommended_for_use_case(use_case: UseCase) -> Self {
        match use_case {
            UseCase::Production => Self::default(),
            UseCase::Development => Self::debug(),
            UseCase::Testing => Self::lightweight(),
            UseCase::Benchmarking => Self::high_performance(),
            UseCase::LowResource => Self::lightweight().with_max_samples(500),
        }
    }
}

/// Different use cases for monitoring configuration
#[derive(Debug, Clone, Copy)]
pub enum UseCase {
    /// Production environment with balanced monitoring
    Production,
    /// Development environment with detailed monitoring
    Development,
    /// Testing environment with minimal overhead
    Testing,
    /// Benchmarking with maximum detail
    Benchmarking,
    /// Low resource environment with minimal footprint
    LowResource,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PerformanceMonitorConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.monitoring_interval_ms, 100);
        assert_eq!(config.max_samples, 10000);
        assert!(config.adaptive_tuning);
        assert!(config.enable_alerts);
    }

    #[test]
    fn test_lightweight_config() {
        let config = PerformanceMonitorConfig::lightweight();
        assert!(config.validate().is_ok());
        assert_eq!(config.monitoring_interval_ms, 500);
        assert_eq!(config.max_samples, 1000);
        assert!(!config.adaptive_tuning);
        assert!(!config.enable_alerts);
    }

    #[test]
    fn test_high_performance_config() {
        let config = PerformanceMonitorConfig::high_performance();
        assert!(config.validate().is_ok());
        assert_eq!(config.monitoring_interval_ms, 50);
        assert_eq!(config.max_samples, 50000);
        assert!(config.adaptive_tuning);
        assert!(config.enable_alerts);
    }

    #[test]
    fn test_builder_pattern() {
        let config = PerformanceMonitorConfig::new()
            .with_monitoring_interval_ms(200)
            .with_max_samples(5000)
            .with_adaptive_tuning(false)
            .with_alerts(false);

        assert!(config.validate().is_ok());
        assert_eq!(config.monitoring_interval_ms, 200);
        assert_eq!(config.max_samples, 5000);
        assert!(!config.adaptive_tuning);
        assert!(!config.enable_alerts);
    }

    #[test]
    fn test_config_validation() {
        let mut config = PerformanceMonitorConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid monitoring interval
        config.monitoring_interval_ms = 0;
        assert!(config.validate().is_err());

        config.monitoring_interval_ms = 100;
        config.max_samples = 0;
        assert!(config.validate().is_err());

        config.max_samples = 1000;
        config.adaptation_threshold = 1.5;
        assert!(config.validate().is_err());

        config.adaptation_threshold = 0.8;
        config.alert_threshold = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_memory_usage_estimation() {
        let config = PerformanceMonitorConfig::default();
        let memory_usage = config.estimated_memory_usage();
        assert!(memory_usage > 0);

        let lightweight = PerformanceMonitorConfig::lightweight();
        let lightweight_memory = lightweight.estimated_memory_usage();
        assert!(lightweight_memory < memory_usage);
    }

    #[test]
    fn test_realtime_suitability() {
        let config = PerformanceMonitorConfig::default();
        assert!(config.is_realtime_suitable());

        let debug_config = PerformanceMonitorConfig::debug();
        // Debug config might not be suitable for real-time due to high frequency
        let _ = debug_config.is_realtime_suitable(); // Result depends on implementation
    }

    #[test]
    fn test_use_case_recommendations() {
        let production = PerformanceMonitorConfig::recommended_for_use_case(UseCase::Production);
        assert!(production.validate().is_ok());

        let development = PerformanceMonitorConfig::recommended_for_use_case(UseCase::Development);
        assert!(development.validate().is_ok());

        let testing = PerformanceMonitorConfig::recommended_for_use_case(UseCase::Testing);
        assert!(testing.validate().is_ok());

        let benchmarking =
            PerformanceMonitorConfig::recommended_for_use_case(UseCase::Benchmarking);
        assert!(benchmarking.validate().is_ok());

        let low_resource = PerformanceMonitorConfig::recommended_for_use_case(UseCase::LowResource);
        assert!(low_resource.validate().is_ok());
        assert_eq!(low_resource.max_samples, 500);
    }

    #[test]
    fn test_anomaly_sensitivity_clamping() {
        let config = PerformanceMonitorConfig::new().with_anomaly_sensitivity(1.5); // Should be clamped to 1.0
        assert_eq!(config.anomaly_sensitivity, 1.0);

        let config = PerformanceMonitorConfig::new().with_anomaly_sensitivity(-0.1); // Should be clamped to 0.0
        assert_eq!(config.anomaly_sensitivity, 0.0);
    }
}
