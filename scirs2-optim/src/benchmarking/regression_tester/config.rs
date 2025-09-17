//! Configuration structures for regression testing framework
//!
//! This module provides all configuration-related types and structures for the
//! performance regression testing system, including main configuration,
//! alert settings, and test environment specifications.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for regression testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Baseline storage directory
    pub baseline_dir: PathBuf,
    /// Maximum history length to keep
    pub max_history_length: usize,
    /// Minimum samples required for baseline
    pub min_baseline_samples: usize,
    /// Statistical significance threshold
    pub significance_threshold: f64,
    /// Performance degradation threshold (percentage)
    pub degradation_threshold: f64,
    /// Memory regression threshold (percentage)
    pub memory_threshold: f64,
    /// Enable CI/CD integration
    pub enable_ci_integration: bool,
    /// Enable automated alerts
    pub enable_alerts: bool,
    /// Outlier detection sensitivity
    pub outlier_sensitivity: f64,
    /// Regression detection algorithms to use
    pub detection_algorithms: Vec<String>,
    /// Export format for CI reports
    pub ci_report_format: CiReportFormat,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            baseline_dir: PathBuf::from("performance_baselines"),
            max_history_length: 1000,
            min_baseline_samples: 10,
            significance_threshold: 0.05,
            degradation_threshold: 5.0, // 5% degradation threshold
            memory_threshold: 10.0,     // 10% memory increase threshold
            enable_ci_integration: true,
            enable_alerts: true,
            outlier_sensitivity: 2.0, // 2 standard deviations
            detection_algorithms: vec![
                "statistical_test".to_string(),
                "sliding_window".to_string(),
                "change_point".to_string(),
            ],
            ci_report_format: CiReportFormat::Json,
        }
    }
}

/// CI report output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CiReportFormat {
    /// JSON format for programmatic processing
    Json,
    /// JUnit XML format for CI systems
    JunitXml,
    /// Markdown format for human-readable reports
    Markdown,
    /// GitHub Actions format
    GitHubActions,
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable alerts globally
    pub enable_alerts: bool,
    /// Enable email alerts
    pub enable_email: bool,
    /// Enable Slack notifications
    pub enable_slack: bool,
    /// Enable GitHub issue creation
    pub enable_github_issues: bool,
    /// Alert severity threshold
    pub severity_threshold: f64,
    /// Cooldown period between alerts (minutes)
    pub cooldown_minutes: u64,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            enable_alerts: true,
            enable_email: false,
            enable_slack: false,
            enable_github_issues: false,
            severity_threshold: 0.05,
            cooldown_minutes: 60,
        }
    }
}

/// Test environment specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestEnvironment {
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu_model: String,
    /// Memory size (MB)
    pub memory_mb: usize,
    /// Rust version
    pub rust_version: String,
    /// Compiler flags
    pub compiler_flags: Vec<String>,
    /// Hardware acceleration available
    pub hardware_acceleration: Vec<String>,
}

impl Default for TestEnvironment {
    fn default() -> Self {
        Self {
            os: std::env::consts::OS.to_string(),
            cpu_model: "unknown".to_string(),
            memory_mb: 0,
            rust_version: "unknown".to_string(),
            compiler_flags: vec![],
            hardware_acceleration: vec![],
        }
    }
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low priority alert
    Low,
    /// Medium priority alert
    Medium,
    /// High priority alert
    High,
    /// Critical alert requiring immediate attention
    Critical,
}

/// Alert status tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert is active and needs attention
    Active,
    /// Alert has been acknowledged by someone
    Acknowledged,
    /// Alert has been resolved
    Resolved,
}

/// Alert notification structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert identifier
    pub id: String,
    /// Unix timestamp when alert was created
    pub timestamp: u64,
    /// Severity level of the alert
    pub severity: AlertSeverity,
    /// Human-readable alert message
    pub message: String,
    /// ID of the regression result that triggered this alert
    pub regression_id: String,
    /// Current status of the alert
    pub status: AlertStatus,
}

impl Alert {
    /// Create a new alert
    pub fn new(
        id: String,
        severity: AlertSeverity,
        message: String,
        regression_id: String,
    ) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};

        Self {
            id,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            severity,
            message,
            regression_id,
            status: AlertStatus::Active,
        }
    }

    /// Mark alert as acknowledged
    pub fn acknowledge(&mut self) {
        self.status = AlertStatus::Acknowledged;
    }

    /// Mark alert as resolved
    pub fn resolve(&mut self) {
        self.status = AlertStatus::Resolved;
    }

    /// Check if alert is still active
    pub fn is_active(&self) -> bool {
        matches!(self.status, AlertStatus::Active)
    }

    /// Get age of alert in seconds
    pub fn age_seconds(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};

        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(self.timestamp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_config_default() {
        let config = RegressionConfig::default();
        assert_eq!(config.degradation_threshold, 5.0);
        assert_eq!(config.memory_threshold, 10.0);
        assert!(config.enable_ci_integration);
        assert!(config.enable_alerts);
        assert_eq!(config.detection_algorithms.len(), 3);
    }

    #[test]
    fn test_alert_config_default() {
        let config = AlertConfig::default();
        assert!(config.enable_alerts);
        assert!(!config.enable_email);
        assert!(!config.enable_slack);
        assert!(!config.enable_github_issues);
        assert_eq!(config.cooldown_minutes, 60);
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::new(
            "test-001".to_string(),
            AlertSeverity::High,
            "Test regression detected".to_string(),
            "regression-123".to_string(),
        );

        assert_eq!(alert.id, "test-001");
        assert_eq!(alert.message, "Test regression detected");
        assert_eq!(alert.regression_id, "regression-123");
        assert!(alert.is_active());
        assert!(matches!(alert.severity, AlertSeverity::High));
    }

    #[test]
    fn test_alert_status_transitions() {
        let mut alert = Alert::new(
            "test-002".to_string(),
            AlertSeverity::Medium,
            "Another test".to_string(),
            "regression-456".to_string(),
        );

        assert!(alert.is_active());

        alert.acknowledge();
        assert!(!alert.is_active());
        assert!(matches!(alert.status, AlertStatus::Acknowledged));

        alert.resolve();
        assert!(!alert.is_active());
        assert!(matches!(alert.status, AlertStatus::Resolved));
    }

    #[test]
    fn test_alert_age_calculation() {
        let alert = Alert::new(
            "test-003".to_string(),
            AlertSeverity::Low,
            "Age test".to_string(),
            "regression-789".to_string(),
        );

        // Age should be very small (just created)
        assert!(alert.age_seconds() < 2);
    }

    #[test]
    fn test_test_environment_default() {
        let env = TestEnvironment::default();
        assert_eq!(env.os, std::env::consts::OS);
        assert_eq!(env.cpu_model, "unknown");
        assert_eq!(env.memory_mb, 0);
        assert!(env.compiler_flags.is_empty());
        assert!(env.hardware_acceleration.is_empty());
    }
}