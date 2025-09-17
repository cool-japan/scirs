//! Alert system for performance regression notifications
//!
//! This module provides a comprehensive alerting system that can notify stakeholders
//! through multiple channels (email, Slack, GitHub issues) when performance
//! regressions are detected.

use crate::benchmarking::regression_tester::config::{Alert, AlertConfig, AlertSeverity, AlertStatus};
use crate::benchmarking::regression_tester::types::RegressionResult;
use crate::error::Result;
use num_traits::Float;
use std::collections::VecDeque;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Alert system for regression notifications
///
/// Manages alert generation, notification delivery, cooldown periods,
/// and integration with external services like email, Slack, and GitHub.
#[derive(Debug)]
pub struct AlertSystem {
    /// Alert configuration
    config: AlertConfig,
    /// Alert history for tracking and cooldown management
    alert_history: VecDeque<Alert>,
}

impl AlertSystem {
    /// Create a new alert system with default configuration
    pub fn new() -> Self {
        Self {
            config: AlertConfig::default(),
            alert_history: VecDeque::new(),
        }
    }

    /// Create a new alert system with custom configuration
    pub fn with_config(config: AlertConfig) -> Self {
        Self {
            config,
            alert_history: VecDeque::new(),
        }
    }

    /// Get the current alert configuration
    pub fn config(&self) -> &AlertConfig {
        &self.config
    }

    /// Update the alert configuration
    pub fn update_config(&mut self, config: AlertConfig) {
        self.config = config;
    }

    /// Get alert history
    pub fn alert_history(&self) -> &VecDeque<Alert> {
        &self.alert_history
    }

    /// Send an alert for a regression
    pub fn send_alert<A: Float>(&mut self, regression: &RegressionResult<A>) -> Result<()> {
        if regression.severity < self.config.severity_threshold {
            return Ok(()); // Below threshold
        }

        let alert = Alert::new(
            format!(
                "alert_{}",
                SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs()
            ),
            self.map_severity(regression.severity),
            format!(
                "Performance regression detected in {}: {:.2}% degradation",
                regression.test_id, regression.performance_change_percent
            ),
            regression.test_id.clone(),
        );

        self.alert_history.push_back(alert.clone());

        // Maintain alert history size
        if self.alert_history.len() > 100 {
            self.alert_history.pop_front();
        }

        // Send actual alerts through configured channels
        self.send_alert_notifications(&alert)?;

        Ok(())
    }

    /// Map numeric severity to AlertSeverity enum
    fn map_severity(&self, severity: f64) -> AlertSeverity {
        match severity {
            s if s >= 0.8 => AlertSeverity::Critical,
            s if s >= 0.6 => AlertSeverity::High,
            s if s >= 0.3 => AlertSeverity::Medium,
            _ => AlertSeverity::Low,
        }
    }

    /// Send alert notifications through configured channels
    fn send_alert_notifications(&self, alert: &Alert) -> Result<()> {
        // Check if alerts are enabled and severity meets threshold
        if !self.config.enable_alerts || self.severity_below_threshold(alert) {
            return Ok(());
        }

        // Check cooldown period
        if self.is_in_cooldown_period(alert)? {
            return Ok(());
        }

        let mut notification_results = Vec::new();

        // Send email notifications
        if self.config.enable_email {
            match self.send_email_notification(alert) {
                Ok(()) => notification_results.push("Email sent successfully".to_string()),
                Err(e) => notification_results.push(format!("Email failed: {}", e)),
            }
        }

        // Send Slack notifications
        if self.config.enable_slack {
            match self.send_slack_notification(alert) {
                Ok(()) => {
                    notification_results.push("Slack notification sent successfully".to_string())
                }
                Err(e) => notification_results.push(format!("Slack notification failed: {}", e)),
            }
        }

        // Create GitHub issues
        if self.config.enable_github_issues {
            match self.create_github_issue(alert) {
                Ok(()) => {
                    notification_results.push("GitHub issue created successfully".to_string())
                }
                Err(e) => notification_results.push(format!("GitHub issue creation failed: {}", e)),
            }
        }

        // Log notification results
        for result in notification_results {
            eprintln!("Alert notification: {}", result);
        }

        Ok(())
    }

    /// Check if alert severity is below configured threshold
    fn severity_below_threshold(&self, alert: &Alert) -> bool {
        let alert_severity_value = match alert.severity {
            AlertSeverity::Critical => 1.0,
            AlertSeverity::High => 0.75,
            AlertSeverity::Medium => 0.5,
            AlertSeverity::Low => 0.25,
        };
        alert_severity_value < self.config.severity_threshold
    }

    /// Check if we're in cooldown period for similar alerts
    fn is_in_cooldown_period(&self, alert: &Alert) -> Result<bool> {
        let cooldown_duration = Duration::from_secs(self.config.cooldown_minutes * 60);
        let current_time = SystemTime::now();

        // Check for similar recent alerts
        for recent_alert in self.alert_history.iter().rev().take(10) {
            if recent_alert.regression_id == alert.regression_id {
                let recent_time = UNIX_EPOCH + Duration::from_secs(recent_alert.timestamp);
                if current_time.duration_since(recent_time)? < cooldown_duration {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    /// Send email notification
    fn send_email_notification(&self, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use an email service like:
        // - SMTP with lettre crate
        // - AWS SES
        // - SendGrid
        // - Mailgun

        let email_body = self.format_email_body(alert);
        let subject = format!("Performance Regression Alert: {}", alert.regression_id);

        // Placeholder implementation - would integrate with actual email service
        eprintln!("EMAIL ALERT:");
        eprintln!("To: performance-team@company.com");
        eprintln!("Subject: {}", subject);
        eprintln!("Body:\n{}", email_body);
        eprintln!("---");

        // TODO: Integrate with actual email service
        // Example with lettre crate:
        // let email = Message::builder()
        //     .from("alerts@company.com".parse()?)
        //     .to("performance-team@company.com".parse()?)
        //     .subject(&subject)
        //     .body(email_body)?;
        // let mailer = SmtpTransport::relay("smtp.company.com")?.build();
        // mailer.send(&email)?;

        Ok(())
    }

    /// Send Slack notification
    fn send_slack_notification(&self, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use:
        // - Slack webhook URL
        // - reqwest crate for HTTP requests
        // - JSON payload formatting

        let slack_message = self.format_slack_message(alert);

        // Placeholder implementation - would make HTTP POST to Slack webhook
        eprintln!("SLACK ALERT:");
        eprintln!("Channel: #performance-alerts");
        eprintln!("Message: {}", slack_message);
        eprintln!("---");

        // TODO: Integrate with actual Slack API
        // Example:
        // let webhook_url = std::env::var("SLACK_WEBHOOK_URL")?;
        // let payload = json!({
        //     "text": slack_message,
        //     "channel": "#performance-alerts",
        //     "username": "Performance Bot"
        // });
        // let client = reqwest::Client::new();
        // client.post(&webhook_url).json(&payload).send()?;

        Ok(())
    }

    /// Create GitHub issue
    fn create_github_issue(&self, alert: &Alert) -> Result<()> {
        // In a real implementation, this would use:
        // - GitHub API with octocrab crate
        // - Personal access token
        // - Repository configuration

        let issue_title = format!("Performance regression in {}", alert.regression_id);
        let issue_body = self.format_github_issue_body(alert);

        // Placeholder implementation - would create actual GitHub issue
        eprintln!("GITHUB ISSUE:");
        eprintln!("Repository: company/performance-monitoring");
        eprintln!("Title: {}", issue_title);
        eprintln!("Body:\n{}", issue_body);
        eprintln!("Labels: performance, regression, automated");
        eprintln!("---");

        // TODO: Integrate with actual GitHub API
        // Example with octocrab:
        // let token = std::env::var("GITHUB_TOKEN")?;
        // let octocrab = octocrab::Octocrab::builder().personal_token(token).build()?;
        // octocrab.issues("company", "performance-monitoring")
        //     .create(&issue_title)
        //     .body(&issue_body)
        //     .labels(vec!["performance", "regression", "automated"])
        //     .send().await?;

        Ok(())
    }

    /// Format email body for alert
    fn format_email_body(&self, alert: &Alert) -> String {
        format!(
            "Performance Regression Alert\n\
            =============================\n\n\
            Alert ID: {}\n\
            Timestamp: {}\n\
            Severity: {:?}\n\
            Test: {}\n\n\
            Details:\n\
            {}\n\n\
            Please investigate this performance regression immediately.\n\
            \n\
            View full details at: https://performance-dashboard.company.com/alerts/{}\n\
            \n\
            Best regards,\n\
            Performance Monitoring System",
            alert.id, alert.timestamp, alert.severity, alert.regression_id, alert.message, alert.id
        )
    }

    /// Format Slack message for alert
    fn format_slack_message(&self, alert: &Alert) -> String {
        let severity_emoji = match alert.severity {
            AlertSeverity::Critical => "🚨",
            AlertSeverity::High => "⚠️",
            AlertSeverity::Medium => "🟡",
            AlertSeverity::Low => "🔵",
        };

        format!(
            "{} *Performance Regression Alert*\n\
            *Test:* {}\n\
            *Severity:* {:?}\n\
            *Details:* {}\n\
            *Time:* <t:{}:F>\n\
            <https://performance-dashboard.company.com/alerts/{}|View Details>",
            severity_emoji,
            alert.regression_id,
            alert.severity,
            alert.message,
            alert.timestamp,
            alert.id
        )
    }

    /// Format GitHub issue body for alert
    fn format_github_issue_body(&self, alert: &Alert) -> String {
        format!(
            "## Performance Regression Detected\n\n\
            **Alert ID:** {}\n\
            **Timestamp:** {}\n\
            **Severity:** {:?}\n\
            **Test:** {}\n\n\
            ### Description\n\
            {}\n\n\
            ### Investigation Steps\n\
            - [ ] Review recent code changes that might affect performance\n\
            - [ ] Check system resource utilization during test execution\n\
            - [ ] Run additional test iterations to confirm regression\n\
            - [ ] Analyze profiling data for performance bottlenecks\n\
            - [ ] Compare with baseline performance metrics\n\n\
            ### Links\n\
            - [Performance Dashboard](https://performance-dashboard.company.com/alerts/{})\n\
            - [Test Results](https://ci.company.com/tests/{})\n\n\
            ---\n\
            *This issue was automatically created by the performance monitoring system.*",
            alert.id,
            alert.timestamp,
            alert.severity,
            alert.regression_id,
            alert.message,
            alert.id,
            alert.regression_id
        )
    }

    /// Get active alerts (not acknowledged or resolved)
    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.alert_history
            .iter()
            .filter(|alert| alert.is_active())
            .collect()
    }

    /// Get recent alerts within the specified duration
    pub fn get_recent_alerts(&self, duration: Duration) -> Vec<&Alert> {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(duration.as_secs());

        self.alert_history
            .iter()
            .filter(|alert| alert.timestamp >= cutoff_time)
            .collect()
    }

    /// Acknowledge an alert by ID
    pub fn acknowledge_alert(&mut self, alert_id: &str) -> Result<()> {
        for alert in &mut self.alert_history {
            if alert.id == alert_id {
                alert.acknowledge();
                return Ok(());
            }
        }
        Err(crate::error::OptimError::InvalidParameter(format!(
            "Alert with ID {} not found",
            alert_id
        )))
    }

    /// Resolve an alert by ID
    pub fn resolve_alert(&mut self, alert_id: &str) -> Result<()> {
        for alert in &mut self.alert_history {
            if alert.id == alert_id {
                alert.resolve();
                return Ok(());
            }
        }
        Err(crate::error::OptimError::InvalidParameter(format!(
            "Alert with ID {} not found",
            alert_id
        )))
    }

    /// Clear old alerts from history
    pub fn cleanup_old_alerts(&mut self, max_age: Duration) -> usize {
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
            .saturating_sub(max_age.as_secs());

        let original_len = self.alert_history.len();
        self.alert_history.retain(|alert| alert.timestamp >= cutoff_time);
        original_len - self.alert_history.len()
    }

    /// Get statistics about alerts
    pub fn get_alert_statistics(&self) -> AlertStatistics {
        let total_alerts = self.alert_history.len();
        let active_alerts = self.get_active_alerts().len();

        let severity_counts = self.alert_history.iter().fold(
            SeverityCounts::default(),
            |mut counts, alert| {
                match alert.severity {
                    AlertSeverity::Critical => counts.critical += 1,
                    AlertSeverity::High => counts.high += 1,
                    AlertSeverity::Medium => counts.medium += 1,
                    AlertSeverity::Low => counts.low += 1,
                }
                counts
            },
        );

        AlertStatistics {
            total_alerts,
            active_alerts,
            severity_counts,
        }
    }
}

impl Default for AlertSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Alert statistics
#[derive(Debug, Clone)]
pub struct AlertStatistics {
    /// Total number of alerts in history
    pub total_alerts: usize,
    /// Number of active (unresolved) alerts
    pub active_alerts: usize,
    /// Count of alerts by severity level
    pub severity_counts: SeverityCounts,
}

/// Count of alerts by severity level
#[derive(Debug, Clone, Default)]
pub struct SeverityCounts {
    /// Number of critical alerts
    pub critical: usize,
    /// Number of high severity alerts
    pub high: usize,
    /// Number of medium severity alerts
    pub medium: usize,
    /// Number of low severity alerts
    pub low: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarking::regression_tester::types::{
        RegressionAnalysis, StatisticalTestResult, TrendAnalysis, TrendDirection,
        ChangePointAnalysis, OutlierAnalysis,
    };

    fn create_test_regression(severity: f64, test_id: &str) -> RegressionResult<f64> {
        RegressionResult {
            test_id: test_id.to_string(),
            regression_detected: true,
            severity,
            confidence: 0.95,
            performance_change_percent: 15.0,
            memory_change_percent: 5.0,
            affected_metrics: vec!["timing".to_string()],
            statistical_tests: vec![],
            analysis: RegressionAnalysis {
                trend_analysis: TrendAnalysis {
                    direction: TrendDirection::Degrading,
                    magnitude: 15.0,
                    significance: 0.95,
                    start_point: None,
                },
                change_point_analysis: ChangePointAnalysis {
                    change_points: vec![],
                    magnitudes: vec![],
                    confidences: vec![],
                },
                outlier_analysis: OutlierAnalysis {
                    outlier_indices: vec![],
                    outlier_scores: vec![],
                    outlier_types: vec![],
                },
                root_cause_hints: vec![],
            },
            recommendations: vec![],
        }
    }

    #[test]
    fn test_alert_system_creation() {
        let alert_system = AlertSystem::new();
        assert!(alert_system.config().enable_alerts);
        assert_eq!(alert_system.alert_history().len(), 0);
    }

    #[test]
    fn test_send_alert_above_threshold() {
        let mut alert_system = AlertSystem::new();
        let regression = create_test_regression(0.8, "test_high_severity");

        let result = alert_system.send_alert(&regression);
        assert!(result.is_ok());
        assert_eq!(alert_system.alert_history().len(), 1);

        let alert = &alert_system.alert_history()[0];
        assert!(matches!(alert.severity, AlertSeverity::Critical));
        assert_eq!(alert.regression_id, "test_high_severity");
    }

    #[test]
    fn test_send_alert_below_threshold() {
        let mut alert_system = AlertSystem::new();
        let regression = create_test_regression(0.1, "test_low_severity");

        let result = alert_system.send_alert(&regression);
        assert!(result.is_ok());
        assert_eq!(alert_system.alert_history().len(), 0); // Below threshold
    }

    #[test]
    fn test_severity_mapping() {
        let alert_system = AlertSystem::new();

        assert!(matches!(
            alert_system.map_severity(0.9),
            AlertSeverity::Critical
        ));
        assert!(matches!(alert_system.map_severity(0.7), AlertSeverity::High));
        assert!(matches!(
            alert_system.map_severity(0.4),
            AlertSeverity::Medium
        ));
        assert!(matches!(alert_system.map_severity(0.1), AlertSeverity::Low));
    }

    #[test]
    fn test_alert_history_limit() {
        let mut alert_system = AlertSystem::new();

        // Add more than 100 alerts
        for i in 0..105 {
            let regression = create_test_regression(0.8, &format!("test_{}", i));
            let _ = alert_system.send_alert(&regression);
        }

        // Should maintain limit of 100
        assert_eq!(alert_system.alert_history().len(), 100);
    }

    #[test]
    fn test_custom_config() {
        let custom_config = AlertConfig {
            enable_alerts: true,
            enable_email: true,
            enable_slack: true,
            enable_github_issues: false,
            severity_threshold: 0.8,
            cooldown_minutes: 30,
        };

        let alert_system = AlertSystem::with_config(custom_config.clone());
        assert_eq!(alert_system.config().severity_threshold, 0.8);
        assert_eq!(alert_system.config().cooldown_minutes, 30);
        assert!(alert_system.config().enable_email);
        assert!(alert_system.config().enable_slack);
        assert!(!alert_system.config().enable_github_issues);
    }

    #[test]
    fn test_alert_statistics() {
        let mut alert_system = AlertSystem::new();

        // Add alerts with different severities
        let _ = alert_system.send_alert(&create_test_regression(0.9, "critical"));
        let _ = alert_system.send_alert(&create_test_regression(0.7, "high"));
        let _ = alert_system.send_alert(&create_test_regression(0.4, "medium"));
        let _ = alert_system.send_alert(&create_test_regression(0.2, "low"));

        let stats = alert_system.get_alert_statistics();
        assert_eq!(stats.total_alerts, 4);
        assert_eq!(stats.active_alerts, 4);
        assert_eq!(stats.severity_counts.critical, 1);
        assert_eq!(stats.severity_counts.high, 1);
        assert_eq!(stats.severity_counts.medium, 1);
        assert_eq!(stats.severity_counts.low, 1);
    }

    #[test]
    fn test_alert_acknowledgment() {
        let mut alert_system = AlertSystem::new();
        let regression = create_test_regression(0.8, "test_ack");

        let _ = alert_system.send_alert(&regression);
        let alert_id = alert_system.alert_history()[0].id.clone();

        let result = alert_system.acknowledge_alert(&alert_id);
        assert!(result.is_ok());

        let alert = &alert_system.alert_history()[0];
        assert!(!alert.is_active());
        assert!(matches!(alert.status, AlertStatus::Acknowledged));
    }

    #[test]
    fn test_cleanup_old_alerts() {
        let mut alert_system = AlertSystem::new();

        // Add some alerts
        for i in 0..5 {
            let regression = create_test_regression(0.8, &format!("test_{}", i));
            let _ = alert_system.send_alert(&regression);
        }

        assert_eq!(alert_system.alert_history().len(), 5);

        // Cleanup alerts older than 0 seconds (should remove all)
        let removed = alert_system.cleanup_old_alerts(Duration::from_secs(0));
        assert_eq!(removed, 5);
        assert_eq!(alert_system.alert_history().len(), 0);
    }
}