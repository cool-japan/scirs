//! Alert Management System
//!
//! This module provides a comprehensive alerting system for performance
//! monitoring, including rule-based alerts and notifications.

use super::metrics::{PerformanceSample, ProcessorType};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Performance alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub timestamp: u64,
    pub severity: AlertSeverity,
    pub message: String,
    pub processor_type: ProcessorType,
    pub processor_id: String,
    pub metric_name: String,
    pub threshold_value: f64,
    pub actual_value: f64,
    pub resolved: bool,
    pub resolution_timestamp: Option<u64>,
    pub duration_ms: Option<u64>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Error => write!(f, "ERROR"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Alert rule configuration
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub id: String,
    pub metric_name: String,
    pub condition: AlertCondition,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub enabled: bool,
    pub processor_types: Vec<ProcessorType>,
    pub cooldown_seconds: u64,
    pub description: String,
    pub auto_resolve: bool,
    pub resolution_threshold: Option<f64>,
}

/// Alert condition types
#[derive(Debug, Clone)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
    PercentageIncrease(f64), // Compared to baseline
    PercentageDecrease(f64), // Compared to baseline
    RateOfChange(f64),       // Rate of change per second
}

/// Notification channels for alerts
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    Console,
    Log,
    Email { address: String },
    Webhook { url: String },
    File { path: String },
}

/// Alert management system
#[derive(Debug)]
pub struct AlertManager {
    pub active_alerts: HashMap<String, Alert>,
    pub alert_history: VecDeque<Alert>,
    pub notification_channels: Vec<NotificationChannel>,
    pub alert_rules: Vec<AlertRule>,
    max_history: usize,
    rule_cooldowns: HashMap<String, u64>, // rule_id -> last_triggered_timestamp
}

impl AlertManager {
    /// Create new alert manager
    pub fn new(max_history: usize) -> Self {
        Self {
            active_alerts: HashMap::new(),
            alert_history: VecDeque::with_capacity(max_history),
            notification_channels: vec![NotificationChannel::Console],
            alert_rules: Self::create_default_alert_rules(),
            max_history,
            rule_cooldowns: HashMap::new(),
        }
    }

    /// Add an alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.alert_rules.push(rule);
    }

    /// Remove an alert rule
    pub fn remove_rule(&mut self, rule_id: &str) {
        self.alert_rules.retain(|rule| rule.id != rule_id);
        self.rule_cooldowns.remove(rule_id);
    }

    /// Enable/disable a rule
    pub fn set_rule_enabled(&mut self, rule_id: &str, enabled: bool) {
        if let Some(rule) = self.alert_rules.iter_mut().find(|r| r.id == rule_id) {
            rule.enabled = enabled;
        }
    }

    /// Process a performance sample for alert evaluation
    pub fn process_sample(&mut self, sample: &PerformanceSample, baseline: Option<f64>) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let mut alerts_to_trigger = Vec::new();

        for rule in &self.alert_rules {
            if !rule.enabled {
                continue;
            }

            // Check if rule applies to this processor type
            if !rule.processor_types.is_empty()
                && !rule.processor_types.contains(&sample.processor_type)
            {
                continue;
            }

            // Check cooldown
            if let Some(&last_triggered) = self.rule_cooldowns.get(&rule.id) {
                if current_time - last_triggered < rule.cooldown_seconds {
                    continue;
                }
            }

            // Get metric value
            if let Some(metric_value) = sample.get_metric(&rule.metric_name) {
                if self.evaluate_condition(&rule.condition, metric_value, baseline) {
                    let alert = self.create_alert(rule, sample, metric_value);
                    alerts_to_trigger.push((alert, rule.id.clone()));
                }
            }
        }

        // Trigger alerts after iteration
        for (alert, rule_id) in alerts_to_trigger {
            self.trigger_alert(alert);
            self.rule_cooldowns.insert(rule_id, current_time);
        }

        // Check for auto-resolution of existing alerts
        self.check_auto_resolution(sample);
    }

    /// Evaluate alert condition
    fn evaluate_condition(
        &self,
        condition: &AlertCondition,
        value: f64,
        baseline: Option<f64>,
    ) -> bool {
        match condition {
            AlertCondition::GreaterThan => value > 0.0,
            AlertCondition::LessThan => value < 0.0,
            AlertCondition::Equals => (value - 0.0).abs() < f64::EPSILON,
            AlertCondition::NotEquals => (value - 0.0).abs() >= f64::EPSILON,
            AlertCondition::PercentageIncrease(threshold) => {
                if let Some(baseline_val) = baseline {
                    if baseline_val > 0.0 {
                        let increase = (value - baseline_val) / baseline_val;
                        increase > threshold / 100.0
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            AlertCondition::PercentageDecrease(threshold) => {
                if let Some(baseline_val) = baseline {
                    if baseline_val > 0.0 {
                        let decrease = (baseline_val - value) / baseline_val;
                        decrease > threshold / 100.0
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            AlertCondition::RateOfChange(_threshold) => {
                // This would require tracking previous values
                // For now, return false
                false
            }
        }
    }

    /// Create an alert from a rule and sample
    fn create_alert(
        &self,
        rule: &AlertRule,
        sample: &PerformanceSample,
        metric_value: f64,
    ) -> Alert {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let alert_id = format!(
            "{}:{}:{}:{}",
            rule.id, sample.processor_type, sample.processor_id, timestamp
        );

        let message = format!(
            "{}: {} = {:.2} (threshold: {:.2}) for {}:{}",
            rule.description,
            rule.metric_name,
            metric_value,
            rule.threshold,
            sample.processor_type,
            sample.processor_id
        );

        Alert {
            id: alert_id,
            timestamp,
            severity: rule.severity,
            message,
            processor_type: sample.processor_type,
            processor_id: sample.processor_id.clone(),
            metric_name: rule.metric_name.clone(),
            threshold_value: rule.threshold,
            actual_value: metric_value,
            resolved: false,
            resolution_timestamp: None,
            duration_ms: None,
        }
    }

    /// Trigger an alert
    fn trigger_alert(&mut self, alert: Alert) {
        // Add to active alerts
        self.active_alerts.insert(alert.id.clone(), alert.clone());

        // Add to history
        if self.alert_history.len() >= self.max_history {
            self.alert_history.pop_front();
        }
        self.alert_history.push_back(alert.clone());

        // Send notifications
        self.send_notifications(&alert);
    }

    /// Send notifications for an alert
    fn send_notifications(&self, alert: &Alert) {
        for channel in &self.notification_channels {
            match channel {
                NotificationChannel::Console => {
                    println!(
                        "[{}] {}: {}",
                        alert.severity, alert.timestamp, alert.message
                    );
                }
                NotificationChannel::Log => {
                    log::warn!(
                        "[{}] {}: {}",
                        alert.severity,
                        alert.timestamp,
                        alert.message
                    );
                }
                NotificationChannel::Email { address: _address } => {
                    // Email notification would be implemented here
                }
                NotificationChannel::Webhook { url: _url } => {
                    // Webhook notification would be implemented here
                }
                NotificationChannel::File { path: _path } => {
                    // File notification would be implemented here
                }
            }
        }
    }

    /// Check for auto-resolution of active alerts
    fn check_auto_resolution(&mut self, sample: &PerformanceSample) {
        let mut resolved_alerts = Vec::new();

        for alert in self.active_alerts.values_mut() {
            if alert.processor_type != sample.processor_type
                || alert.processor_id != sample.processor_id
            {
                continue;
            }

            // Find corresponding rule
            if let Some(rule) = self
                .alert_rules
                .iter()
                .find(|r| r.id.starts_with(&alert.metric_name))
            {
                if rule.auto_resolve {
                    if let Some(metric_value) = sample.get_metric(&alert.metric_name) {
                        let should_resolve = match rule.resolution_threshold {
                            Some(threshold) => match rule.condition {
                                AlertCondition::GreaterThan => metric_value <= threshold,
                                AlertCondition::LessThan => metric_value >= threshold,
                                _ => false,
                            },
                            None => {
                                // Use original threshold for resolution
                                match rule.condition {
                                    AlertCondition::GreaterThan => metric_value <= rule.threshold,
                                    AlertCondition::LessThan => metric_value >= rule.threshold,
                                    _ => false,
                                }
                            }
                        };

                        if should_resolve {
                            alert.resolved = true;
                            alert.resolution_timestamp = Some(sample.timestamp);
                            alert.duration_ms = Some(sample.timestamp - alert.timestamp);
                            resolved_alerts.push(alert.id.clone());
                        }
                    }
                }
            }
        }

        // Remove resolved alerts from active list
        for alert_id in resolved_alerts {
            self.active_alerts.remove(&alert_id);
        }
    }

    /// Manually resolve an alert
    pub fn resolve_alert(&mut self, alert_id: &str) -> bool {
        if let Some(alert) = self.active_alerts.get_mut(alert_id) {
            let current_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            alert.resolved = true;
            alert.resolution_timestamp = Some(current_time);
            alert.duration_ms = Some(current_time - alert.timestamp);

            self.active_alerts.remove(alert_id);
            true
        } else {
            false
        }
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> Vec<&Alert> {
        self.active_alerts.values().collect()
    }

    /// Get alerts by severity
    pub fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<&Alert> {
        self.active_alerts
            .values()
            .filter(|alert| alert.severity == severity)
            .collect()
    }

    /// Get alert statistics
    pub fn get_alert_stats(&self) -> AlertStats {
        let total_alerts = self.alert_history.len();
        let active_alerts = self.active_alerts.len();

        let mut severity_counts = HashMap::new();
        for alert in &self.alert_history {
            *severity_counts.entry(alert.severity).or_insert(0) += 1;
        }

        let resolved_alerts = self
            .alert_history
            .iter()
            .filter(|alert| alert.resolved)
            .count();

        let avg_resolution_time = if resolved_alerts > 0 {
            let total_duration: u64 = self
                .alert_history
                .iter()
                .filter_map(|alert| alert.duration_ms)
                .sum();
            Some(total_duration as f64 / resolved_alerts as f64)
        } else {
            None
        };

        AlertStats {
            total_alerts,
            active_alerts,
            resolved_alerts,
            severity_counts,
            avg_resolution_time_ms: avg_resolution_time,
        }
    }

    /// Create default alert rules
    pub fn create_default_alert_rules() -> Vec<AlertRule> {
        vec![
            AlertRule {
                id: "high_execution_time".to_string(),
                metric_name: "execution_time_ms".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 1000.0,
                severity: AlertSeverity::Warning,
                enabled: true,
                processor_types: vec![],
                cooldown_seconds: 60,
                description: "High execution time detected".to_string(),
                auto_resolve: true,
                resolution_threshold: Some(500.0),
            },
            AlertRule {
                id: "low_throughput".to_string(),
                metric_name: "throughput_ops_per_sec".to_string(),
                condition: AlertCondition::LessThan,
                threshold: 10.0,
                severity: AlertSeverity::Error,
                enabled: true,
                processor_types: vec![],
                cooldown_seconds: 120,
                description: "Low throughput detected".to_string(),
                auto_resolve: true,
                resolution_threshold: Some(50.0),
            },
            AlertRule {
                id: "high_error_rate".to_string(),
                metric_name: "error_rate".to_string(),
                condition: AlertCondition::GreaterThan,
                threshold: 0.1,
                severity: AlertSeverity::Critical,
                enabled: true,
                processor_types: vec![],
                cooldown_seconds: 30,
                description: "High error rate detected".to_string(),
                auto_resolve: true,
                resolution_threshold: Some(0.05),
            },
            AlertRule {
                id: "low_cache_hit_ratio".to_string(),
                metric_name: "cache_hit_ratio".to_string(),
                condition: AlertCondition::LessThan,
                threshold: 0.5,
                severity: AlertSeverity::Warning,
                enabled: true,
                processor_types: vec![],
                cooldown_seconds: 300,
                description: "Low cache hit ratio detected".to_string(),
                auto_resolve: true,
                resolution_threshold: Some(0.7),
            },
        ]
    }

    /// Clear all alerts
    pub fn clear_all_alerts(&mut self) {
        self.active_alerts.clear();
        self.alert_history.clear();
        self.rule_cooldowns.clear();
    }

    /// Add notification channel
    pub fn add_notification_channel(&mut self, channel: NotificationChannel) {
        self.notification_channels.push(channel);
    }
}

/// Alert statistics
#[derive(Debug, Clone)]
pub struct AlertStats {
    pub total_alerts: usize,
    pub active_alerts: usize,
    pub resolved_alerts: usize,
    pub severity_counts: HashMap<AlertSeverity, usize>,
    pub avg_resolution_time_ms: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_creation() {
        let alert = Alert {
            id: "test-alert".to_string(),
            timestamp: 12345,
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            processor_type: ProcessorType::QuantumInspired,
            processor_id: "test-processor".to_string(),
            metric_name: "execution_time_ms".to_string(),
            threshold_value: 1000.0,
            actual_value: 1500.0,
            resolved: false,
            resolution_timestamp: None,
            duration_ms: None,
        };

        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert!(!alert.resolved);
    }

    #[test]
    fn test_alert_manager_creation() {
        let manager = AlertManager::new(1000);
        assert_eq!(manager.max_history, 1000);
        assert!(!manager.alert_rules.is_empty()); // Should have default rules
    }

    #[test]
    fn test_add_remove_rules() {
        let mut manager = AlertManager::new(1000);
        let initial_count = manager.alert_rules.len();

        let rule = AlertRule {
            id: "test_rule".to_string(),
            metric_name: "test_metric".to_string(),
            condition: AlertCondition::GreaterThan,
            threshold: 100.0,
            severity: AlertSeverity::Info,
            enabled: true,
            processor_types: vec![],
            cooldown_seconds: 60,
            description: "Test rule".to_string(),
            auto_resolve: false,
            resolution_threshold: None,
        };

        manager.add_rule(rule);
        assert_eq!(manager.alert_rules.len(), initial_count + 1);

        manager.remove_rule("test_rule");
        assert_eq!(manager.alert_rules.len(), initial_count);
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::Error);
        assert!(AlertSeverity::Error > AlertSeverity::Warning);
        assert!(AlertSeverity::Warning > AlertSeverity::Info);
    }

    #[test]
    fn test_alert_severity_display() {
        assert_eq!(AlertSeverity::Info.to_string(), "INFO");
        assert_eq!(AlertSeverity::Warning.to_string(), "WARNING");
        assert_eq!(AlertSeverity::Error.to_string(), "ERROR");
        assert_eq!(AlertSeverity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_alert_stats() {
        let mut manager = AlertManager::new(1000);
        let stats = manager.get_alert_stats();

        assert_eq!(stats.total_alerts, 0);
        assert_eq!(stats.active_alerts, 0);
        assert_eq!(stats.resolved_alerts, 0);
    }

    #[test]
    fn test_rule_enable_disable() {
        let mut manager = AlertManager::new(1000);

        if let Some(rule) = manager.alert_rules.first() {
            let rule_id = rule.id.clone();
            let initial_enabled = rule.enabled;

            manager.set_rule_enabled(&rule_id, !initial_enabled);

            if let Some(updated_rule) = manager.alert_rules.iter().find(|r| r.id == rule_id) {
                assert_eq!(updated_rule.enabled, !initial_enabled);
            }
        }
    }

    #[test]
    fn test_notification_channels() {
        let mut manager = AlertManager::new(1000);
        let initial_count = manager.notification_channels.len();

        manager.add_notification_channel(NotificationChannel::Email {
            address: "test@example.com".to_string(),
        });

        assert_eq!(manager.notification_channels.len(), initial_count + 1);
    }
}
