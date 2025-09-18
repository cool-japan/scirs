//! Alerting and notification system for performance monitoring

use super::types::{
    AlertCondition, AlertEvent, AlertEventType, AlertRule, AlertSeverity, ComparisonOperator,
    ComprehensivePerformanceMetrics, NotificationChannel, PerformanceAlert,
};
use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::VecDeque;
use std::time::{Duration, Instant, SystemTime};

/// Comprehensive alerting system for performance monitoring
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlertingSystem {
    active_alerts: Vec<PerformanceAlert>,
    alert_rules: Vec<AlertRule>,
    alert_history: VecDeque<AlertEvent>,
    notification_channels: Vec<NotificationChannel>,
}

impl AlertingSystem {
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            active_alerts: Vec::new(),
            alert_rules: Self::default_alert_rules(),
            alert_history: VecDeque::with_capacity(1000),
            notification_channels: Vec::new(),
        })
    }

    fn default_alert_rules() -> Vec<AlertRule> {
        vec![
            AlertRule {
                name: "High CPU Usage".to_string(),
                condition: AlertCondition::Threshold {
                    metric: "cpu_utilization".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    value: 0.9,
                },
                severity: AlertSeverity::Warning,
                duration: Duration::from_secs(60),
            },
            AlertRule {
                name: "Critical CPU Usage".to_string(),
                condition: AlertCondition::Threshold {
                    metric: "cpu_utilization".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    value: 0.95,
                },
                severity: AlertSeverity::Critical,
                duration: Duration::from_secs(30),
            },
            AlertRule {
                name: "High Memory Usage".to_string(),
                condition: AlertCondition::Threshold {
                    metric: "memory_utilization".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    value: 0.9,
                },
                severity: AlertSeverity::Warning,
                duration: Duration::from_secs(120),
            },
            AlertRule {
                name: "Low Throughput".to_string(),
                condition: AlertCondition::Threshold {
                    metric: "operations_per_second".to_string(),
                    operator: ComparisonOperator::LessThan,
                    value: 100.0,
                },
                severity: AlertSeverity::Warning,
                duration: Duration::from_secs(180),
            },
        ]
    }

    pub fn check_and_trigger_alerts(
        &mut self,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<()> {
        // Collect rules that need to trigger alerts to avoid borrowing conflicts
        let mut rules_to_trigger = Vec::new();
        for rule in &self.alert_rules {
            if self.evaluate_rule(rule, metrics)? {
                rules_to_trigger.push(rule.clone());
            }
        }

        // Trigger alerts for collected rules
        for rule in rules_to_trigger {
            let alert = PerformanceAlert {
                id: format!(
                    "alert_{}",
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_millis()
                ),
                rule_name: rule.name.clone(),
                severity: rule.severity,
                message: format!("Alert triggered for rule: {}", rule.name),
                triggered_at: Instant::now(),
                acknowledged: false,
                resolved: false,
            };

            // Add to active alerts
            self.active_alerts.push(alert.clone());

            // Add to history
            self.alert_history.push_back(AlertEvent {
                alert,
                event_type: AlertEventType::Triggered,
                timestamp: Instant::now(),
            });
        }

        // Clean up resolved alerts
        self.clean_up_resolved_alerts(metrics)?;

        Ok(())
    }

    fn evaluate_rule(
        &self,
        rule: &AlertRule,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<bool> {
        match &rule.condition {
            AlertCondition::Threshold {
                metric,
                operator,
                value,
            } => {
                let metric_value = self.get_metric_value(metric, metrics)?;

                let condition_met = match operator {
                    ComparisonOperator::GreaterThan => metric_value > *value,
                    ComparisonOperator::LessThan => metric_value < *value,
                    ComparisonOperator::Equal => (metric_value - value).abs() < 0.001,
                };

                Ok(condition_met)
            }
            AlertCondition::RateOfChange {
                metric,
                threshold,
                timeframe: _,
            } => {
                // Simplified rate of change calculation
                let current_value = self.get_metric_value(metric, metrics)?;
                // Would need historical data for proper rate calculation
                Ok(current_value.abs() > *threshold)
            }
        }
    }

    fn get_metric_value(
        &self,
        metric: &str,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<f64> {
        match metric {
            "cpu_utilization" => Ok(metrics.cpu_utilization),
            "memory_utilization" => Ok(metrics.memory_utilization),
            "operations_per_second" => Ok(metrics.operations_per_second),
            "average_latency_ms" => Ok(metrics.average_latency_ms),
            "cache_miss_rate" => Ok(metrics.cache_miss_rate),
            _ => Err(CoreError::ValidationError(ErrorContext {
                message: format!("Unknown metric: {metric}"),
                location: None,
                cause: None,
            })),
        }
    }

    fn check_alerts(&mut self, metrics: &ComprehensivePerformanceMetrics) -> CoreResult<()> {
        // Collect rules that need to trigger alerts to avoid borrowing conflicts
        let mut rules_to_trigger = Vec::new();
        for rule in &self.alert_rules {
            if self.evaluate_rule(rule, metrics)? {
                // Check if alert is already active
                if !self
                    .active_alerts
                    .iter()
                    .any(|alert| alert.rule_name == rule.name)
                {
                    rules_to_trigger.push(rule.clone());
                }
            }
        }

        // Trigger alerts for rules that are not already active
        for rule in rules_to_trigger {
            let alert = PerformanceAlert {
                id: format!(
                    "alert_{}",
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_millis()
                ),
                rule_name: rule.name.clone(),
                severity: rule.severity,
                message: format!("Alert triggered for rule: {}", rule.name),
                triggered_at: Instant::now(),
                acknowledged: false,
                resolved: false,
            };

            // Add to active alerts
            self.active_alerts.push(alert.clone());

            // Add to history
            self.alert_history.push_back(AlertEvent {
                alert,
                event_type: AlertEventType::Triggered,
                timestamp: Instant::now(),
            });

            // Implement notification sending
            self.send_notifications(&rule.name, rule.severity)?;
        }

        Ok(())
    }

    fn clean_up_resolved_alerts(
        &mut self,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<()> {
        let mut resolved_alerts = Vec::new();

        // Collect rules and alert info to avoid borrowing conflicts
        let rule_evaluations: Vec<(usize, bool)> = self
            .active_alerts
            .iter()
            .enumerate()
            .map(|(index, alert)| {
                if let Some(rule) = self.alert_rules.iter().find(|r| r.name == alert.rule_name) {
                    let is_resolved = match self.evaluate_rule(rule, metrics) {
                        Ok(condition_met) => !condition_met,
                        Err(_) => false,
                    };
                    (index, is_resolved)
                } else {
                    (index, false)
                }
            })
            .collect();

        // Mark alerts as resolved based on evaluations
        for (index, is_resolved) in rule_evaluations {
            if is_resolved {
                if let Some(alert) = self.active_alerts.get_mut(index) {
                    alert.resolved = true;
                    resolved_alerts.push(alert.clone());
                }
            }
        }

        // Remove resolved alerts from active list
        self.active_alerts.retain(|alert| !alert.resolved);

        // Add resolved events to history
        for alert in resolved_alerts {
            self.alert_history.push_back(AlertEvent {
                alert,
                event_type: AlertEventType::Resolved,
                timestamp: Instant::now(),
            });
        }

        Ok(())
    }

    fn send_alert(&self, alertname: &str, severity: AlertSeverity) -> CoreResult<()> {
        for channel in &self.notification_channels {
            channel.send_notification(alertname, severity)?;
        }
        Ok(())
    }

    pub fn get_active_alerts(&self) -> CoreResult<Vec<PerformanceAlert>> {
        Ok(self.active_alerts.clone())
    }

    pub fn acknowledge_alert(&mut self, alertid: &str) -> CoreResult<()> {
        if let Some(alert) = self.active_alerts.iter_mut().find(|a| a.id == alertid) {
            alert.acknowledged = true;
        }
        Ok(())
    }

    /// Send notifications through all enabled channels
    fn send_notifications(&self, alert_name: &str, severity: AlertSeverity) -> CoreResult<()> {
        // Check if notifications should be sent based on severity
        let should_notify = match severity {
            AlertSeverity::Critical => true,
            AlertSeverity::Warning => {
                // For warning severity, check if there are multiple active alerts
                self.active_alerts.len() > 3
            }
            AlertSeverity::Info => false, // Don't notify for info severity
        };

        if !should_notify {
            return Ok(());
        }

        // Send notification through each enabled channel
        let mut errors = Vec::new();
        for channel in &self.notification_channels {
            if let Err(e) = channel.send_notification(alert_name, severity) {
                // Collect errors but don't fail the entire operation
                errors.push(format!("Failed to send notification: {}", e));
            }
        }

        // Log any notification errors (in production, this would use proper logging)
        if !errors.is_empty() {
            #[cfg(feature = "logging")]
            log::warn!("Notification errors: {:?}", errors);
        }

        Ok(())
    }

    /// Add a new alert rule to the system
    pub fn add_alert_rule(&mut self, rule: AlertRule) -> CoreResult<()> {
        // Check if rule with same name already exists
        if self.alert_rules.iter().any(|r| r.name == rule.name) {
            return Err(CoreError::ValidationError(ErrorContext {
                message: format!("Alert rule with name '{}' already exists", rule.name),
                location: None,
                cause: None,
            }));
        }

        self.alert_rules.push(rule);
        Ok(())
    }

    /// Remove an alert rule by name
    pub fn remove_alert_rule(&mut self, rule_name: &str) -> CoreResult<bool> {
        let initial_len = self.alert_rules.len();
        self.alert_rules.retain(|rule| rule.name != rule_name);
        Ok(self.alert_rules.len() < initial_len)
    }

    /// Add a notification channel
    pub fn add_notification_channel(&mut self, channel: NotificationChannel) -> CoreResult<()> {
        self.notification_channels.push(channel);
        Ok(())
    }

    /// Get alert history (last N events)
    pub fn get_alert_history(&self, limit: Option<usize>) -> CoreResult<Vec<AlertEvent>> {
        let history: Vec<AlertEvent> = if let Some(limit) = limit {
            self.alert_history
                .iter()
                .rev()
                .take(limit)
                .cloned()
                .collect()
        } else {
            self.alert_history.iter().cloned().collect()
        };

        Ok(history)
    }

    /// Get statistics about alerting system
    pub fn get_alerting_stats(&self) -> CoreResult<AlertingStats> {
        let total_rules = self.alert_rules.len();
        let active_alerts_count = self.active_alerts.len();
        let total_events = self.alert_history.len();

        // Count events by type
        let mut triggered_count = 0;
        let mut resolved_count = 0;
        let mut acknowledged_count = 0;

        for event in &self.alert_history {
            match event.event_type {
                AlertEventType::Triggered => triggered_count += 1,
                AlertEventType::Resolved => resolved_count += 1,
                AlertEventType::Acknowledged => acknowledged_count += 1,
            }
        }

        // Count alerts by severity
        let mut critical_count = 0;
        let mut warning_count = 0;
        let mut info_count = 0;

        for alert in &self.active_alerts {
            match alert.severity {
                AlertSeverity::Critical => critical_count += 1,
                AlertSeverity::Warning => warning_count += 1,
                AlertSeverity::Info => info_count += 1,
            }
        }

        Ok(AlertingStats {
            total_rules,
            active_alerts_count,
            total_events,
            triggered_count,
            resolved_count,
            acknowledged_count,
            critical_count,
            warning_count,
            info_count,
            notification_channels_count: self.notification_channels.len(),
        })
    }

    /// Resolve all alerts for a specific rule
    pub fn resolve_alerts_for_rule(&mut self, rule_name: &str) -> CoreResult<usize> {
        let mut resolved_count = 0;
        let mut resolved_alerts = Vec::new();

        // Mark matching alerts as resolved
        for alert in &mut self.active_alerts {
            if alert.rule_name == rule_name && !alert.resolved {
                alert.resolved = true;
                resolved_alerts.push(alert.clone());
                resolved_count += 1;
            }
        }

        // Remove resolved alerts from active list
        self.active_alerts.retain(|alert| !alert.resolved);

        // Add resolved events to history
        for alert in resolved_alerts {
            self.alert_history.push_back(AlertEvent {
                alert,
                event_type: AlertEventType::Resolved,
                timestamp: Instant::now(),
            });
        }

        Ok(resolved_count)
    }

    /// Clear old alert history to prevent memory buildup
    pub fn cleanup_alert_history(&mut self, max_age: Duration) -> CoreResult<usize> {
        let cutoff_time = Instant::now() - max_age;
        let initial_count = self.alert_history.len();

        // Remove events older than cutoff time
        self.alert_history.retain(|event| event.timestamp > cutoff_time);

        let removed_count = initial_count - self.alert_history.len();
        Ok(removed_count)
    }
}

/// Statistics about the alerting system
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AlertingStats {
    pub total_rules: usize,
    pub active_alerts_count: usize,
    pub total_events: usize,
    pub triggered_count: usize,
    pub resolved_count: usize,
    pub acknowledged_count: usize,
    pub critical_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub notification_channels_count: usize,
}