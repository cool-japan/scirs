//! Alert management for streaming metrics
//!
//! This module provides comprehensive alerting capabilities for streaming
//! metrics systems with rate limiting and multiple channels.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{Alert, AlertConfig, SentAlert};
use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Alerts manager
#[derive(Debug, Clone)]
pub struct AlertsManager {
    config: AlertConfig,
    pending_alerts: VecDeque<Alert>,
    sent_alerts: VecDeque<SentAlert>,
    rate_limiter: HashMap<String, Instant>,
}

impl AlertsManager {
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config,
            pending_alerts: VecDeque::new(),
            sent_alerts: VecDeque::new(),
            rate_limiter: HashMap::new(),
        }
    }

    pub fn send_alert(&mut self, alert: Alert) -> Result<()> {
        // Check rate limiting
        let key = format!("{}_{:?}", alert.title, alert.severity);
        let now = Instant::now();

        if let Some(&last_sent) = self.rate_limiter.get(&key) {
            if now.duration_since(last_sent) < self.config.rate_limit {
                return Ok(()); // Rate limited
            }
        }

        self.rate_limiter.insert(key, now);
        self.pending_alerts.push_back(alert);

        // Process pending alerts
        self.process_pending_alerts()?;

        Ok(())
    }

    fn process_pending_alerts(&mut self) -> Result<()> {
        while let Some(alert) = self.pending_alerts.pop_front() {
            let success = self.deliver_alert(&alert)?;

            let sent_alert = SentAlert {
                alert,
                sent_at: Instant::now(),
                channels: self.get_enabled_channels(),
                success,
                error_message: if success {
                    None
                } else {
                    Some("Delivery failed".to_string())
                },
            };

            self.sent_alerts.push_back(sent_alert);

            // Keep only recent sent alerts for memory management
            if self.sent_alerts.len() > 1000 {
                self.sent_alerts.pop_front();
            }
        }

        Ok(())
    }

    fn deliver_alert(&self, alert: &Alert) -> Result<bool> {
        let mut delivery_success = false;

        // Email delivery
        if self.config.email_enabled && !self.config.email_addresses.is_empty() {
            delivery_success |= self.send_email_alert(alert)?;
        }

        // Webhook delivery
        if self.config.webhook_enabled && !self.config.webhook_urls.is_empty() {
            delivery_success |= self.send_webhook_alert(alert)?;
        }

        // Log delivery
        if self.config.log_enabled {
            delivery_success |= self.log_alert(alert)?;
        }

        Ok(delivery_success)
    }

    fn send_email_alert(&self, alert: &Alert) -> Result<bool> {
        // In a real implementation, this would send emails
        println!("EMAIL ALERT: {} - {}", alert.title, alert.message);
        Ok(true)
    }

    fn send_webhook_alert(&self, alert: &Alert) -> Result<bool> {
        // In a real implementation, this would make HTTP requests
        println!("WEBHOOK ALERT: {} - {}", alert.title, alert.message);
        Ok(true)
    }

    fn log_alert(&self, alert: &Alert) -> Result<bool> {
        // In a real implementation, this would write to log files
        let log_message = format!(
            "[{}] {:?} - {}: {}",
            alert.timestamp.elapsed().as_secs(),
            alert.severity,
            alert.title,
            alert.message
        );

        if let Some(ref log_file) = self.config.log_file {
            println!("LOG to {}: {}", log_file, log_message);
        } else {
            println!("LOG: {}", log_message);
        }

        Ok(true)
    }

    fn get_enabled_channels(&self) -> Vec<String> {
        let mut channels = Vec::new();

        if self.config.email_enabled {
            channels.push("email".to_string());
        }
        if self.config.webhook_enabled {
            channels.push("webhook".to_string());
        }
        if self.config.log_enabled {
            channels.push("log".to_string());
        }

        channels
    }

    pub fn get_pending_alerts(&self) -> &VecDeque<Alert> {
        &self.pending_alerts
    }

    pub fn get_sent_alerts(&self) -> &VecDeque<SentAlert> {
        &self.sent_alerts
    }

    pub fn get_alert_statistics(&self) -> AlertStatistics {
        let total_sent = self.sent_alerts.len();
        let successful_sent = self.sent_alerts.iter().filter(|a| a.success).count();
        let failed_sent = total_sent - successful_sent;

        AlertStatistics {
            total_alerts: total_sent,
            successful_alerts: successful_sent,
            failed_alerts: failed_sent,
            pending_alerts: self.pending_alerts.len(),
            success_rate: if total_sent > 0 {
                successful_sent as f64 / total_sent as f64
            } else {
                0.0
            },
        }
    }

    pub fn clear_sent_alerts(&mut self) {
        self.sent_alerts.clear();
    }

    pub fn update_config(&mut self, config: AlertConfig) {
        self.config = config;
    }

    pub fn get_config(&self) -> &AlertConfig {
        &self.config
    }

    pub fn cleanup_rate_limiter(&mut self, retention_period_secs: u64) {
        let now = Instant::now();
        let retention_duration = std::time::Duration::from_secs(retention_period_secs);

        self.rate_limiter
            .retain(|_, &mut last_sent| now.duration_since(last_sent) < retention_duration);
    }
}

/// Alert system statistics
#[derive(Debug, Clone)]
pub struct AlertStatistics {
    pub total_alerts: usize,
    pub successful_alerts: usize,
    pub failed_alerts: usize,
    pub pending_alerts: usize,
    pub success_rate: f64,
}
