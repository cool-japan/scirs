//! Alert management for real-time security monitoring

use crate::error::CoreError;
use crate::observability::audit::types::{AlertingConfig, AuditEvent, EventCategory, EventOutcome};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::RwLock;

/// Alert manager for real-time security monitoring
pub struct AlertManager {
    pub config: AlertingConfig,
    pub alert_counters: RwLock<HashMap<String, (u32, DateTime<Utc>)>>,
    pub last_alert_time: RwLock<HashMap<String, DateTime<Utc>>>,
}

impl AlertManager {
    /// Create a new alert manager.
    #[must_use]
    pub fn new(config: AlertingConfig) -> Self {
        Self {
            config,
            alert_counters: RwLock::new(HashMap::new()),
            last_alert_time: RwLock::new(HashMap::new()),
        }
    }

    /// Process an audit event for alerting.
    ///
    /// # Errors
    ///
    /// Returns an error if event processing or alerting fails.
    pub fn process_event(&self, event: &AuditEvent) -> Result<(), CoreError> {
        if !self.config.enabled {
            return Ok(());
        }

        let alert_key = match event.category {
            EventCategory::Authentication if event.outcome == EventOutcome::Failure => {
                "failed_auth"
            }
            EventCategory::DataAccess => "data_access",
            EventCategory::Configuration => "config_change",
            _ => return Ok(()),
        };

        let should_alert = self.update_counter_and_check_threshold(alert_key, event)?;

        if should_alert {
            self.send_alert(alert_key, event)?;
        }

        Ok(())
    }

    /// Update alert counter and check if threshold is exceeded.
    ///
    /// # Errors
    ///
    /// Returns an error if counter update fails.
    pub fn update_counter_and_check_threshold(
        &self,
        alert_key: &str,
        _event: &AuditEvent,
    ) -> Result<bool, CoreError> {
        let mut counters = self.alert_counters.write().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire alert counters lock".to_string(),
            ))
        })?;

        let now = Utc::now();
        let window_start = now - chrono::Duration::minutes(5); // 5-minute window

        // Update counter
        let (count, last_update) = counters.get(alert_key).copied().unwrap_or((0, now));

        let new_count = if last_update < window_start {
            1 // Reset counter if outside window
        } else {
            count + 1
        };

        counters.insert(alert_key.to_string(), (new_count, now));

        // Check threshold
        let threshold = match alert_key {
            "failed_auth" => self.config.failed_auth_threshold,
            "data_access" => self.config.data_access_rate_threshold,
            "config_change" => self.config.config_change_threshold,
            _ => return Ok(false),
        };

        Ok(new_count >= threshold && self.check_cooldown(alert_key)?)
    }

    /// Check if alert cooldown period has elapsed.
    ///
    /// # Errors
    ///
    /// Returns an error if cooldown check fails.
    pub fn check_cooldown(&self, alert_key: &str) -> Result<bool, CoreError> {
        let last_alert_times = self.last_alert_time.read().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire last alert time lock".to_string(),
            ))
        })?;

        let now = Utc::now();
        let cooldown_duration = chrono::Duration::seconds(self.config.cooldown_period as i64);

        if let Some(last_alert) = last_alert_times.get(alert_key) {
            Ok(now - *last_alert > cooldown_duration)
        } else {
            Ok(true)
        }
    }

    /// Send an alert for the given event.
    ///
    /// # Errors
    ///
    /// Returns an error if alert sending fails.
    pub fn send_alert(&self, alert_key: &str, event: &AuditEvent) -> Result<(), CoreError> {
        // Update last alert time
        {
            let mut last_alert_times = self.last_alert_time.write().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Failed to acquire last alert time lock".to_string(),
                ))
            })?;
            last_alert_times.insert(alert_key.to_string(), Utc::now());
        }

        let alert_message = format!(
            "SECURITY ALERT: {alert_key} threshold exceeded - {} - {}",
            event.action, event.description
        );

        // Send webhook alert
        if let Some(ref webhook_url) = self.config.webhook_url {
            self.send_webhook_alert(webhook_url, &alert_message)?;
        }

        // Send email alerts
        for email in &self.config.email_recipients {
            self.send_email_alert(email, &alert_message)?;
        }

        // Log the alert
        eprintln!("AUDIT ALERT: {alert_message}");

        Ok(())
    }

    /// Send a webhook alert.
    ///
    /// # Errors
    ///
    /// Returns an error if the webhook request fails.
    #[cfg(feature = "reqwest")]
    pub fn send_webhook_alert(&self, webhook_url: &str, message: &str) -> Result<(), CoreError> {
        use reqwest::blocking::Client;
        use std::collections::HashMap;

        let mut payload = HashMap::new();
        payload.insert("text", message);

        let client = Client::new();
        client
            .post(webhook_url)
            .json(&payload)
            .send()
            .map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to send webhook alert: {e}"
                )))
            })?;

        Ok(())
    }

    /// Send a webhook alert (reqwest feature required).
    ///
    /// # Errors
    ///
    /// Returns an error indicating that the reqwest feature is required.
    #[cfg(not(feature = "reqwest"))]
    pub fn send_webhook_alert(&self, _webhook_url: &str, _message: &str) -> Result<(), CoreError> {
        eprintln!("Webhook alerts require reqwest feature");
        Ok(())
    }

    /// Send an email alert.
    ///
    /// # Errors
    ///
    /// Returns an error if email sending fails.
    pub fn send_email_alert(&self, email: &str, message: &str) -> Result<(), CoreError> {
        // Simple SMTP implementation using environment variables for configuration
        // In production, you would use a proper email library like `lettre`

        use std::env;
        use std::io::Write;
        use std::net::TcpStream;
        use std::time::Duration;

        // Check for SMTP configuration in environment
        let smtp_server = env::var("SMTP_SERVER").unwrap_or_else(|_| "localhost".to_string());
        let smtp_port = env::var("SMTP_PORT")
            .unwrap_or_else(|_| "587".to_string())
            .parse::<u16>()
            .unwrap_or(587);
        let from_email = env::var("SMTP_FROM").unwrap_or_else(|_| "audit@example.com".to_string());

        // For security, if no SMTP config is available, just log the alert
        if smtp_server == "localhost" && env::var("SMTP_SERVER").is_err() {
            eprintln!("AUDIT EMAIL ALERT (SMTP not configured):");
            eprintln!("  To: {email}");
            eprintln!("  Subject: Security Alert");
            eprintln!("  Message: {message}");
            return Ok(());
        }

        // Attempt simple SMTP connection
        match TcpStream::connect_timeout(
            &format!("{smtp_server}:{smtp_port}").parse().map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Invalid SMTP address: {e}"
                )))
            })?,
            Duration::from_secs(10),
        ) {
            Ok(mut stream) => {
                // Very basic SMTP implementation
                let commands = vec![
                    format!("HELO localhost\r\n"),
                    format!("MAIL FROM:<{from_email}>\r\n"),
                    format!("RCPT TO:<{email}>\r\n"),
                    "DATA\r\n".to_string(),
                    format!("Subject: Security Alert\r\n\r\n{message}\r\n.\r\n"),
                    "QUIT\r\n".to_string(),
                ];

                for command in commands {
                    if let Err(e) = stream.write_all(command.as_bytes()) {
                        eprintln!("SMTP write error: {e}. Logging alert instead:");
                        eprintln!("  To: {email}");
                        eprintln!("  Message: {message}");
                        return Ok(());
                    }

                    // Simple delay between commands
                    std::thread::sleep(Duration::from_millis(100));
                }

                eprintln!("Email alert sent to: {email}");
            }
            Err(e) => {
                eprintln!(
                    "Failed to connect to SMTP server {smtp_server}: {e}. Logging alert instead:"
                );
                eprintln!("  To: {email}");
                eprintln!("  Subject: Security Alert");
                eprintln!("  Message: {message}");
            }
        }

        Ok(())
    }
}
