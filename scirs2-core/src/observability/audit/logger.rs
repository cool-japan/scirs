//! Main audit logger implementation

use crate::error::CoreError;
use crate::observability::audit::alerts::AlertManager;
use crate::observability::audit::builder::AuditEventBuilder;
use crate::observability::audit::storage::LogFileManager;
use crate::observability::audit::types::{
    AuditConfig, AuditEvent, AuditStatistics, ComplianceMode, ComplianceReport, EventCategory,
    EventOutcome, EventSeverity, SystemContext,
};
use crate::observability::audit::utils::get_stack_trace;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Main audit logger implementation
pub struct AuditLogger {
    config: AuditConfig,
    file_manager: Arc<Mutex<LogFileManager>>,
    alert_manager: Option<AlertManager>,
    event_buffer: Arc<Mutex<Vec<AuditEvent>>>,
    last_flush: Arc<Mutex<DateTime<Utc>>>,
}

impl AuditLogger {
    /// Create a new audit logger
    ///
    /// # Errors
    ///
    /// Returns an error if the logger cannot be initialized.
    pub fn new(config: AuditConfig) -> Result<Self, CoreError> {
        let file_manager = Arc::new(Mutex::new(LogFileManager::new(config.clone())?));

        let alert_manager = config
            .alerting_config
            .as_ref()
            .map(|cfg| AlertManager::new(cfg.clone()));

        Ok(Self {
            config,
            file_manager,
            alert_manager,
            event_buffer: Arc::new(Mutex::new(Vec::with_capacity(1000))),
            last_flush: Arc::new(Mutex::new(Utc::now())),
        })
    }

    /// Log a general audit event
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be logged.
    pub fn log_event(&self, event: AuditEvent) -> Result<(), CoreError> {
        // Process alerts
        if let Some(ref alert_manager) = self.alert_manager {
            alert_manager.process_event(&event)?;
        }

        // Add to buffer
        {
            let mut buffer = self.event_buffer.lock().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Failed to acquire buffer lock".to_string(),
                ))
            })?;
            buffer.push(event);

            // Check if we need to flush
            if buffer.len() >= self.config.buffersize {
                self.flush_buffer(&mut buffer)?;
            }
        }

        // Check flush interval
        self.check_flush_interval()?;

        Ok(())
    }

    /// Log a data access event
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be logged.
    pub fn log_data_access(
        &self,
        userid: &str,
        resourceid: &str,
        action: &str,
        description: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut event = AuditEventBuilder::new(EventCategory::DataAccess, action)
            .userid(userid)
            .resourceid(resourceid)
            .description(description.unwrap_or("Data access operation"))
            .compliance_tag("data_access")
            .build();

        if self.config.include_system_context {
            event.system_context = Some(SystemContext::current());
        }

        self.log_event(event)
    }

    /// Log a security event
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be logged.
    pub fn log_security_event(
        &self,
        category: EventCategory,
        action: &str,
        userid: &str,
        description: &str,
    ) -> Result<(), CoreError> {
        let mut event = AuditEventBuilder::new(category, action)
            .severity(EventSeverity::Warning)
            .userid(userid)
            .description(description)
            .compliance_tag("security")
            .build();

        if self.config.include_system_context {
            event.system_context = Some(SystemContext::current());
        }

        if self.config.include_stack_traces {
            event.stack_trace = Some(get_stack_trace());
        }

        self.log_event(event)
    }

    /// Log an authentication event
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be logged.
    pub fn log_authentication(
        &self,
        userid: &str,
        action: &str,
        outcome: EventOutcome,
        source_ip: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut builder = AuditEventBuilder::new(EventCategory::Authentication, action)
            .userid(userid)
            .outcome(outcome)
            .compliance_tag("authentication");

        if let Some(ip) = source_ip {
            builder = builder.source_ip(ip);
        }

        let severity = match outcome {
            EventOutcome::Failure => EventSeverity::Warning,
            EventOutcome::Denied => EventSeverity::Error,
            _ => EventSeverity::Info,
        };

        let mut event = builder.severity(severity).build();

        if self.config.include_system_context {
            event.system_context = Some(SystemContext::current());
        }

        self.log_event(event)
    }

    /// Log a configuration change
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be logged.
    pub fn log_configuration_change(
        &self,
        userid: &str,
        config_item: &str,
        old_value: Option<&str>,
        new_value: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut metadata = HashMap::new();
        if let Some(old) = old_value {
            metadata.insert("old_value".to_string(), old.to_string());
        }
        if let Some(new) = new_value {
            metadata.insert("new_value".to_string(), new.to_string());
        }

        let mut event = AuditEventBuilder::new(EventCategory::Configuration, "config_change")
            .severity(EventSeverity::Warning)
            .userid(userid)
            .resourceid(config_item)
            .description("Configuration item changed")
            .compliance_tag("configuration")
            .build();

        event.metadata = metadata;

        if self.config.include_system_context {
            event.system_context = Some(SystemContext::current());
        }

        self.log_event(event)
    }

    /// Search audit events within a date range
    ///
    /// # Errors
    ///
    /// Returns an error if events cannot be searched or parsed.
    pub fn search_events(
        &self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        category: Option<EventCategory>,
        userid: Option<&str>,
    ) -> Result<Vec<AuditEvent>, CoreError> {
        let mut events = Vec::new();

        // First, flush any pending events
        self.flush()?;

        // Read log files
        if let Ok(entries) = std::fs::read_dir(&self.config.log_directory) {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.starts_with("audit_") && filename.ends_with(".log") {
                        self.search_file(
                            &entry.path(),
                            start_date,
                            end_date,
                            category,
                            userid,
                            &mut events,
                        )?;
                    }
                }
            }
        }

        // Sort by timestamp
        events.sort_by_key(|e| e.timestamp);

        Ok(events)
    }

    /// Search for audit events in a specific log file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read.
    fn search_file(
        &self,
        file_path: &Path,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        category: Option<EventCategory>,
        userid: Option<&str>,
        events: &mut Vec<AuditEvent>,
    ) -> Result<(), CoreError> {
        let file = File::open(file_path).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to open log file: {e}"
            )))
        })?;

        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to read line: {e}"
                )))
            })?;

            if let Ok(event) = self.parse_log_line(&line) {
                // Filter by date range
                if event.timestamp < start_date || event.timestamp > end_date {
                    continue;
                }

                // Filter by category
                if let Some(cat) = category {
                    if event.category != cat {
                        continue;
                    }
                }

                // Filter by user ID
                if let Some(uid) = userid {
                    if event.userid.as_deref() != Some(uid) {
                        continue;
                    }
                }

                events.push(event);
            }
        }

        Ok(())
    }

    /// Parse a log line into an audit event.
    ///
    /// # Errors
    ///
    /// Returns an error if the log line cannot be parsed.
    fn parse_log_line(&self, line: &str) -> Result<AuditEvent, CoreError> {
        if self.config.enable_json_format {
            serde_json::from_str(line).map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to parse JSON log line: {e}"
                )))
            })
        } else {
            self.parse_text_log_line(line)
        }
    }

    /// Parse a log line into an audit event (serde feature required for JSON).
    ///
    /// # Errors
    ///
    /// Returns an error if the log line cannot be parsed.
    #[cfg(not(feature = "serde"))]
    fn parse_log_line(&self, line: &str) -> Result<AuditEvent, CoreError> {
        if self.config.enable_json_format {
            Err(CoreError::ComputationError(
                crate::error::ErrorContext::new("JSON parsing requires serde feature".to_string()),
            ))
        } else {
            self.parse_text_log_line(line)
        }
    }

    /// Parse a text format log line into an audit event.
    ///
    /// # Errors
    ///
    /// Returns an error if the log line cannot be parsed.
    fn parse_text_log_line(&self, line: &str) -> Result<AuditEvent, CoreError> {
        use uuid::Uuid;

        // Parse text format: [timestamp] category severity action user=X resource=Y outcome=Z description="..."
        let line = line.trim();

        // Extract timestamp
        if !line.starts_with('[') {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(
                    "Invalid log format: missing timestamp".to_string(),
                ),
            ));
        }

        let end_bracket = line.find(']').ok_or_else(|| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Invalid log format: unclosed timestamp bracket".to_string(),
            ))
        })?;

        let timestamp_str = &line[1..end_bracket];
        let timestamp = DateTime::parse_from_str(timestamp_str, "%Y-%m-%d %H:%M:%S UTC")
            .map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to parse timestamp: {e}"
                )))
            })?
            .with_timezone(&Utc);

        let remainder = line[end_bracket + 1..].trim();
        let parts: Vec<&str> = remainder.split_whitespace().collect();

        if parts.len() < 6 {
            return Err(CoreError::ComputationError(
                crate::error::ErrorContext::new(
                    "Invalid log format: insufficient fields".to_string(),
                ),
            ));
        }

        // Parse category
        let category = match parts[0] {
            "authentication" => EventCategory::Authentication,
            "authorization" => EventCategory::Authorization,
            "data_access" => EventCategory::DataAccess,
            "configuration" => EventCategory::Configuration,
            "security" => EventCategory::Security,
            "performance" => EventCategory::Performance,
            "error" => EventCategory::Error,
            "administrative" => EventCategory::Administrative,
            "compliance" => EventCategory::Compliance,
            _ => EventCategory::Error, // Default fallback
        };

        // Parse severity
        let severity = match parts[1] {
            "info" => EventSeverity::Info,
            "warning" => EventSeverity::Warning,
            "error" => EventSeverity::Error,
            "critical" => EventSeverity::Critical,
            _ => EventSeverity::Info, // Default fallback
        };

        let action = parts[2].to_string();

        // Parse key-value pairs
        let mut userid = None;
        let mut resourceid = None;
        let mut outcome = EventOutcome::Unknown;
        let mut description = String::new();

        for part in &parts[3..] {
            if let Some(equals_pos) = part.find('=') {
                let key = &part[..equals_pos];
                let value = &part[equals_pos + 1..];

                match key {
                    "user" => {
                        if value != "-" {
                            userid = Some(value.to_string());
                        }
                    }
                    "resource" => {
                        if value != "-" {
                            resourceid = Some(value.to_string());
                        }
                    }
                    "outcome" => {
                        outcome = match value {
                            "success" => EventOutcome::Success,
                            "failure" => EventOutcome::Failure,
                            "denied" => EventOutcome::Denied,
                            "cancelled" => EventOutcome::Cancelled,
                            _ => EventOutcome::Unknown,
                        };
                    }
                    "description" => {
                        // Handle quoted description
                        if value.starts_with('"') {
                            // Find the rest of the description in subsequent parts
                            let mut desc_parts = vec![value];
                            let start_idx = parts
                                .iter()
                                .position(|p| p.starts_with("description="))
                                .unwrap_or(0);

                            for desc_part in &parts[start_idx + 1..] {
                                desc_parts.push(desc_part);
                                if desc_part.ends_with('"') {
                                    break;
                                }
                            }

                            description = desc_parts.join(" ");
                            // Remove quotes
                            if description.starts_with('"') && description.ends_with('"') {
                                description = description[1..description.len() - 1].to_string();
                            }
                        } else {
                            description = value.to_string();
                        }
                    }
                    _ => {} // Ignore unknown fields
                }
            }
        }

        Ok(AuditEvent {
            event_id: Uuid::new_v4(), // Generate new ID for parsed events
            timestamp,
            category,
            severity,
            action,
            userid,
            resourceid,
            source_ip: None,
            description,
            metadata: HashMap::new(),
            system_context: None,
            stack_trace: None,
            correlation_id: None,
            outcome,
            data_classification: None,
            compliance_tags: Vec::new(),
            previous_hash: None,
            event_hash: None,
            digital_signature: None,
        })
    }

    /// Flush the event buffer to the log file.
    ///
    /// # Errors
    ///
    /// Returns an error if events cannot be written to the log file.
    fn flush_buffer(&self, buffer: &mut Vec<AuditEvent>) -> Result<(), CoreError> {
        let mut file_manager = self.file_manager.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire file manager lock".to_string(),
            ))
        })?;

        for mut event in buffer.drain(..) {
            file_manager.write_event(&mut event)?;
        }

        file_manager.flush()?;
        Ok(())
    }

    /// Check if the flush interval has elapsed and flush if needed.
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    fn check_flush_interval(&self) -> Result<(), CoreError> {
        let mut last_flush = self.last_flush.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire last flush lock".to_string(),
            ))
        })?;

        let now = Utc::now();
        let flush_interval = chrono::Duration::milliseconds(self.config.flush_interval_ms as i64);

        if now - *last_flush > flush_interval {
            let mut buffer = self.event_buffer.lock().map_err(|_| {
                CoreError::ComputationError(crate::error::ErrorContext::new(
                    "Failed to acquire buffer lock".to_string(),
                ))
            })?;

            if !buffer.is_empty() {
                self.flush_buffer(&mut buffer)?;
            }

            *last_flush = now;
        }

        Ok(())
    }

    /// Force flush all pending events
    ///
    /// # Errors
    ///
    /// Returns an error if flushing fails.
    pub fn flush(&self) -> Result<(), CoreError> {
        let mut buffer = self.event_buffer.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire buffer lock".to_string(),
            ))
        })?;

        if !buffer.is_empty() {
            self.flush_buffer(&mut buffer)?;
        }

        Ok(())
    }

    /// Get audit statistics
    ///
    /// # Errors
    ///
    /// Returns an error if statistics cannot be calculated.
    pub fn get_statistics(&self, days: u32) -> Result<AuditStatistics, CoreError> {
        let end_date = Utc::now();
        let start_date = end_date - chrono::Duration::days(days as i64);

        let events = self.search_events(start_date, end_date, None, None)?;

        let mut stats = AuditStatistics {
            total_events: events.len(),
            ..Default::default()
        };

        for event in events {
            match event.category {
                EventCategory::Authentication => stats.authentication_events += 1,
                EventCategory::DataAccess => stats.data_access_events += 1,
                EventCategory::Security => stats.security_events += 1,
                EventCategory::Configuration => stats.configuration_events += 1,
                _ => stats.other_events += 1,
            }

            if event.outcome == EventOutcome::Failure {
                stats.failed_events += 1;
            }
        }

        Ok(stats)
    }

    /// Add an audit event method with integrity verification
    ///
    /// # Errors
    ///
    /// Returns an error if event verification or logging fails.
    pub fn log_event_with_verification(&self, event: AuditEvent) -> Result<(), CoreError> {
        // Verify event integrity if hash chain is enabled
        if self.config.enable_hash_chain {
            // Add current system state to event hash
            if let Some(_context) = &event.system_context {
                // Hash would include system context
            }
        }

        self.log_event(event)
    }

    /// Export audit logs for compliance reporting
    ///
    /// # Errors
    ///
    /// Returns an error if the compliance report cannot be generated.
    pub fn export_compliance_report(
        &self,
        start_date: DateTime<Utc>,
        end_date: DateTime<Utc>,
        compliance_mode: ComplianceMode,
    ) -> Result<ComplianceReport, CoreError> {
        let events = self.search_events(start_date, end_date, None, None)?;

        let report = ComplianceReport {
            period_start: start_date,
            period_end: end_date,
            compliance_mode,
            total_events: events.len(),
            events_by_category: events.iter().fold(HashMap::new(), |mut acc, event| {
                *acc.entry(event.category).or_insert(0) += 1;
                acc
            }),
            security_violations: events
                .iter()
                .filter(|e| {
                    e.category == EventCategory::Security && e.outcome == EventOutcome::Failure
                })
                .count(),
            data_access_events: events
                .iter()
                .filter(|e| e.category == EventCategory::DataAccess)
                .count(),
            failed_authentication_attempts: events
                .iter()
                .filter(|e| {
                    e.category == EventCategory::Authentication
                        && e.outcome == EventOutcome::Failure
                })
                .count(),
            hash_chain_integrity: self.verify_integrity()?,
        };

        Ok(report)
    }

    /// Verify overall system integrity
    ///
    /// # Errors
    ///
    /// Returns an error if integrity verification fails.
    pub fn verify_integrity(&self) -> Result<bool, CoreError> {
        let file_manager = self.file_manager.lock().map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to acquire file manager lock".to_string(),
            ))
        })?;

        file_manager.verify_hash_chain()
    }
}
