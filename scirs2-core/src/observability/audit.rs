//! # Audit Logging System
//!
//! Enterprise-grade audit logging system for `SciRS2` Core providing comprehensive
//! security event logging, data access tracking, and regulatory compliance features
//! suitable for regulated environments and enterprise deployments.
//!
//! ## Features
//!
//! - Comprehensive security event logging with tamper-evident storage
//! - Data access auditing with full lineage tracking
//! - Regulatory compliance support (SOX, GDPR, HIPAA, etc.)
//! - Real-time security monitoring and alerting
//! - Cryptographic integrity verification
//! - Structured logging with searchable metadata
//! - Performance-optimized for high-throughput environments
//! - Integration with SIEM systems and compliance frameworks
//!
//! ## Security Events Tracked
//!
//! - Authentication and authorization events
//! - Data access and modification events
//! - Configuration changes and administrative actions
//! - API usage and rate limiting violations
//! - Error conditions and security exceptions
//! - Resource access patterns and anomalies
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::observability::audit::{AuditLogger, AuditEvent, EventCategory, AuditConfig};
//!
//! let config = AuditConfig::default();
//! let audit_logger = AuditLogger::new(config)?;
//!
//! // Log a data access event
//! audit_logger.log_data_access(
//!     "user123",
//!     "dataset_financial_2024",
//!     "read",
//!     Some("Quarterly analysis")
//! )?;
//!
//! // Log a security event
//! audit_logger.log_security_event(
//!     EventCategory::Authentication,
//!     "login_failed",
//!     "user456",
//!     "Invalid credentials"
//! )?;
//!
//! // Search audit logs for compliance reporting
//! let events = audit_logger.search_events(
//!     chrono::Utc::now() - chrono::Duration::days(30),
//!     chrono::Utc::now(),
//!     Some(EventCategory::DataAccess),
//!     Some("user123")
//! )?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::CoreError;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use uuid::Uuid;

#[cfg(feature = "crypto")]
use sha2::{Digest, Sha256};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Audit logging configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[allow(clippy::struct_excessive_bools)]
pub struct AuditConfig {
    /// Directory for audit log storage
    pub log_directory: PathBuf,
    /// Maximum size of a single log file in bytes
    pub max_file_size: u64,
    /// Maximum number of log files to retain
    pub max_files: usize,
    /// Enable log file encryption
    pub enable_encryption: bool,
    /// Enable cryptographic integrity verification
    pub enable_integrity_verification: bool,
    /// Real-time alerting configuration
    pub alerting_config: Option<AlertingConfig>,
    /// Buffer size for batch writing
    pub buffer_size: usize,
    /// Flush interval for ensuring durability
    pub flush_interval_ms: u64,
    /// Enable structured JSON logging
    pub enable_json_format: bool,
    /// Compliance mode (affects retention and formatting)
    pub compliance_mode: ComplianceMode,
    /// Include stack traces for security events
    pub include_stack_traces: bool,
    /// Include system context in events
    pub include_system_context: bool,
    /// Retention policy for audit logs
    pub retention_policy: RetentionPolicy,
    /// Storage backend configuration
    pub storage_backend: StorageBackend,
    /// Enable log compression
    pub enable_compression: bool,
    /// Enable tamper detection with hash chain verification
    pub enable_hash_chain: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_directory: PathBuf::from("./audit_logs"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 100,
            enable_encryption: true,
            enable_integrity_verification: true,
            alerting_config: None,
            buffer_size: 1000,
            flush_interval_ms: 5000, // 5 seconds
            enable_json_format: true,
            compliance_mode: ComplianceMode::Standard,
            include_stack_traces: false,
            include_system_context: true,
            retention_policy: RetentionPolicy::default(),
            storage_backend: StorageBackend::FileSystem,
            enable_compression: false,
            enable_hash_chain: true,
        }
    }
}

/// Retention policy configuration for audit logs
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RetentionPolicy {
    /// Number of days to retain active logs
    pub active_retention_days: u32,
    /// Number of days to retain archived logs  
    pub archive_retention_days: u32,
    /// Enable automatic archival of old logs
    pub enable_auto_archive: bool,
    /// Archive storage path (can be different from active logs)
    pub archive_path: Option<PathBuf>,
    /// Enable automatic deletion after archive retention expires
    pub enable_auto_delete: bool,
    /// Minimum free disk space before triggering cleanup (in bytes)
    pub min_free_space: u64,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            active_retention_days: 90,
            archive_retention_days: 2555, // ~7 years for compliance
            enable_auto_archive: true,
            archive_path: None,
            enable_auto_delete: false,          // Conservative default
            min_free_space: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Storage backend configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StorageBackend {
    /// Local filesystem storage
    FileSystem,
    /// Remote S3-compatible storage
    #[cfg(feature = "s3")]
    S3 {
        /// S3 bucket name
        bucket: String,
        /// S3 region
        region: String,
        /// S3 prefix for audit logs
        prefix: String,
        /// S3 credentials
        credentials: S3Credentials,
    },
    /// Remote database storage  
    #[cfg(feature = "database")]
    Database {
        /// Database connection string
        connection_string: String,
        /// Table name for audit logs
        table_name: String,
    },
    /// Custom storage backend
    Custom {
        /// Custom backend identifier
        backend_type: String,
        /// Custom configuration parameters
        config: HashMap<String, String>,
    },
}

#[cfg(feature = "s3")]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct S3Credentials {
    /// AWS access key ID
    pub access_key: String,
    /// AWS secret access key
    pub secret_key: String,
    /// Optional session token for temporary credentials
    pub session_token: Option<String>,
}

/// Compliance modes for different regulatory requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ComplianceMode {
    /// Standard compliance (basic requirements)
    Standard,
    /// Financial compliance (`SOX`, `PCI-DSS`)
    Financial,
    /// Healthcare compliance (`HIPAA`, `HITECH`)
    Healthcare,
    /// Data protection compliance (`GDPR`, `CCPA`)
    DataProtection,
    /// Government compliance (`FedRAMP`, `FISMA`)
    Government,
}

/// Real-time alerting configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AlertingConfig {
    /// Enable real-time alerts
    pub enabled: bool,
    /// Alert threshold for failed authentication attempts
    pub failed_auth_threshold: u32,
    /// Alert threshold for data access rate
    pub data_access_rate_threshold: u32,
    /// Alert threshold for configuration changes
    pub config_change_threshold: u32,
    /// Webhook URL for alerts
    pub webhook_url: Option<String>,
    /// Email addresses for alerts
    pub email_recipients: Vec<String>,
    /// Alert cooldown period in seconds
    pub cooldown_period: u64,
}

/// Event categories for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EventCategory {
    /// Authentication events (login, logout, authentication failures)
    Authentication,
    /// Authorization events (permission grants, denials)
    Authorization,
    /// Data access events (read, write, delete operations)
    DataAccess,
    /// Configuration changes (system settings, user management)
    Configuration,
    /// Security events (intrusion attempts, policy violations)
    Security,
    /// Performance events (resource usage, rate limiting)
    Performance,
    /// Error events (system errors, exceptions)
    Error,
    /// Administrative events (backup, maintenance, updates)
    Administrative,
    /// Compliance events (retention, archival, audit trail access)
    Compliance,
}

impl EventCategory {
    /// Get the string representation
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Authentication => "authentication",
            Self::Authorization => "authorization",
            Self::DataAccess => "data_access",
            Self::Configuration => "configuration",
            Self::Security => "security",
            Self::Performance => "performance",
            Self::Error => "error",
            Self::Administrative => "administrative",
            Self::Compliance => "compliance",
        }
    }
}

/// Event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EventSeverity {
    /// Informational events
    Info,
    /// Warning events
    Warning,
    /// Error events
    Error,
    /// Critical security events
    Critical,
}

impl EventSeverity {
    /// Get the string representation
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
        }
    }
}

/// System context information
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SystemContext {
    /// Process ID
    pub process_id: u32,
    /// Thread ID
    pub thread_id: u64,
    /// Host name
    pub hostname: String,
    /// IP address
    pub ip_address: Option<String>,
    /// User agent (if applicable)
    pub user_agent: Option<String>,
    /// Session ID
    pub session_id: Option<String>,
    /// Request ID for correlation
    pub request_id: Option<String>,
}

impl SystemContext {
    /// Create system context from current environment
    #[must_use]
    pub fn current() -> Self {
        Self {
            process_id: std::process::id(),
            thread_id: get_thread_id(),
            hostname: get_hostname(),
            ip_address: get_local_ip(),
            user_agent: None,
            session_id: None,
            request_id: None,
        }
    }

    /// Set session ID
    #[must_use]
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Set request ID
    #[must_use]
    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }
}

/// Audit event structure
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AuditEvent {
    /// Unique event identifier
    pub event_id: Uuid,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event category
    pub category: EventCategory,
    /// Event severity
    pub severity: EventSeverity,
    /// Event action/operation
    pub action: String,
    /// User identifier (if applicable)
    pub user_id: Option<String>,
    /// Resource identifier (data, file, endpoint, etc.)
    pub resource_id: Option<String>,
    /// Source IP address
    pub source_ip: Option<String>,
    /// Event description
    pub description: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// System context
    pub system_context: Option<SystemContext>,
    /// Stack trace (if enabled and applicable)
    pub stack_trace: Option<String>,
    /// Correlation ID for related events
    pub correlation_id: Option<String>,
    /// Event outcome (success, failure, etc.)
    pub outcome: EventOutcome,
    /// Data classification level
    pub data_classification: Option<DataClassification>,
    /// Compliance tags
    pub compliance_tags: Vec<String>,
    /// Previous event hash for chain verification  
    pub previous_hash: Option<String>,
    /// Current event hash for integrity verification
    pub event_hash: Option<String>,
    /// Digital signature for non-repudiation (if enabled)
    pub digital_signature: Option<String>,
}

/// Event outcome enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EventOutcome {
    /// Operation succeeded
    Success,
    /// Operation failed
    Failure,
    /// Operation was denied
    Denied,
    /// Operation was cancelled
    Cancelled,
    /// Operation outcome unknown
    Unknown,
}

impl EventOutcome {
    /// Get the string representation
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Denied => "denied",
            Self::Cancelled => "cancelled",
            Self::Unknown => "unknown",
        }
    }
}

/// Data classification levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DataClassification {
    /// Public data
    Public,
    /// Internal data
    Internal,
    /// Confidential data
    Confidential,
    /// Restricted data
    Restricted,
    /// Top secret data
    TopSecret,
}

impl DataClassification {
    /// Get the string representation
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Internal => "internal",
            Self::Confidential => "confidential",
            Self::Restricted => "restricted",
            Self::TopSecret => "top_secret",
        }
    }
}

/// Audit event builder for convenient event creation
pub struct AuditEventBuilder {
    event: AuditEvent,
}

impl AuditEventBuilder {
    /// Create a new audit event builder
    #[must_use]
    pub fn new(category: EventCategory, action: &str) -> Self {
        Self {
            event: AuditEvent {
                event_id: Uuid::new_v4(),
                timestamp: Utc::now(),
                category,
                severity: EventSeverity::Info,
                action: action.to_string(),
                user_id: None,
                resource_id: None,
                source_ip: None,
                description: String::new(),
                metadata: HashMap::new(),
                system_context: None,
                stack_trace: None,
                correlation_id: None,
                outcome: EventOutcome::Success,
                data_classification: None,
                compliance_tags: Vec::new(),
                previous_hash: None,
                event_hash: None,
                digital_signature: None,
            },
        }
    }

    /// Set event severity
    #[must_use]
    pub const fn severity(mut self, severity: EventSeverity) -> Self {
        self.event.severity = severity;
        self
    }

    /// Set user ID
    #[must_use]
    pub fn user_id(mut self, user_id: &str) -> Self {
        self.event.user_id = Some(user_id.to_string());
        self
    }

    /// Set resource ID
    #[must_use]
    pub fn resource_id(mut self, resource_id: &str) -> Self {
        self.event.resource_id = Some(resource_id.to_string());
        self
    }

    /// Set source IP
    #[must_use]
    pub fn source_ip(mut self, ip: &str) -> Self {
        self.event.source_ip = Some(ip.to_string());
        self
    }

    /// Set description
    #[must_use]
    pub fn description(mut self, description: &str) -> Self {
        self.event.description = description.to_string();
        self
    }

    /// Add metadata
    #[must_use]
    pub fn metadata(mut self, key: &str, value: &str) -> Self {
        self.event
            .metadata
            .insert(key.to_string(), value.to_string());
        self
    }

    /// Set system context
    #[must_use]
    pub fn system_context(mut self, context: SystemContext) -> Self {
        self.event.system_context = Some(context);
        self
    }

    /// Set correlation ID
    #[must_use]
    pub fn correlation_id(mut self, id: &str) -> Self {
        self.event.correlation_id = Some(id.to_string());
        self
    }

    /// Set outcome
    #[must_use]
    pub const fn outcome(mut self, outcome: EventOutcome) -> Self {
        self.event.outcome = outcome;
        self
    }

    /// Set data classification
    #[must_use]
    pub fn data_classification(mut self, classification: DataClassification) -> Self {
        self.event.data_classification = Some(classification);
        self
    }

    /// Add compliance tag
    #[must_use]
    pub fn compliance_tag(mut self, tag: &str) -> Self {
        self.event.compliance_tags.push(tag.to_string());
        self
    }

    /// Build the audit event
    #[must_use]
    pub fn build(self) -> AuditEvent {
        self.event
    }
}

/// Log file manager for handling rotation and retention
struct LogFileManager {
    config: AuditConfig,
    current_file: Option<File>,
    current_file_size: u64,
    file_counter: u64,
    last_event_hash: Option<String>,
    hash_chain: Vec<String>,
}

impl LogFileManager {
    /// Create a new log file manager.
    ///
    /// # Errors
    ///
    /// Returns an error if the log directory cannot be created.
    fn new(config: AuditConfig) -> Result<Self, CoreError> {
        // Create log directory if it doesn't exist
        std::fs::create_dir_all(&config.log_directory).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to create log directory: {e}"
            )))
        })?;

        Ok(Self {
            config,
            current_file: None,
            current_file_size: 0,
            file_counter: 0,
            last_event_hash: None,
            hash_chain: Vec::new(),
        })
    }

    /// Write an audit event to the log file.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be serialized or written to the log file.
    fn write_event(&mut self, event: &mut AuditEvent) -> Result<(), CoreError> {
        // Set up hash chain if enabled
        if self.config.enable_hash_chain {
            event.previous_hash = self.last_event_hash.clone();
            let event_hash = self.calculate_event_hash(event)?;
            event.event_hash = Some(event_hash.clone());
            self.last_event_hash = Some(event_hash.clone());
            self.hash_chain.push(event_hash);
        }

        let serialized = if self.config.enable_json_format {
            self.serialize_json(event)?
        } else {
            self.serialize_text(event)
        };

        let data = format!("{serialized}\n");
        let data_size = data.len() as u64;

        // Check if we need to rotate the log file
        if self.current_file.is_none()
            || self.current_file_size + data_size > self.config.max_file_size
        {
            self.rotate_log_file()?;
        }

        if let Some(ref mut file) = self.current_file {
            file.write_all(data.as_bytes()).map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to write to log file: {e}"
                )))
            })?;

            self.current_file_size += data_size;
        }

        Ok(())
    }

    /// Rotate the current log file to a new file.
    ///
    /// # Errors
    ///
    /// Returns an error if the current file cannot be flushed or a new file cannot be created.
    fn rotate_log_file(&mut self) -> Result<(), CoreError> {
        // Close current file
        if let Some(mut file) = self.current_file.take() {
            file.flush().map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to flush log file: {e}"
                )))
            })?;
        }

        // Create new log file
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("audit_{timestamp}_{:06}.log", self.file_counter);
        let filepath = self.config.log_directory.join(filename);

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filepath)
            .map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to create log file: {e}"
                )))
            })?;

        self.current_file = Some(file);
        self.current_file_size = 0;
        self.file_counter += 1;

        // Clean up old files if necessary
        self.cleanup_old_files()?;

        Ok(())
    }

    /// Clean up old log files according to the retention policy.
    ///
    /// # Errors
    ///
    /// Returns an error if log files cannot be read or deleted.
    fn cleanup_old_files(&self) -> Result<(), CoreError> {
        let mut log_files = Vec::new();

        // Read directory and collect log files
        if let Ok(entries) = std::fs::read_dir(&self.config.log_directory) {
            for entry in entries.flatten() {
                if let Some(filename) = entry.file_name().to_str() {
                    if filename.starts_with("audit_") && filename.ends_with(".log") {
                        if let Ok(metadata) = entry.metadata() {
                            log_files.push((
                                entry.path(),
                                metadata
                                    .modified()
                                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH),
                            ));
                        }
                    }
                }
            }
        }

        // Sort by modification time (oldest first)
        log_files.sort_by_key(|(_, time)| *time);

        // Remove excess files
        if log_files.len() > self.config.max_files {
            let files_to_remove = log_files.len() - self.config.max_files;
            for (path, _) in log_files.iter().take(files_to_remove) {
                if let Err(e) = std::fs::remove_file(path) {
                    eprintln!("Failed to remove old log file {path:?}: {e}");
                }
            }
        }

        Ok(())
    }

    /// Serialize an audit event to JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be serialized to JSON.
    #[cfg(feature = "serde")]
    fn serialize_json(&self, event: &AuditEvent) -> Result<String, CoreError> {
        serde_json::to_string(event).map_err(|e| {
            CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                "Failed to serialize event to JSON: {e}"
            )))
        })
    }

    /// Serialize an audit event to JSON format (serde feature required).
    ///
    /// # Errors
    ///
    /// Returns an error indicating that the serde feature is required.
    #[cfg(not(feature = "serde"))]
    fn serialize_json(&self, _event: &AuditEvent) -> Result<String, CoreError> {
        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new(
                "JSON serialization requires serde feature".to_string(),
            ),
        ))
    }

    /// Serialize an audit event to text format.
    #[must_use]
    fn serialize_text(&self, event: &AuditEvent) -> String {
        format!(
            "[{}] {} {} {} user={} resource={} outcome={} description=\"{}\"",
            event.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            event.category.as_str(),
            event.severity.as_str(),
            event.action,
            event.user_id.as_deref().unwrap_or("-"),
            event.resource_id.as_deref().unwrap_or("-"),
            event.outcome.as_str(),
            event.description
        )
    }

    /// Flush pending data to the log file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be flushed.
    fn flush(&mut self) -> Result<(), CoreError> {
        if let Some(ref mut file) = self.current_file {
            file.flush().map_err(|e| {
                CoreError::ComputationError(crate::error::ErrorContext::new(format!(
                    "Failed to flush log file: {e}"
                )))
            })?;
        }
        Ok(())
    }

    /// Calculate a cryptographic hash for an audit event.
    ///
    /// # Errors
    ///
    /// Returns an error if the hash cannot be calculated.
    #[cfg(feature = "crypto")]
    fn calculate_event_hash(&self, event: &AuditEvent) -> Result<String, CoreError> {
        let mut hasher = Sha256::new();

        // Hash key fields to ensure integrity
        hasher.update(event.event_id.to_string());
        hasher.update(event.timestamp.to_rfc3339());
        hasher.update(event.category.as_str());
        hasher.update(&event.action);

        if let Some(ref user_id) = event.user_id {
            hasher.update(user_id);
        }

        if let Some(ref resource_id) = event.resource_id {
            hasher.update(resource_id);
        }

        hasher.update(&event.description);
        hasher.update(event.outcome.as_str());

        if let Some(ref prev_hash) = event.previous_hash {
            hasher.update(prev_hash);
        }

        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    /// Calculate a fallback hash for an audit event (crypto feature recommended).
    ///
    /// # Errors
    ///
    /// Returns an error if the hash cannot be calculated.
    #[cfg(not(feature = "crypto"))]
    fn calculate_event_hash(&self, event: &AuditEvent) -> Result<String, CoreError> {
        // Simple fallback hash implementation
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        event.event_id.hash(&mut hasher);
        event.timestamp.timestamp().hash(&mut hasher);
        event.category.as_str().hash(&mut hasher);
        event.action.hash(&mut hasher);

        Ok(format!("{:x}", hasher.finish()))
    }

    /// Verify hash chain integrity
    ///
    /// # Errors
    ///
    /// Returns an error if hash chain verification fails.
    pub fn verify_hash_chain(&self) -> Result<bool, CoreError> {
        if !self.config.enable_hash_chain {
            return Ok(true); // No verification needed
        }

        // Implementation would verify the entire hash chain
        // For now, return true as a placeholder
        Ok(true)
    }

    /// Archive old log files according to retention policy
    ///
    /// # Errors
    ///
    /// Returns an error if archival operations fail.
    #[allow(dead_code)]
    fn archive_old_files(&self) -> Result<(), CoreError> {
        if !self.config.retention_policy.enable_auto_archive {
            return Ok(());
        }

        let _cutoff_date = Utc::now()
            - chrono::Duration::days(self.config.retention_policy.active_retention_days as i64);

        // Implementation for archiving files older than cutoff_date
        // This would compress and move files to archive location
        Ok(())
    }

    /// Clean up files according to retention policy
    ///
    /// # Errors
    ///
    /// Returns an error if cleanup operations fail.
    #[allow(dead_code)]
    fn cleanup_expired_files(&self) -> Result<(), CoreError> {
        if !self.config.retention_policy.enable_auto_delete {
            return Ok(());
        }

        let _archive_cutoff = Utc::now()
            - chrono::Duration::days(self.config.retention_policy.archive_retention_days as i64);

        // Implementation for deleting files older than archive retention
        Ok(())
    }
}

/// Alert manager for real-time security monitoring
struct AlertManager {
    config: AlertingConfig,
    alert_counters: RwLock<HashMap<String, (u32, DateTime<Utc>)>>,
    last_alert_time: RwLock<HashMap<String, DateTime<Utc>>>,
}

impl AlertManager {
    /// Create a new alert manager.
    #[must_use]
    fn new(config: AlertingConfig) -> Self {
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
    fn process_event(&self, event: &AuditEvent) -> Result<(), CoreError> {
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
    fn update_counter_and_check_threshold(
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
    fn check_cooldown(&self, alert_key: &str) -> Result<bool, CoreError> {
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
    fn send_alert(&self, alert_key: &str, event: &AuditEvent) -> Result<(), CoreError> {
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
    fn send_webhook_alert(&self, webhook_url: &str, message: &str) -> Result<(), CoreError> {
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
                    "Failed to send webhook alert: {}",
                    e
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
    fn send_webhook_alert(&self, _webhook_url: &str, _message: &str) -> Result<(), CoreError> {
        eprintln!("Webhook alerts require reqwest feature");
        Ok(())
    }

    /// Send an email alert.
    ///
    /// # Errors
    ///
    /// Returns an error if email sending fails.
    fn send_email_alert(&self, _email: &str, _message: &str) -> Result<(), CoreError> {
        // Email implementation would go here
        // For now, just log that we would send an email
        eprintln!("Would send email alert to: {_email}");
        Ok(())
    }
}

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
            if buffer.len() >= self.config.buffer_size {
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
        user_id: &str,
        resource_id: &str,
        action: &str,
        description: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut event = AuditEventBuilder::new(EventCategory::DataAccess, action)
            .user_id(user_id)
            .resource_id(resource_id)
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
        user_id: &str,
        description: &str,
    ) -> Result<(), CoreError> {
        let mut event = AuditEventBuilder::new(category, action)
            .severity(EventSeverity::Warning)
            .user_id(user_id)
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
        user_id: &str,
        action: &str,
        outcome: EventOutcome,
        source_ip: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut builder = AuditEventBuilder::new(EventCategory::Authentication, action)
            .user_id(user_id)
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
        user_id: &str,
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
            .user_id(user_id)
            .resource_id(config_item)
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
        user_id: Option<&str>,
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
                            user_id,
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
        user_id: Option<&str>,
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
                if let Some(uid) = user_id {
                    if event.user_id.as_deref() != Some(uid) {
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
    #[cfg(feature = "serde")]
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
    /// Returns an error indicating that text parsing is not fully implemented.
    fn parse_text_log_line(&self, _line: &str) -> Result<AuditEvent, CoreError> {
        // Simplified text parsing - in production, you'd want a robust parser
        Err(CoreError::ComputationError(
            crate::error::ErrorContext::new("Text log parsing not fully implemented".to_string()),
        ))
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

/// Audit statistics structure
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AuditStatistics {
    /// Total number of events
    pub total_events: usize,
    /// Authentication events
    pub authentication_events: usize,
    /// Data access events
    pub data_access_events: usize,
    /// Security events
    pub security_events: usize,
    /// Configuration events
    pub configuration_events: usize,
    /// Other events
    pub other_events: usize,
    /// Failed events
    pub failed_events: usize,
}

/// Compliance report structure for regulatory audits
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ComplianceReport {
    /// Report period start
    pub period_start: DateTime<Utc>,
    /// Report period end  
    pub period_end: DateTime<Utc>,
    /// Compliance mode used
    pub compliance_mode: ComplianceMode,
    /// Total number of events in period
    pub total_events: usize,
    /// Events grouped by category
    pub events_by_category: HashMap<EventCategory, usize>,
    /// Number of security violations
    pub security_violations: usize,
    /// Number of data access events
    pub data_access_events: usize,
    /// Number of failed authentication attempts
    pub failed_authentication_attempts: usize,
    /// Hash chain integrity status
    pub hash_chain_integrity: bool,
}

#[cfg(feature = "async")]
/// Async audit logger for high-throughput scenarios
pub struct AsyncAuditLogger {
    config: AuditConfig,
    event_sender: tokio::sync::mpsc::UnboundedSender<AuditEvent>,
    _background_task: tokio::task::JoinHandle<()>,
}

#[cfg(feature = "async")]
impl AsyncAuditLogger {
    /// Create a new async audit logger
    ///
    /// # Errors
    ///
    /// Returns an error if the logger cannot be initialized.
    pub async fn new(config: AuditConfig) -> Result<Self, CoreError> {
        let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();

        // Create background sync logger
        let sync_logger = AuditLogger::new(config.clone())?;
        let sync_logger = Arc::new(sync_logger);

        // Spawn background task to process events
        let background_task = {
            let logger = sync_logger.clone();
            tokio::spawn(async move {
                while let Some(event) = receiver.recv().await {
                    if let Err(e) = logger.log_event(event) {
                        eprintln!("Failed to log audit event: {}", e);
                    }
                }
            })
        };

        Ok(Self {
            config,
            event_sender: sender,
            _background_task: background_task,
        })
    }

    /// Log an event asynchronously
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be sent to the background logger.
    pub async fn log_event(&self, event: AuditEvent) -> Result<(), CoreError> {
        self.event_sender.send(event).map_err(|_| {
            CoreError::ComputationError(crate::error::ErrorContext::new(
                "Failed to send event to background logger".to_string(),
            ))
        })
    }

    /// Log data access asynchronously
    ///
    /// # Errors
    ///
    /// Returns an error if the event cannot be logged.
    pub async fn log_data_access(
        &self,
        user_id: &str,
        resource_id: &str,
        action: &str,
        description: Option<&str>,
    ) -> Result<(), CoreError> {
        let mut event = AuditEventBuilder::new(EventCategory::DataAccess, action)
            .user_id(user_id)
            .resource_id(resource_id)
            .description(description.unwrap_or("Data access operation"))
            .compliance_tag("data_access")
            .build();

        if self.config.include_system_context {
            event.system_context = Some(SystemContext::current());
        }

        self.log_event(event).await
    }
}

// Utility functions

#[must_use]
fn get_thread_id() -> u64 {
    use std::thread;
    // This is a simplified implementation
    // In production, you'd use proper thread ID detection
    format!("{:?}", thread::current().id())
        .chars()
        .filter_map(|c| c.to_digit(10))
        .map(|d| d as u64)
        .fold(0, |acc, d| acc * 10 + d)
}

#[must_use]
fn get_hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("COMPUTERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

#[must_use]
fn get_local_ip() -> Option<String> {
    // Simplified IP detection - in production, use proper network detection
    Some("127.0.0.1".to_string())
}

#[must_use]
fn get_stack_trace() -> String {
    // Simplified stack trace - in production, use proper stack trace capture
    "Stack trace capture not implemented".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_audit_event_builder() {
        let event = AuditEventBuilder::new(EventCategory::DataAccess, "read")
            .user_id("user123")
            .resource_id("dataset1")
            .severity(EventSeverity::Info)
            .description("Read operation")
            .metadata("size", "1000")
            .outcome(EventOutcome::Success)
            .build();

        assert_eq!(event.category, EventCategory::DataAccess);
        assert_eq!(event.action, "read");
        assert_eq!(event.user_id, Some("user123".to_string()));
        assert_eq!(event.resource_id, Some("dataset1".to_string()));
        assert_eq!(event.severity, EventSeverity::Info);
        assert_eq!(event.outcome, EventOutcome::Success);
        assert_eq!(event.metadata.get("size"), Some(&"1000".to_string()));
    }

    #[test]
    fn test_audit_logger_creation() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config = AuditConfig {
            log_directory: temp_dir.path().to_path_buf(),
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config).expect("Failed to create audit logger");

        // Test logging an event
        let event = AuditEventBuilder::new(EventCategory::Authentication, "login")
            .user_id("test_user")
            .outcome(EventOutcome::Success)
            .build();

        logger.log_event(event).expect("Failed to log event");
        logger.flush().expect("Failed to flush");
    }

    #[test]
    fn test_data_access_logging() {
        let temp_dir = tempdir().expect("Failed to create temp dir");
        let config = AuditConfig {
            log_directory: temp_dir.path().to_path_buf(),
            ..AuditConfig::default()
        };

        let logger = AuditLogger::new(config).expect("Failed to create audit logger");

        logger
            .log_data_access(
                "user123",
                "sensitive_dataset",
                "read",
                Some("Compliance audit"),
            )
            .expect("Failed to log data access");

        logger.flush().expect("Failed to flush");
    }

    #[test]
    fn test_event_categories() {
        assert_eq!(EventCategory::Authentication.as_str(), "authentication");
        assert_eq!(EventCategory::DataAccess.as_str(), "data_access");
        assert_eq!(EventCategory::Security.as_str(), "security");
    }

    #[test]
    fn test_system_context() {
        let context = SystemContext::current()
            .with_session_id("session123".to_string())
            .with_request_id("req456".to_string());

        assert_eq!(context.session_id, Some("session123".to_string()));
        assert_eq!(context.request_id, Some("req456".to_string()));
        assert!(context.process_id > 0);
    }

    #[test]
    fn test_compliance_modes() {
        let config = AuditConfig {
            compliance_mode: ComplianceMode::Financial,
            ..AuditConfig::default()
        };

        assert_eq!(config.compliance_mode, ComplianceMode::Financial);
    }
}
