//! Types and configuration structures for the audit logging system

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use uuid::Uuid;

/// Audit logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub buffersize: usize,
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
            log_directory: PathBuf::from("./auditlogs"),
            max_file_size: 100 * 1024 * 1024, // 100MB
            max_files: 100,
            enable_encryption: true,
            enable_integrity_verification: true,
            alerting_config: None,
            buffersize: 1000,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct S3Credentials {
    /// AWS access key ID
    pub access_key: String,
    /// AWS secret access key
    pub secret_key: String,
    /// Optional session token for temporary credentials
    pub session_token: Option<String>,
}

/// Compliance modes for different regulatory requirements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub sessionid: Option<String>,
    /// Request ID for correlation
    pub requestid: Option<String>,
}

impl SystemContext {
    /// Create system context from current environment
    #[must_use]
    pub fn current() -> Self {
        use crate::observability::audit::utils::{get_hostname, get_local_ip, get_thread_id};

        Self {
            process_id: std::process::id(),
            thread_id: get_thread_id(),
            hostname: get_hostname(),
            ip_address: get_local_ip(),
            user_agent: None,
            sessionid: None,
            requestid: None,
        }
    }

    /// Set session ID
    #[must_use]
    pub fn with_sessionid(mut self, sessionid: String) -> Self {
        self.sessionid = Some(sessionid);
        self
    }

    /// Set request ID
    #[must_use]
    pub fn with_requestid(mut self, requestid: String) -> Self {
        self.requestid = Some(requestid);
        self
    }
}

/// Audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub userid: Option<String>,
    /// Resource identifier (data, file, endpoint, etc.)
    pub resourceid: Option<String>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Audit statistics structure
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
