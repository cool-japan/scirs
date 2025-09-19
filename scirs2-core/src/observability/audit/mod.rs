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

// Module declarations
pub mod alerts;
#[cfg(feature = "async")]
pub mod async_logger;
pub mod builder;
pub mod logger;
pub mod storage;
pub mod types;
pub mod utils;

// Re-export public types and traits for easy access
pub use alerts::AlertManager;
#[cfg(feature = "async")]
pub use async_logger::AsyncAuditLogger;
pub use builder::AuditEventBuilder;
pub use logger::AuditLogger;
pub use storage::LogFileManager;
pub use types::{
    AlertingConfig, AuditConfig, AuditEvent, AuditStatistics, ComplianceMode, ComplianceReport,
    DataClassification, EventCategory, EventOutcome, EventSeverity, RetentionPolicy,
    StorageBackend, SystemContext,
};

#[cfg(feature = "s3")]
pub use types::S3Credentials;

// Re-export utility functions that might be useful externally
pub use utils::{get_hostname, get_local_ip, get_stack_trace, get_thread_id};

// Tests module for integration tests
#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use tempfile::tempdir;

    #[test]
    fn test_audit_event_builder() {
        let event = AuditEventBuilder::new(EventCategory::DataAccess, "read")
            .userid("user123")
            .resourceid("dataset1")
            .severity(EventSeverity::Info)
            .description("Read operation")
            .metadata("size", "1000")
            .outcome(EventOutcome::Success)
            .build();

        assert_eq!(event.category, EventCategory::DataAccess);
        assert_eq!(event.action, "read");
        assert_eq!(event.userid, Some("user123".to_string()));
        assert_eq!(event.resourceid, Some("dataset1".to_string()));
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
            .userid("test_user")
            .outcome(EventOutcome::Success)
            .build();

        logger.log_event(event).expect("Failed to log event");
        logger.flush().expect("Failed to flush");
    }

    #[test]
    #[ignore] // Slow test - takes > 30s
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
    #[ignore] // Slow test - takes > 30s
    fn test_system_context() {
        let context = SystemContext::current()
            .with_sessionid("session123".to_string())
            .with_requestid("req456".to_string());

        assert_eq!(context.sessionid, Some("session123".to_string()));
        assert_eq!(context.requestid, Some("req456".to_string()));
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
