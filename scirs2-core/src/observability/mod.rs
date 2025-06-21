//! # Observability Infrastructure
//!
//! Production-grade observability infrastructure for `SciRS2` Core providing
//! comprehensive monitoring, tracing, and auditing capabilities for enterprise
//! deployments and regulated environments.
//!
//! ## Modules
//!
//! - `tracing`: Distributed tracing system with OpenTelemetry integration
//! - `audit`: Audit logging for security events and compliance
//!
//! ## Features
//!
//! - OpenTelemetry-compatible distributed tracing
//! - Enterprise-grade audit logging
//! - Real-time security monitoring
//! - Compliance reporting capabilities
//! - Integration with SIEM systems
//! - Performance attribution and analysis
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::observability::{tracing, audit};
//!
//! // Initialize tracing
//! let tracing_config = tracing::TracingConfig::default();
//! tracing::init_tracing(tracing_config)?;
//!
//! // Initialize audit logging
//! let audit_config = audit::AuditConfig::default();
//! let audit_logger = audit::AuditLogger::new(audit_config)?;
//!
//! // Use tracing
//! if let Some(tracer) = tracing::global_tracer() {
//!     let span = tracer.start_span("data_processing")?;
//!     span.in_span(|| {
//!         // Your code here
//!     });
//! }
//!
//! // Log audit events
//! audit_logger.log_data_access("user123", "dataset", "read", None)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod audit;
pub mod tracing;

// Re-export main types for convenience
pub use tracing::{
    example_matrix_computation_with_tracing, ActiveSpan, AdaptiveSampler, BatchExporter,
    ConsoleExporter, EnhancedSpanMetrics, NegotiationResult, ProbabilitySampler,
    RateLimitingSampler, ResourceAttribution, SpanBuilder, SpanKind, SpanStatus, TraceContext,
    TracingConfig, TracingSystem, TracingVersion,
};

#[cfg(feature = "observability")]
pub use tracing::integrate_with_metrics_system;

pub use audit::{
    AlertingConfig, AuditConfig, AuditEvent, AuditEventBuilder, AuditLogger, AuditStatistics,
    ComplianceMode, ComplianceReport, DataClassification, EventCategory, EventOutcome,
    EventSeverity, RetentionPolicy, StorageBackend, SystemContext,
};

#[cfg(feature = "s3")]
pub use audit::S3Credentials;

#[cfg(feature = "async")]
pub use audit::AsyncAuditLogger;
