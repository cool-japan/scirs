//! Builder pattern for creating audit events

use crate::observability::audit::types::{
    AuditEvent, DataClassification, EventCategory, EventOutcome, EventSeverity, SystemContext,
};
use chrono::Utc;
use std::collections::HashMap;
use uuid::Uuid;

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
                userid: None,
                resourceid: None,
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
    pub fn userid(mut self, userid: &str) -> Self {
        self.event.userid = Some(userid.to_string());
        self
    }

    /// Set resource ID
    #[must_use]
    pub fn resourceid(mut self, resourceid: &str) -> Self {
        self.event.resourceid = Some(resourceid.to_string());
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
