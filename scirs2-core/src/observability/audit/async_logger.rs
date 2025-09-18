//! Async audit logger for high-throughput scenarios

use crate::error::CoreError;
use crate::observability::audit::builder::AuditEventBuilder;
use crate::observability::audit::logger::AuditLogger;
use crate::observability::audit::types::{AuditConfig, AuditEvent, EventCategory, SystemContext};
use std::sync::Arc;

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
                        eprintln!("Failed to log audit event: {e}");
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

        self.log_event(event).await
    }
}
