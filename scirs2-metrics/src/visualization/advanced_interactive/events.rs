//! Event system for interactive visualization
//!
//! This module provides event handling, routing, and management for
//! interactive dashboard components.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::widgets::{WidgetEvent, WidgetEventResponse};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Event system for managing dashboard interactions
#[derive(Debug)]
pub struct EventSystem {
    /// Event handlers
    handlers: HashMap<String, Vec<Box<dyn EventHandler + Send + Sync>>>,
    /// Event queue
    event_queue: VecDeque<DashboardEvent>,
    /// Event history
    event_history: VecDeque<DashboardEvent>,
    /// Configuration
    config: EventSystemConfig,
}

/// Event handler trait
pub trait EventHandler: std::fmt::Debug + Send + Sync {
    /// Handle event
    fn handle_event(&self, event: &DashboardEvent) -> Result<Option<EventResponse>>;

    /// Get handler priority
    fn priority(&self) -> u32;

    /// Check if handler can handle event type
    fn can_handle(&self, event_type: &str) -> bool;
}

/// Dashboard event
#[derive(Debug, Clone)]
pub struct DashboardEvent {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: String,
    /// Source widget or component
    pub source: String,
    /// Target widget or component
    pub target: Option<String>,
    /// Event timestamp
    pub timestamp: Instant,
    /// Event data
    pub data: HashMap<String, Value>,
    /// Event metadata
    pub metadata: EventMetadata,
}

/// Event metadata
#[derive(Debug, Clone)]
pub struct EventMetadata {
    /// User session ID
    pub session_id: Option<String>,
    /// User ID
    pub user_id: Option<String>,
    /// Event priority
    pub priority: EventPriority,
    /// Propagation settings
    pub propagation: PropagationSettings,
    /// Context information
    pub context: HashMap<String, Value>,
}

/// Event priority enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Propagation settings
#[derive(Debug, Clone)]
pub struct PropagationSettings {
    /// Stop propagation
    pub stop_propagation: bool,
    /// Prevent default
    pub prevent_default: bool,
    /// Bubble up
    pub bubble: bool,
    /// Capture phase
    pub capture: bool,
}

/// Event response
#[derive(Debug, Clone)]
pub struct EventResponse {
    /// Response ID
    pub id: String,
    /// Actions to perform
    pub actions: Vec<EventAction>,
    /// Response data
    pub data: HashMap<String, Value>,
    /// Should stop propagation
    pub stop_propagation: bool,
}

/// Event action enumeration
#[derive(Debug, Clone)]
pub enum EventAction {
    /// Update widget
    UpdateWidget {
        widget_id: String,
        updates: HashMap<String, Value>,
    },
    /// Trigger new event
    TriggerEvent(DashboardEvent),
    /// Execute script
    ExecuteScript {
        script: String,
        context: HashMap<String, Value>,
    },
    /// Send notification
    SendNotification {
        message: String,
        level: NotificationLevel,
    },
    /// Update data source
    UpdateDataSource { source_id: String, data: Value },
    /// Custom action
    Custom {
        action_type: String,
        parameters: HashMap<String, Value>,
    },
}

/// Notification level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationLevel {
    /// Info notification
    Info,
    /// Success notification
    Success,
    /// Warning notification
    Warning,
    /// Error notification
    Error,
}

/// Event system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSystemConfig {
    /// Maximum event queue size
    pub max_queue_size: usize,
    /// Maximum history size
    pub max_history_size: usize,
    /// Event processing batch size
    pub batch_size: usize,
    /// Enable event debugging
    pub debug_enabled: bool,
    /// Performance monitoring
    pub performance_monitoring: bool,
}

impl EventSystem {
    /// Create new event system
    pub fn new(config: EventSystemConfig) -> Self {
        Self {
            handlers: HashMap::new(),
            event_queue: VecDeque::new(),
            event_history: VecDeque::new(),
            config,
        }
    }

    /// Register event handler
    pub fn register_handler(
        &mut self,
        event_type: String,
        handler: Box<dyn EventHandler + Send + Sync>,
    ) {
        self.handlers.entry(event_type).or_default().push(handler);
    }

    /// Queue event for processing
    pub fn queue_event(&mut self, event: DashboardEvent) -> Result<()> {
        if self.event_queue.len() >= self.config.max_queue_size {
            return Err(MetricsError::ComputationError(
                "Event queue is full".to_string(),
            ));
        }

        self.event_queue.push_back(event);
        Ok(())
    }

    /// Process queued events
    pub fn process_events(&mut self) -> Result<Vec<EventResponse>> {
        let mut responses = Vec::new();
        let batch_size = self.config.batch_size.min(self.event_queue.len());

        for _ in 0..batch_size {
            if let Some(event) = self.event_queue.pop_front() {
                if let Ok(response) = self.process_single_event(&event) {
                    if let Some(resp) = response {
                        responses.push(resp);
                    }
                }

                // Add to history
                self.add_to_history(event);
            }
        }

        Ok(responses)
    }

    /// Process single event
    fn process_single_event(&self, event: &DashboardEvent) -> Result<Option<EventResponse>> {
        if let Some(handlers) = self.handlers.get(&event.event_type) {
            // Sort handlers by priority
            let mut sorted_handlers: Vec<_> = handlers.iter().collect();
            sorted_handlers.sort_by(|a, b| b.priority().cmp(&a.priority()));

            for handler in sorted_handlers {
                if handler.can_handle(&event.event_type) {
                    if let Ok(Some(response)) = handler.handle_event(event) {
                        if response.stop_propagation {
                            return Ok(Some(response));
                        }
                        return Ok(Some(response));
                    }
                }
            }
        }

        Ok(None)
    }

    /// Add event to history
    fn add_to_history(&mut self, event: DashboardEvent) {
        if self.event_history.len() >= self.config.max_history_size {
            self.event_history.pop_front();
        }
        self.event_history.push_back(event);
    }

    /// Get event history
    pub fn get_history(&self, limit: Option<usize>) -> Vec<DashboardEvent> {
        let limit = limit.unwrap_or(self.event_history.len());
        self.event_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Clear event queue
    pub fn clear_queue(&mut self) {
        self.event_queue.clear();
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.event_queue.len()
    }
}

impl Default for EventSystemConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            max_history_size: 5000,
            batch_size: 50,
            debug_enabled: false,
            performance_monitoring: true,
        }
    }
}

impl Default for EventMetadata {
    fn default() -> Self {
        Self {
            session_id: None,
            user_id: None,
            priority: EventPriority::Normal,
            propagation: PropagationSettings::default(),
            context: HashMap::new(),
        }
    }
}

impl Default for PropagationSettings {
    fn default() -> Self {
        Self {
            stop_propagation: false,
            prevent_default: false,
            bubble: true,
            capture: false,
        }
    }
}
