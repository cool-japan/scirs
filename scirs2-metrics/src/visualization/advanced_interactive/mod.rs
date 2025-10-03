//! Advanced interactive visualization with real-time capabilities
//!
//! This module provides sophisticated interactive visualization features including:
//! - Float-time data streaming and updates
//! - Advanced widget systems (sliders, dropdowns, filters)
//! - Multi-dimensional visualization support
//! - Interactive dashboard components
//! - Collaborative visualization features
//! - WebGL-accelerated rendering

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

// Module declarations
pub mod collaboration;
pub mod core;
pub mod data_sources;
pub mod events;
pub mod layout;
pub mod rendering;
pub mod widgets;

// Re-export core types
pub use core::*;

// Re-export widget system
pub use widgets::{
    BorderConfig, ChartType, DataBindingConfig, EventType, FontConfig, InputType,
    InteractiveWidget, RenderContent, RenderContext, ShaderProgram, StyleConfig, WidgetConfig,
    WidgetEvent, WidgetEventResponse, WidgetType,
};

// Re-export data sources
pub use data_sources::{
    ChangeType, ConnectionConfig, DataFormat, DataSource, DataSourceConfig, DataSourceManager,
    DataSourceType, DataUpdate, ValidationConfig,
};

// Re-export event system
pub use events::{
    DashboardEvent, EventAction, EventHandler, EventMetadata, EventPriority, EventResponse,
    EventSystem, NotificationLevel,
};

// Re-export layout management
pub use layout::{
    ContainerConstraints, GridPosition, LayoutConstraints, LayoutManager, Margin, WidgetLayout,
};

// Re-export rendering
pub use rendering::{
    PerformanceMonitor, RenderStatistics, RenderingBackend, RenderingConfig, RenderingEngine,
    UpdateManager, UpdateRequest, UpdateType,
};

// Re-export collaboration
pub use collaboration::{
    CollaborationManager, Conflict, ConflictResolver, ConflictType, CursorPosition, Operation,
    OperationType, Selection, SharedState, UserSession,
};

/// Advanced interactive dashboard for real-time metrics visualization
pub struct InteractiveDashboard {
    /// Dashboard configuration
    config: DashboardConfig,
    /// Collection of widgets
    widgets: Arc<RwLock<HashMap<String, Box<dyn InteractiveWidget + Send + Sync>>>>,
    /// Data sources for real-time updates
    data_sources: Arc<RwLock<HashMap<String, Box<dyn DataSource + Send + Sync>>>>,
    /// Event system for widget interactions
    event_system: Arc<Mutex<EventSystem>>,
    /// Layout manager
    layout_manager: Arc<Mutex<LayoutManager>>,
    /// Rendering engine
    renderer: Arc<Mutex<Box<dyn RenderingEngine + Send + Sync>>>,
    /// Float-time update manager
    update_manager: Arc<Mutex<UpdateManager>>,
    /// Collaboration manager
    collaboration: Arc<Mutex<CollaborationManager>>,
    /// Dashboard state
    state: Arc<RwLock<DashboardState>>,
}

impl std::fmt::Debug for InteractiveDashboard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractiveDashboard")
            .field("config", &self.config)
            .field(
                "widgets",
                &format!("{} widgets", self.widgets.read().unwrap().len()),
            )
            .field(
                "data_sources",
                &format!("{} data sources", self.data_sources.read().unwrap().len()),
            )
            .field("event_system", &"<event_system>")
            .field("layout_manager", &"<layout_manager>")
            .field("renderer", &"<renderer>")
            .field("update_manager", &"<update_manager>")
            .field("collaboration", &"<collaboration>")
            .field("state", &"<state>")
            .finish()
    }
}

/// Dashboard state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardState {
    /// Current view state
    pub view_state: ViewState,
    /// Filter state
    pub filters: HashMap<String, Value>,
    /// Selection state
    pub selections: Vec<String>,
    /// Zoom and pan state
    pub viewport: ViewportState,
    /// Time range state
    pub time_range: Option<TimeRange>,
    /// Custom state variables
    pub custom_state: HashMap<String, Value>,
}

/// View state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewState {
    /// Current view mode
    pub mode: ViewMode,
    /// Visible widgets
    pub visible_widgets: Vec<String>,
    /// Widget z-order
    pub z_order: Vec<String>,
    /// Layout mode
    pub layout_mode: String,
}

/// View mode enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ViewMode {
    /// Normal view
    Normal,
    /// Full-screen view
    FullScreen,
    /// Presentation mode
    Presentation,
    /// Edit mode
    Edit,
    /// Preview mode
    Preview,
}

/// Viewport state for zoom and pan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportState {
    /// Zoom level
    pub zoom: f64,
    /// Pan offset X
    pub pan_x: f64,
    /// Pan offset Y
    pub pan_y: f64,
    /// Viewport bounds
    pub bounds: ViewportBounds,
}

/// Viewport bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewportBounds {
    /// Minimum X
    pub min_x: f64,
    /// Maximum X
    pub max_x: f64,
    /// Minimum Y
    pub min_y: f64,
    /// Maximum Y
    pub max_y: f64,
}

/// Time range for temporal data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time
    pub start: std::time::SystemTime,
    /// End time
    pub end: std::time::SystemTime,
    /// Time zone
    pub timezone: Option<String>,
}

impl InteractiveDashboard {
    /// Create new interactive dashboard
    pub fn new(
        config: DashboardConfig,
        renderer: Box<dyn RenderingEngine + Send + Sync>,
    ) -> Result<Self> {
        let container_constraints = ContainerConstraints {
            width: config.width as f64,
            height: config.height as f64,
            ..Default::default()
        };

        let layout_manager = LayoutManager::new(config.layout.clone(), container_constraints);
        let event_system = EventSystem::new(events::EventSystemConfig::default());
        let update_manager = UpdateManager::new(rendering::UpdateConfig::default());

        let collaboration = if let Some(collab_config) = &config.collaboration_config {
            CollaborationManager::new(collab_config.clone())
        } else {
            // Create a disabled collaboration manager
            CollaborationManager::new(CollaborationConfig {
                enabled: false,
                auth: core::AuthConfig {
                    method: core::AuthMethod::None,
                    session_timeout: std::time::Duration::from_secs(3600),
                    guest_access: true,
                    required_permissions: Vec::new(),
                },
                sharing: core::ShareConfig {
                    public_sharing: false,
                    default_permission: core::PermissionLevel::ReadOnly,
                    link_expiration: None,
                    password_protection: false,
                },
                sync: core::SyncConfig {
                    sync_interval: std::time::Duration::from_secs(5),
                    conflict_resolution: core::ConflictResolution::LastWriterWins,
                    operational_transforms: false,
                    history_retention: std::time::Duration::from_secs(3600),
                },
                max_collaborators: 1,
            })
        };

        Ok(Self {
            config,
            widgets: Arc::new(RwLock::new(HashMap::new())),
            data_sources: Arc::new(RwLock::new(HashMap::new())),
            event_system: Arc::new(Mutex::new(event_system)),
            layout_manager: Arc::new(Mutex::new(layout_manager)),
            renderer: Arc::new(Mutex::new(renderer)),
            update_manager: Arc::new(Mutex::new(update_manager)),
            collaboration: Arc::new(Mutex::new(collaboration)),
            state: Arc::new(RwLock::new(DashboardState::default())),
        })
    }

    /// Add widget to dashboard
    pub fn add_widget(&self, widget: Box<dyn InteractiveWidget + Send + Sync>) -> Result<()> {
        let widget_id = widget.id().to_string();
        let widget_config = widget.config().clone();

        // Add to widgets collection
        self.widgets
            .write()
            .unwrap()
            .insert(widget_id.clone(), widget);

        // Update layout
        self.layout_manager
            .lock()
            .unwrap()
            .add_widget(&widget_config)?;

        // Update view state
        let mut state = self.state.write().unwrap();
        state.view_state.visible_widgets.push(widget_id.clone());
        state.view_state.z_order.push(widget_id);

        Ok(())
    }

    /// Remove widget from dashboard
    pub fn remove_widget(&self, widget_id: &str) -> Result<()> {
        // Remove from widgets collection
        self.widgets.write().unwrap().remove(widget_id);

        // Update layout
        self.layout_manager
            .lock()
            .unwrap()
            .remove_widget(widget_id)?;

        // Update view state
        let mut state = self.state.write().unwrap();
        state
            .view_state
            .visible_widgets
            .retain(|id| id != widget_id);
        state.view_state.z_order.retain(|id| id != widget_id);

        Ok(())
    }

    /// Register data source
    pub fn register_data_source(&self, source: Box<dyn DataSource + Send + Sync>) -> Result<()> {
        let source_id = source.id().to_string();
        self.data_sources.write().unwrap().insert(source_id, source);
        Ok(())
    }

    /// Handle user interaction
    pub fn handle_interaction(&self, event: WidgetEvent) -> Result<()> {
        // Convert to dashboard event
        let dashboard_event = DashboardEvent {
            id: event.id.clone(),
            event_type: format!("{:?}", event.event_type),
            source: event.source_widget,
            target: event.target,
            timestamp: event.timestamp,
            data: event.data,
            metadata: EventMetadata::default(),
        };

        // Queue event for processing
        self.event_system
            .lock()
            .unwrap()
            .queue_event(dashboard_event)?;

        Ok(())
    }

    /// Update dashboard state
    pub fn update_state(&self, updates: HashMap<String, Value>) -> Result<()> {
        let mut state = self.state.write().unwrap();

        for (key, value) in updates {
            state.custom_state.insert(key, value);
        }

        Ok(())
    }

    /// Render dashboard
    pub fn render(&self, context: &RenderContext) -> Result<()> {
        let widgets = self.widgets.read().unwrap();
        let renderer = self.renderer.lock().unwrap();

        // Clear render target
        renderer.clear([0.0, 0.0, 0.0, 1.0])?;

        // Get visible widgets in z-order
        let state = self.state.read().unwrap();
        let mut visible_widgets: Vec<_> = state
            .view_state
            .z_order
            .iter()
            .filter(|id| state.view_state.visible_widgets.contains(id))
            .filter_map(|id| widgets.get(id))
            .collect();

        // Sort by z-index
        visible_widgets.sort_by_key(|widget| widget.config().z_index);

        // Render each widget
        for widget in visible_widgets {
            if let Ok(widget_render) = widget.render(context) {
                renderer.render_widget(&widget_render, context)?;
            }
        }

        // Present frame
        renderer.present()?;

        Ok(())
    }

    /// Process real-time updates
    pub fn process_updates(&self) -> Result<()> {
        // Process events
        let _responses = self.event_system.lock().unwrap().process_events()?;

        // Process data updates
        let _update_requests = self.update_manager.lock().unwrap().process_updates()?;

        // Apply collaboration updates
        if self
            .config
            .collaboration_config
            .as_ref()
            .is_some_and(|c| c.enabled)
        {
            let _resolved_operations = self.collaboration.lock().unwrap().resolve_conflicts()?;
        }

        Ok(())
    }

    /// Get dashboard configuration
    pub fn config(&self) -> &DashboardConfig {
        &self.config
    }

    /// Get current dashboard state
    pub fn state(&self) -> DashboardState {
        self.state.read().unwrap().clone()
    }

    /// Export dashboard configuration
    pub fn export_config(&self) -> Result<Value> {
        serde_json::to_value(&self.config).map_err(|e| MetricsError::InvalidInput(e.to_string()))
    }

    /// Import dashboard configuration
    pub fn import_config(&mut self, config_data: Value) -> Result<()> {
        self.config = serde_json::from_value(config_data)
            .map_err(|e| MetricsError::InvalidInput(e.to_string()))?;
        Ok(())
    }
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            mode: ViewMode::Normal,
            visible_widgets: Vec::new(),
            z_order: Vec::new(),
            layout_mode: "grid".to_string(),
        }
    }
}

impl Default for ViewportState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            bounds: ViewportBounds::default(),
        }
    }
}

impl Default for ViewportBounds {
    fn default() -> Self {
        Self {
            min_x: 0.0,
            max_x: 1200.0,
            min_y: 0.0,
            max_y: 800.0,
        }
    }
}
