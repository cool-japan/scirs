//! Core types and configurations for advanced interactive visualization
//!
//! This module provides the fundamental types, configurations, and structures
//! that are shared across the interactive visualization system.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard title
    pub title: String,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Theme configuration
    pub theme: ThemeConfig,
    /// Layout configuration
    pub layout: LayoutConfig,
    /// Float-time update settings
    pub realtime_config: RealtimeConfig,
    /// Interaction settings
    pub interaction_config: InteractionConfig,
    /// Export settings
    pub export_config: ExportConfig,
    /// Collaboration settings
    pub collaboration_config: Option<CollaborationConfig>,
}

/// Theme configuration for the dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThemeConfig {
    /// Primary color scheme
    pub primary_color: String,
    /// Secondary color scheme
    pub secondary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
    /// Font family
    pub font_family: String,
    /// Font size
    pub font_size: u32,
    /// Border radius
    pub border_radius: u32,
    /// Shadow settings
    pub shadow_enabled: bool,
    /// Custom CSS variables
    pub custom_variables: HashMap<String, String>,
}

/// Layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutConfig {
    /// Layout type
    pub layout_type: LayoutType,
    /// Grid configuration (if using grid layout)
    pub grid_config: Option<GridConfig>,
    /// Spacing configuration
    pub spacing: SpacingConfig,
    /// Animation configuration
    pub animation: AnimationConfig,
    /// Responsive breakpoints
    pub breakpoints: HashMap<String, u32>,
}

/// Layout type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayoutType {
    /// Fixed layout
    Fixed,
    /// Responsive grid layout
    Grid,
    /// Flexible layout
    Flexbox,
    /// Masonry layout
    Masonry,
    /// Custom layout
    Custom(String),
}

/// Grid configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Number of columns
    pub columns: u32,
    /// Number of rows
    pub rows: u32,
    /// Column gap
    pub column_gap: u32,
    /// Row gap
    pub row_gap: u32,
    /// Auto-fit columns
    pub auto_fit: bool,
}

/// Spacing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpacingConfig {
    /// Margin
    pub margin: u32,
    /// Padding
    pub padding: u32,
    /// Widget spacing
    pub widget_spacing: u32,
    /// Container spacing
    pub container_spacing: u32,
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Enable animations
    pub enabled: bool,
    /// Animation duration
    pub duration: Duration,
    /// Easing function
    pub easing: String,
    /// Performance mode
    pub performance_mode: bool,
}

/// Float-time configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealtimeConfig {
    /// Enable real-time updates
    pub enabled: bool,
    /// Update interval
    pub update_interval: Duration,
    /// Buffer size for data
    pub buffer_size: usize,
    /// Maximum concurrent connections
    pub max_connections: u32,
    /// Streaming protocol
    pub protocol: StreamingProtocol,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Retry attempts
    pub retry_attempts: u32,
}

/// Streaming protocol enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamingProtocol {
    /// WebSocket protocol
    WebSocket,
    /// Server-Sent Events
    SSE,
    /// Long polling
    LongPolling,
    /// WebRTC
    WebRTC,
}

/// Interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfig {
    /// Enable touch interactions
    pub touch_enabled: bool,
    /// Enable keyboard shortcuts
    pub keyboard_shortcuts: bool,
    /// Enable zoom and pan
    pub zoom_pan_enabled: bool,
    /// Enable selection
    pub selection_enabled: bool,
    /// Enable drag and drop
    pub drag_drop_enabled: bool,
    /// Double-click threshold
    pub double_click_threshold: Duration,
    /// Hover delay
    pub hover_delay: Duration,
    /// Custom interaction handlers
    pub custom_handlers: HashMap<String, String>,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Supported export formats
    pub formats: Vec<ExportFormat>,
    /// Default format
    pub default_format: ExportFormat,
    /// Quality settings
    pub quality: u32,
    /// Include metadata
    pub include_metadata: bool,
    /// Compression enabled
    pub compression: bool,
}

/// Export format enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    /// PNG image
    PNG,
    /// JPEG image
    JPEG,
    /// SVG vector
    SVG,
    /// PDF document
    PDF,
    /// HTML file
    HTML,
    /// JSON data
    JSON,
}

/// Collaboration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollaborationConfig {
    /// Enable collaboration
    pub enabled: bool,
    /// Authentication configuration
    pub auth: AuthConfig,
    /// Sharing configuration
    pub sharing: ShareConfig,
    /// Synchronization configuration
    pub sync: SyncConfig,
    /// Maximum collaborators
    pub max_collaborators: u32,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Authentication method
    pub method: AuthMethod,
    /// Session timeout
    pub session_timeout: Duration,
    /// Enable guest access
    pub guest_access: bool,
    /// Required permissions
    pub required_permissions: Vec<String>,
}

/// Authentication method enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthMethod {
    /// No authentication
    None,
    /// Token-based authentication
    Token,
    /// OAuth authentication
    OAuth,
    /// API key authentication
    ApiKey,
    /// Custom authentication
    Custom(String),
}

/// Share configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShareConfig {
    /// Enable public sharing
    pub public_sharing: bool,
    /// Default permission level
    pub default_permission: PermissionLevel,
    /// Link expiration
    pub link_expiration: Option<Duration>,
    /// Password protection
    pub password_protection: bool,
}

/// Permission level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PermissionLevel {
    /// Read-only access
    ReadOnly,
    /// Comment access
    Comment,
    /// Edit access
    Edit,
    /// Admin access
    Admin,
}

/// Synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConfig {
    /// Sync interval
    pub sync_interval: Duration,
    /// Conflict resolution strategy
    pub conflict_resolution: ConflictResolution,
    /// Enable operational transforms
    pub operational_transforms: bool,
    /// History retention
    pub history_retention: Duration,
}

/// Conflict resolution strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Last writer wins
    LastWriterWins,
    /// First writer wins
    FirstWriterWins,
    /// Manual resolution
    Manual,
    /// Merge changes
    Merge,
}

/// Position configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

/// Size configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Size {
    /// Width
    pub width: f64,
    /// Height
    pub height: f64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            title: "Interactive Dashboard".to_string(),
            width: 1200,
            height: 800,
            theme: ThemeConfig::default(),
            layout: LayoutConfig::default(),
            realtime_config: RealtimeConfig::default(),
            interaction_config: InteractionConfig::default(),
            export_config: ExportConfig::default(),
            collaboration_config: None,
        }
    }
}

impl Default for ThemeConfig {
    fn default() -> Self {
        Self {
            primary_color: "#007bff".to_string(),
            secondary_color: "#6c757d".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#212529".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: 14,
            border_radius: 4,
            shadow_enabled: true,
            custom_variables: HashMap::new(),
        }
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            layout_type: LayoutType::Grid,
            grid_config: Some(GridConfig::default()),
            spacing: SpacingConfig::default(),
            animation: AnimationConfig::default(),
            breakpoints: HashMap::new(),
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            columns: 12,
            rows: 8,
            column_gap: 16,
            row_gap: 16,
            auto_fit: true,
        }
    }
}

impl Default for SpacingConfig {
    fn default() -> Self {
        Self {
            margin: 16,
            padding: 16,
            widget_spacing: 8,
            container_spacing: 24,
        }
    }
}

impl Default for AnimationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            duration: Duration::from_millis(300),
            easing: "ease-in-out".to_string(),
            performance_mode: false,
        }
    }
}

impl Default for RealtimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval: Duration::from_millis(1000),
            buffer_size: 1000,
            max_connections: 100,
            protocol: StreamingProtocol::WebSocket,
            connection_timeout: Duration::from_secs(30),
            retry_attempts: 3,
        }
    }
}

impl Default for InteractionConfig {
    fn default() -> Self {
        Self {
            touch_enabled: true,
            keyboard_shortcuts: true,
            zoom_pan_enabled: true,
            selection_enabled: true,
            drag_drop_enabled: true,
            double_click_threshold: Duration::from_millis(500),
            hover_delay: Duration::from_millis(300),
            custom_handlers: HashMap::new(),
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            formats: vec![ExportFormat::PNG, ExportFormat::SVG, ExportFormat::JSON],
            default_format: ExportFormat::PNG,
            quality: 90,
            include_metadata: true,
            compression: true,
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl Default for Size {
    fn default() -> Self {
        Self {
            width: 100.0,
            height: 100.0,
        }
    }
}
