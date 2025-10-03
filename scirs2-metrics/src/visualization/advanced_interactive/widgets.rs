//! Interactive widget system for advanced visualization
//!
//! This module provides the widget framework for creating interactive
//! dashboard components with real-time data binding and user interactions.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use super::core::{Position, Size};
use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Interactive widget trait
pub trait InteractiveWidget: std::fmt::Debug + Send + Sync {
    /// Get widget ID
    fn id(&self) -> &str;

    /// Get widget type
    fn widget_type(&self) -> WidgetType;

    /// Update widget with new data
    fn update_data(&mut self, data: Value) -> Result<()>;

    /// Handle user interaction
    fn handle_interaction(&mut self, event: WidgetEvent) -> Result<Option<WidgetEventResponse>>;

    /// Render widget to context
    fn render(&self, context: &RenderContext) -> Result<WidgetRender>;

    /// Get widget configuration
    fn config(&self) -> &WidgetConfig;

    /// Update widget configuration
    fn update_config(&mut self, config: WidgetConfig) -> Result<()>;

    /// Get current state
    fn state(&self) -> Value;

    /// Restore from state
    fn restore_state(&mut self, state: Value) -> Result<()>;

    /// Validate widget data
    fn validate_data(&self, data: &Value) -> Result<()>;
}

/// Widget type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    /// Chart widget (line, bar, scatter, etc.)
    Chart(ChartType),
    /// Table widget
    Table,
    /// Text widget
    Text,
    /// Input widget (slider, dropdown, etc.)
    Input(InputType),
    /// Container widget
    Container,
    /// Custom widget
    Custom(String),
}

/// Chart type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart
    Line,
    /// Bar chart
    Bar,
    /// Scatter plot
    Scatter,
    /// Heatmap
    Heatmap,
    /// Pie chart
    Pie,
    /// Area chart
    Area,
    /// Histogram
    Histogram,
    /// Box plot
    BoxPlot,
}

/// Input type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputType {
    /// Slider input
    Slider,
    /// Dropdown selection
    Dropdown,
    /// Text input
    TextInput,
    /// Checkbox
    Checkbox,
    /// Radio buttons
    RadioButton,
    /// Date picker
    DatePicker,
    /// File upload
    FileUpload,
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetConfig {
    /// Widget ID
    pub id: String,
    /// Widget title
    pub title: String,
    /// Position in the dashboard
    pub position: Position,
    /// Size of the widget
    pub size: Size,
    /// Style configuration
    pub style: StyleConfig,
    /// Data binding configuration
    pub data_binding: DataBindingConfig,
    /// Interaction settings
    pub interactions_enabled: bool,
    /// Animation settings
    pub animation_enabled: bool,
    /// Visibility
    pub visible: bool,
    /// z-index for layering
    pub z_index: i32,
}

/// Style configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleConfig {
    /// Background color
    pub background_color: String,
    /// Border configuration
    pub border: BorderConfig,
    /// Shadow configuration
    pub shadow: ShadowConfig,
    /// Font configuration
    pub font: FontConfig,
    /// Custom CSS classes
    pub css_classes: Vec<String>,
    /// Custom CSS properties
    pub css_properties: HashMap<String, String>,
}

/// Border configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderConfig {
    /// Border width
    pub width: u32,
    /// Border color
    pub color: String,
    /// Border style
    pub style: BorderStyle,
    /// Border radius
    pub radius: u32,
}

/// Border style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BorderStyle {
    /// Solid border
    Solid,
    /// Dashed border
    Dashed,
    /// Dotted border
    Dotted,
    /// No border
    None,
}

/// Shadow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShadowConfig {
    /// Enable shadow
    pub enabled: bool,
    /// Shadow offset X
    pub offset_x: i32,
    /// Shadow offset Y
    pub offset_y: i32,
    /// Shadow blur radius
    pub blur_radius: u32,
    /// Shadow color
    pub color: String,
}

/// Font configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    /// Font family
    pub family: String,
    /// Font size
    pub size: u32,
    /// Font weight
    pub weight: FontWeight,
    /// Font style
    pub style: FontStyle,
    /// Text color
    pub color: String,
}

/// Font weight enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    /// Normal weight
    Normal,
    /// Bold weight
    Bold,
    /// Light weight
    Light,
    /// Custom weight (100-900)
    Custom(u32),
}

/// Font style enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontStyle {
    /// Normal style
    Normal,
    /// Italic style
    Italic,
    /// Oblique style
    Oblique,
}

/// Data binding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataBindingConfig {
    /// Data source ID
    pub source_id: String,
    /// Data field mappings
    pub field_mappings: HashMap<String, String>,
    /// Update frequency
    pub update_frequency: UpdateFrequency,
    /// Data transformations
    pub transformations: Vec<DataTransformation>,
    /// Filtering configuration
    pub filters: HashMap<String, Value>,
    /// Aggregation method
    pub aggregation: Option<AggregationMethod>,
}

/// Update frequency enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateFrequency {
    /// Float-time updates
    RealTime,
    /// Fixed interval updates
    Interval(Duration),
    /// Manual updates only
    Manual,
    /// On-demand updates
    OnDemand,
}

/// Data transformation enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataTransformation {
    /// Filter data
    Filter(String),
    /// Sort data
    Sort(String, bool), // field, ascending
    /// Group data
    Group(String),
    /// Aggregate data
    Aggregate(String, AggregationMethod),
    /// Custom transformation
    Custom(String),
}

/// Aggregation method enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Sum aggregation
    Sum,
    /// Average aggregation
    Average,
    /// Count aggregation
    Count,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Standard deviation
    StdDev,
    /// Custom aggregation
    Custom(String),
}

/// Render context for widgets
#[derive(Debug, Clone)]
pub struct RenderContext {
    /// Canvas or surface to render to
    pub canvas_id: String,
    /// Device capabilities
    pub device: DeviceCapabilities,
    /// Rendering options
    pub options: RenderOptions,
    /// Current timestamp
    pub timestamp: Instant,
    /// Global theme settings
    pub theme: super::core::ThemeConfig,
}

/// Device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Screen width
    pub screen_width: u32,
    /// Screen height
    pub screen_height: u32,
    /// Device pixel ratio
    pub pixel_ratio: f64,
    /// WebGL support
    pub webgl_supported: bool,
    /// Touch support
    pub touch_supported: bool,
    /// Maximum texture size
    pub max_texture_size: u32,
}

/// Render options
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Render quality
    pub quality: RenderQuality,
    /// Enable anti-aliasing
    pub antialiasing: bool,
    /// Enable transparency
    pub transparency: bool,
    /// Preserve drawing buffer
    pub preserve_buffer: bool,
    /// Power preference
    pub power_preference: String,
}

/// Render quality enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RenderQuality {
    /// Low quality (performance)
    Low,
    /// Medium quality
    Medium,
    /// High quality
    High,
    /// Ultra quality
    Ultra,
}

/// Widget render result
#[derive(Debug, Clone)]
pub struct WidgetRender {
    /// Rendered content
    pub content: RenderContent,
    /// Render metadata
    pub metadata: RenderMetadata,
    /// Required resources
    pub resources: Vec<String>,
}

/// Render content enumeration
#[derive(Debug, Clone)]
pub enum RenderContent {
    /// HTML content
    Html(String),
    /// SVG content
    Svg(String),
    /// Canvas drawing commands
    Canvas(Vec<CanvasCommand>),
    /// WebGL shader program
    WebGL(ShaderProgram),
}

/// Canvas drawing command
#[derive(Debug, Clone)]
pub enum CanvasCommand {
    /// Draw line
    DrawLine {
        from: Position,
        to: Position,
        color: String,
        width: f64,
    },
    /// Draw rectangle
    DrawRect {
        position: Position,
        size: Size,
        color: String,
    },
    /// Draw circle
    DrawCircle {
        center: Position,
        radius: f64,
        color: String,
    },
    /// Draw text
    DrawText {
        position: Position,
        text: String,
        font: FontConfig,
    },
    /// Custom command
    Custom(String, HashMap<String, Value>),
}

/// Shader program configuration
#[derive(Debug, Clone)]
pub struct ShaderProgram {
    /// Vertex shader source
    pub vertex_shader: String,
    /// Fragment shader source
    pub fragment_shader: String,
    /// Uniform variables
    pub uniforms: HashMap<String, UniformValue>,
    /// Attribute bindings
    pub attributes: HashMap<String, AttributeBinding>,
}

/// Uniform value enumeration
#[derive(Debug, Clone)]
pub enum UniformValue {
    /// Float value
    Float(f32),
    /// Vector2 value
    Vec2([f32; 2]),
    /// Vector3 value
    Vec3([f32; 3]),
    /// Vector4 value
    Vec4([f32; 4]),
    /// Matrix4 value
    Mat4([[f32; 4]; 4]),
    /// Texture
    Texture(String),
}

/// Attribute binding
#[derive(Debug, Clone)]
pub struct AttributeBinding {
    /// Buffer name
    pub buffer: String,
    /// Component count
    pub components: u32,
    /// Data type
    pub data_type: AttributeType,
    /// Normalized
    pub normalized: bool,
}

/// Attribute type enumeration
#[derive(Debug, Clone)]
pub enum AttributeType {
    /// Float type
    Float,
    /// Unsigned byte type
    UnsignedByte,
    /// Short type
    Short,
    /// Unsigned short type
    UnsignedShort,
}

/// Render metadata
#[derive(Debug, Clone)]
pub struct RenderMetadata {
    /// Render time
    pub render_time: Duration,
    /// Frame rate
    pub frame_rate: f64,
    /// Memory usage
    pub memory_usage: u64,
    /// Error count
    pub error_count: u32,
}

/// Widget event
#[derive(Debug, Clone)]
pub struct WidgetEvent {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: EventType,
    /// Timestamp
    pub timestamp: Instant,
    /// Event data
    pub data: HashMap<String, Value>,
    /// Source widget
    pub source_widget: String,
    /// Target element
    pub target: Option<String>,
}

/// Event type enumeration
#[derive(Debug, Clone)]
pub enum EventType {
    /// Click event
    Click { position: Position, button: u32 },
    /// Double click event
    DoubleClick { position: Position },
    /// Mouse move event
    MouseMove { position: Position, delta: Position },
    /// Mouse enter event
    MouseEnter { position: Position },
    /// Mouse leave event
    MouseLeave { position: Position },
    /// Key press event
    KeyPress { key: String, modifiers: Vec<String> },
    /// Touch event
    Touch { touches: Vec<TouchPoint> },
    /// Resize event
    Resize { new_size: Size },
    /// Focus event
    Focus,
    /// Blur event
    Blur,
    /// Custom event
    Custom { name: String, data: Value },
}

/// Touch point
#[derive(Debug, Clone)]
pub struct TouchPoint {
    /// Touch ID
    pub id: u32,
    /// Position
    pub position: Position,
    /// Pressure
    pub pressure: f64,
    /// Radius
    pub radius: f64,
}

/// Widget event response
#[derive(Debug, Clone)]
pub struct WidgetEventResponse {
    /// Response ID
    pub id: String,
    /// Actions to perform
    pub actions: Vec<ResponseAction>,
    /// Data updates
    pub data_updates: HashMap<String, Value>,
    /// State changes
    pub state_changes: HashMap<String, Value>,
}

/// Response action enumeration
#[derive(Debug, Clone)]
pub enum ResponseAction {
    /// Update widget data
    UpdateData { widget_id: String, data: Value },
    /// Trigger event
    TriggerEvent { event: WidgetEvent },
    /// Navigate to URL
    Navigate { url: String },
    /// Show notification
    ShowNotification {
        message: String,
        level: NotificationLevel,
    },
    /// Execute custom action
    Custom {
        action: String,
        params: HashMap<String, Value>,
    },
}

/// Notification level enumeration
#[derive(Debug, Clone)]
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

impl Default for WidgetConfig {
    fn default() -> Self {
        Self {
            id: "widget".to_string(),
            title: "Widget".to_string(),
            position: Position::default(),
            size: Size::default(),
            style: StyleConfig::default(),
            data_binding: DataBindingConfig::default(),
            interactions_enabled: true,
            animation_enabled: true,
            visible: true,
            z_index: 0,
        }
    }
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            background_color: "#ffffff".to_string(),
            border: BorderConfig::default(),
            shadow: ShadowConfig::default(),
            font: FontConfig::default(),
            css_classes: Vec::new(),
            css_properties: HashMap::new(),
        }
    }
}

impl Default for BorderConfig {
    fn default() -> Self {
        Self {
            width: 1,
            color: "#cccccc".to_string(),
            style: BorderStyle::Solid,
            radius: 4,
        }
    }
}

impl Default for ShadowConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            offset_x: 2,
            offset_y: 2,
            blur_radius: 4,
            color: "rgba(0,0,0,0.1)".to_string(),
        }
    }
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "Arial, sans-serif".to_string(),
            size: 14,
            weight: FontWeight::Normal,
            style: FontStyle::Normal,
            color: "#333333".to_string(),
        }
    }
}

impl Default for DataBindingConfig {
    fn default() -> Self {
        Self {
            source_id: "default".to_string(),
            field_mappings: HashMap::new(),
            update_frequency: UpdateFrequency::RealTime,
            transformations: Vec::new(),
            filters: HashMap::new(),
            aggregation: None,
        }
    }
}
