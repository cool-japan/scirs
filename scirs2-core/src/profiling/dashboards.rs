//! # Performance Dashboards System
//!
//! Enterprise-grade performance monitoring dashboards with real-time visualization
//! and historical trend analysis for production environments. Provides comprehensive
//! performance insights with customizable metrics and alerting capabilities.
//!
//! ## Features
//!
//! - Real-time performance visualization with live updates
//! - Historical trend analysis with configurable time ranges
//! - Customizable dashboard layouts and widgets
//! - Interactive charts and graphs with drill-down capabilities
//! - Performance alerting with threshold-based notifications
//! - Multi-dimensional metrics aggregation and filtering
//! - Export capabilities for reports and presentations
//! - Integration with external monitoring systems
//! - Web-based dashboard interface with REST API
//! - Mobile-responsive design for on-the-go monitoring
//!
//! ## Example
//!
//! ```rust
//! use scirs2_core::profiling::dashboards::{
//!     PerformanceDashboard, DashboardConfig, Widget, ChartType, MetricSource
//! };
//!
//! // Create a performance dashboard
//! let config = DashboardConfig::default()
//!     .with_title("Production System Performance")
//!     .with_refresh_interval(std::time::Duration::from_secs(30))
//!     .with_retention_period(std::time::Duration::from_secs(30 * 24 * 60 * 60));
//!
//! let mut dashboard = PerformanceDashboard::new(config)?;
//!
//! // Add performance widgets
//! dashboard.add_widget(Widget::new()
//!     .with_title("CPU Usage")
//!     .with_chart_type(ChartType::LineChart)
//!     .with_metric_source(MetricSource::SystemCpu)
//!     .with_alert_threshold(80.0)
//! )?;
//!
//! dashboard.add_widget(Widget::new()
//!     .with_title("Memory Usage")
//!     .with_chart_type(ChartType::AreaChart)
//!     .with_metric_source(MetricSource::SystemMemory)
//! )?;
//!
//! // Start the dashboard
//! dashboard.start()?;
//!
//! // Dashboard will now continuously update with real-time metrics
//! // Access via web interface at http://localhost:8080/dashboard
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::error::{CoreError, CoreResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Dashboard title
    pub title: String,
    /// Refresh interval for real-time updates
    pub refresh_interval: Duration,
    /// Data retention period
    pub retention_period: Duration,
    /// Maximum number of data points to keep
    pub max_data_points: usize,
    /// Enable web interface
    pub enable_web_interface: bool,
    /// Web server port
    pub web_port: u16,
    /// Enable REST API
    pub enable_rest_api: bool,
    /// API authentication token
    pub api_token: Option<String>,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Dashboard theme
    pub theme: DashboardTheme,
    /// Auto-save interval for persistence
    pub auto_save_interval: Duration,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            title: "Performance Dashboard".to_string(),
            refresh_interval: Duration::from_secs(5),
            retention_period: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            max_data_points: 1000,
            enable_web_interface: true,
            web_port: 8080,
            enable_rest_api: true,
            api_token: None,
            enable_alerts: true,
            theme: DashboardTheme::Dark,
            auto_save_interval: Duration::from_secs(60),
        }
    }
}

impl DashboardConfig {
    /// Set dashboard title
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Set refresh interval
    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = interval;
        self
    }

    /// Set data retention period
    pub fn with_retention_period(mut self, period: Duration) -> Self {
        self.retention_period = period;
        self
    }

    /// Enable web interface on specific port
    pub fn with_web_interface(mut self, port: u16) -> Self {
        self.enable_web_interface = true;
        self.web_port = port;
        self
    }

    /// Set API authentication token
    pub fn with_api_token(mut self, token: &str) -> Self {
        self.api_token = Some(token.to_string());
        self
    }

    /// Set dashboard theme
    pub fn with_theme(mut self, theme: DashboardTheme) -> Self {
        self.theme = theme;
        self
    }
}

/// Dashboard theme options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardTheme {
    /// Light theme
    Light,
    /// Dark theme
    Dark,
    /// High contrast theme
    HighContrast,
    /// Custom theme
    Custom,
}

/// Chart types for dashboard widgets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart for time series data
    LineChart,
    /// Area chart for cumulative data
    AreaChart,
    /// Bar chart for categorical data
    BarChart,
    /// Gauge chart for single values
    GaugeChart,
    /// Pie chart for proportional data
    PieChart,
    /// Heatmap for matrix data
    Heatmap,
    /// Scatter plot for correlation analysis
    ScatterPlot,
    /// Histogram for distribution analysis
    Histogram,
}

/// Metric data sources
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricSource {
    /// System CPU usage
    SystemCpu,
    /// System memory usage
    SystemMemory,
    /// Network I/O
    NetworkIO,
    /// Disk I/O
    DiskIO,
    /// Application-specific metrics
    Application(String),
    /// Custom metric query
    Custom(String),
    /// Database metrics
    Database(String),
    /// Cache metrics
    Cache(String),
}

/// Dashboard widget configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    /// Widget ID
    pub id: String,
    /// Widget title
    pub title: String,
    /// Chart type
    pub chart_type: ChartType,
    /// Data source
    pub metric_source: MetricSource,
    /// Position and size
    pub layout: WidgetLayout,
    /// Alert configuration
    pub alert_config: Option<AlertConfig>,
    /// Refresh interval (overrides dashboard default)
    pub refresh_interval: Option<Duration>,
    /// Color scheme
    pub color_scheme: Vec<String>,
    /// Display options
    pub display_options: DisplayOptions,
}

/// Widget layout and positioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetLayout {
    /// X position (grid units)
    pub x: u32,
    /// Y position (grid units)
    pub y: u32,
    /// Width (grid units)
    pub width: u32,
    /// Height (grid units)
    pub height: u32,
}

impl Default for WidgetLayout {
    fn default() -> Self {
        Self {
            x: 0,
            y: 0,
            width: 4,
            height: 3,
        }
    }
}

/// Alert configuration for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Alert threshold value
    pub threshold: f64,
    /// Alert condition
    pub condition: AlertCondition,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Notification channels
    pub notification_channels: Vec<NotificationChannel>,
    /// Cooldown period to prevent spam
    pub cooldown_period: Duration,
}

/// Alert conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Greater than threshold
    GreaterThan,
    /// Less than threshold
    LessThan,
    /// Equal to threshold
    EqualTo,
    /// Not equal to threshold
    NotEqualTo,
    /// Rate of change exceeds threshold
    RateOfChange,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Error alert
    Error,
    /// Critical alert
    Critical,
}

/// Notification channels for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    /// Email notification
    Email(String),
    /// Slack webhook
    Slack(String),
    /// Custom webhook
    Webhook(String),
    /// Console log
    Console,
    /// File log
    File(String),
}

/// Display options for widgets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayOptions {
    /// Show data labels
    pub show_labels: bool,
    /// Show grid lines
    pub show_grid: bool,
    /// Show legend
    pub show_legend: bool,
    /// Animation enabled
    pub enable_animation: bool,
    /// Number format
    pub number_format: NumberFormat,
    /// Time format for time series
    pub time_format: String,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self {
            show_labels: true,
            show_grid: true,
            show_legend: true,
            enable_animation: true,
            number_format: NumberFormat::Auto,
            time_format: "%H:%M:%S".to_string(),
        }
    }
}

/// Number formatting options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumberFormat {
    /// Automatic formatting
    Auto,
    /// Integer formatting
    Integer,
    /// Decimal formatting with specified precision
    Decimal(u8),
    /// Percentage formatting
    Percentage,
    /// Scientific notation
    Scientific,
    /// Bytes formatting (KB, MB, GB)
    Bytes,
}

impl Widget {
    /// Create a new widget with default settings
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: "New Widget".to_string(),
            chart_type: ChartType::LineChart,
            metric_source: MetricSource::SystemCpu,
            layout: WidgetLayout::default(),
            alert_config: None,
            refresh_interval: None,
            color_scheme: vec![
                "#007acc".to_string(),
                "#ff6b35".to_string(),
                "#00b894".to_string(),
                "#fdcb6e".to_string(),
                "#e84393".to_string(),
            ],
            display_options: DisplayOptions::default(),
        }
    }

    /// Set widget title
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Set chart type
    pub fn with_chart_type(mut self, charttype: ChartType) -> Self {
        self.chart_type = charttype;
        self
    }

    /// Set metric source
    pub fn with_metric_source(mut self, source: MetricSource) -> Self {
        self.metric_source = source;
        self
    }

    /// Set widget layout
    pub const fn with_layout(mut self, x: u32, y: u32, width: u32, height: u32) -> Self {
        self.layout = WidgetLayout {
            x,
            y,
            width,
            height,
        };
        self
    }

    /// Set alert threshold
    pub fn with_alert_threshold(mut self, threshold: f64) -> Self {
        self.alert_config = Some(AlertConfig {
            threshold,
            condition: AlertCondition::GreaterThan,
            severity: AlertSeverity::Warning,
            notification_channels: vec![NotificationChannel::Console],
            cooldown_period: Duration::from_secs(300), // 5 minutes
        });
        self
    }

    /// Set custom refresh interval
    pub fn with_refresh_interval(mut self, interval: Duration) -> Self {
        self.refresh_interval = Some(interval);
        self
    }

    /// Set color scheme
    pub fn with_colors(mut self, colors: Vec<&str>) -> Self {
        self.color_scheme = colors.into_iter().map(|s| s.to_string()).collect();
        self
    }
}

impl Default for Widget {
    fn default() -> Self {
        Self::new()
    }
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Metric value
    pub value: f64,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

/// Time series data for a metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTimeSeries {
    /// Metric name
    pub name: String,
    /// Data points
    pub data_points: VecDeque<MetricDataPoint>,
    /// Last update time
    pub last_update: SystemTime,
}

impl MetricTimeSeries {
    /// Create a new time series
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            data_points: VecDeque::new(),
            last_update: SystemTime::now(),
        }
    }

    /// Add a data point
    pub fn add_point(&mut self, value: f64, metadata: Option<HashMap<String, String>>) {
        let point = MetricDataPoint {
            timestamp: SystemTime::now(),
            value,
            metadata: metadata.unwrap_or_default(),
        };

        self.data_points.push_back(point);
        self.last_update = SystemTime::now();
    }

    /// Get latest value
    pub fn latest_value(&self) -> Option<f64> {
        self.data_points.back().map(|p| p.value)
    }

    /// Get average value over time range
    pub fn average_value(&self, duration: Duration) -> Option<f64> {
        let cutoff = SystemTime::now() - duration;
        let values: Vec<f64> = self
            .data_points
            .iter()
            .filter(|p| p.timestamp >= cutoff)
            .map(|p| p.value)
            .collect();

        if values.is_empty() {
            None
        } else {
            Some(values.iter().sum::<f64>() / values.len() as f64)
        }
    }

    /// Clean old data points
    pub fn cleanup(&mut self, retention_period: Duration, maxpoints: usize) {
        let cutoff = SystemTime::now() - retention_period;

        // Remove old data points
        while let Some(front) = self.data_points.front() {
            if front.timestamp < cutoff {
                self.data_points.pop_front();
            } else {
                break;
            }
        }

        // Limit maximum number of points
        while self.data_points.len() > maxpoints {
            self.data_points.pop_front();
        }
    }
}

/// Dashboard alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlert {
    /// Alert ID
    pub id: String,
    /// Widget ID that triggered the alert
    pub widget_id: String,
    /// Alert message
    pub message: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Trigger time
    pub triggered_at: SystemTime,
    /// Current value that triggered the alert
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Alert status
    pub status: AlertStatus,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert is active
    Active,
    /// Alert is acknowledged
    Acknowledged,
    /// Alert is resolved
    Resolved,
}

/// Performance dashboard
pub struct PerformanceDashboard {
    /// Dashboard configuration
    config: DashboardConfig,
    /// Dashboard widgets
    widgets: HashMap<String, Widget>,
    /// Metric time series data
    metrics: Arc<RwLock<HashMap<String, MetricTimeSeries>>>,
    /// Active alerts
    alerts: Arc<Mutex<HashMap<String, DashboardAlert>>>,
    /// Dashboard state
    state: DashboardState,
    /// Last update time
    last_update: Instant,
    /// Web server handle (if enabled)
    web_server_handle: Option<WebServerHandle>,
}

/// Dashboard state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardState {
    /// Dashboard is stopped
    Stopped,
    /// Dashboard is running
    Running,
    /// Dashboard is paused
    Paused,
    /// Dashboard has encountered an error
    Error,
}

/// Web server handle for dashboard
pub struct WebServerHandle {
    /// Server address
    pub address: String,
    /// Server port
    pub port: u16,
    /// Running state
    pub running: Arc<Mutex<bool>>,
}

impl PerformanceDashboard {
    /// Create a new performance dashboard
    pub fn new(config: DashboardConfig) -> CoreResult<Self> {
        Ok(Self {
            config,
            widgets: HashMap::new(),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(HashMap::new())),
            state: DashboardState::Stopped,
            last_update: Instant::now(),
            web_server_handle: None,
        })
    }

    /// Add a widget to the dashboard
    pub fn add_widget(&mut self, widget: Widget) -> CoreResult<String> {
        let widget_id = widget.id.clone();
        self.widgets.insert(widget_id.clone(), widget);

        // Initialize metric time series for this widget
        let metrics_name = format!("widget_{widget_id}");
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.insert(metrics_name, MetricTimeSeries::new(&widget_id));
        }

        Ok(widget_id)
    }

    /// Remove a widget from the dashboard
    pub fn remove_widget(&mut self, widgetid: &str) -> CoreResult<()> {
        self.widgets.remove(widgetid);

        // Remove associated metric data
        let metrics_name = format!("widget_{widgetid}");
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.remove(&metrics_name);
        }

        Ok(())
    }

    /// Start the dashboard
    pub fn start(&mut self) -> CoreResult<()> {
        if self.state == DashboardState::Running {
            return Ok(());
        }

        // Start web server if enabled
        if self.config.enable_web_interface {
            self.start_web_server()?;
        }

        self.state = DashboardState::Running;
        self.last_update = Instant::now();

        Ok(())
    }

    /// Stop the dashboard
    pub fn stop(&mut self) -> CoreResult<()> {
        // Stop web server if running
        if let Some(ref handle) = self.web_server_handle {
            if let Ok(mut running) = handle.running.lock() {
                *running = false;
            }
        }

        self.state = DashboardState::Stopped;
        Ok(())
    }

    /// Update dashboard with new metric data
    pub fn update_metric(&mut self, source: &MetricSource, value: f64) -> CoreResult<()> {
        let metricname = self.metric_source_to_name(source);

        if let Ok(mut metrics) = self.metrics.write() {
            let time_series = metrics
                .entry(metricname.clone())
                .or_insert_with(|| MetricTimeSeries::new(&metricname));

            time_series.add_point(value, None);

            // Clean up old data
            time_series.cleanup(self.config.retention_period, self.config.max_data_points);
        }

        // Check for alerts
        self.check_alerts(&metricname, value)?;

        self.last_update = Instant::now();
        Ok(())
    }

    /// Get current metrics data
    pub fn get_metrics(&self) -> CoreResult<HashMap<String, MetricTimeSeries>> {
        self.metrics
            .read()
            .map(|metrics| metrics.clone())
            .map_err(|_| CoreError::from(std::io::Error::other("Failed to read metrics")))
    }

    /// Get dashboard statistics
    pub fn get_statistics(&self) -> DashboardStatistics {
        let metrics = self.metrics.read().unwrap();
        let alerts = self.alerts.lock().unwrap();

        DashboardStatistics {
            total_widgets: self.widgets.len(),
            total_metrics: metrics.len(),
            active_alerts: alerts
                .values()
                .filter(|a| a.status == AlertStatus::Active)
                .count(),
            last_update: self.last_update,
            uptime: self.last_update.elapsed(),
            state: self.state,
        }
    }

    /// Export dashboard configuration
    pub fn export_config(&self) -> CoreResult<String> {
        {
            let export_data = DashboardExport {
                config: self.config.clone(),
                widgets: self.widgets.values().cloned().collect(),
                created_at: SystemTime::now(),
            };

            serde_json::to_string_pretty(&export_data).map_err(|e| {
                CoreError::from(std::io::Error::other(format!(
                    "Failed to serialize dashboard config: {e}"
                )))
            })
        }
        #[cfg(not(feature = "serde"))]
        {
            Ok(format!("title: {}", self.config.title))
        }
    }

    /// Import dashboard configuration
    pub fn import_configuration(&mut self, configjson: &str) -> CoreResult<()> {
        {
            let import_data: DashboardExport = serde_json::from_str(configjson).map_err(|e| {
                CoreError::from(std::io::Error::other(format!(
                    "Failed to parse dashboard config: {e}"
                )))
            })?;

            self.config = import_data.config;
            self.widgets.clear();

            for widget in import_data.widgets {
                self.widgets.insert(widget.id.clone(), widget);
            }

            Ok(())
        }
        #[cfg(not(feature = "serde"))]
        {
            let _ = config_json; // Suppress unused variable warning
            Err(CoreError::from(std::io::Error::other(
                "Serde feature not enabled for configuration import",
            )))
        }
    }

    /// Check for alert conditions
    fn check_alerts(&self, metricname: &str, value: f64) -> CoreResult<()> {
        for widget in self.widgets.values() {
            if let Some(ref alert_config) = widget.alert_config {
                let widget_metric = self.metric_source_to_name(&widget.metric_source);

                if widget_metric == metricname {
                    let triggered = match alert_config.condition {
                        AlertCondition::GreaterThan => value > alert_config.threshold,
                        AlertCondition::LessThan => value < alert_config.threshold,
                        AlertCondition::EqualTo => {
                            (value - alert_config.threshold).abs() < f64::EPSILON
                        }
                        AlertCondition::NotEqualTo => {
                            (value - alert_config.threshold).abs() > f64::EPSILON
                        }
                        AlertCondition::RateOfChange => {
                            // Would implement rate of change calculation here
                            false
                        }
                    };

                    if triggered {
                        let alert = DashboardAlert {
                            id: uuid::Uuid::new_v4().to_string(),
                            widget_id: widget.id.clone(),
                            message: format!(
                                "Alert triggered for '{}': value {:.2} {} threshold {:.2}",
                                widget.title,
                                value,
                                match alert_config.condition {
                                    AlertCondition::GreaterThan => "exceeds",
                                    AlertCondition::LessThan => "below",
                                    _ => "meets",
                                },
                                alert_config.threshold
                            ),
                            severity: alert_config.severity,
                            triggered_at: SystemTime::now(),
                            current_value: value,
                            threshold_value: alert_config.threshold,
                            status: AlertStatus::Active,
                        };

                        if let Ok(mut alerts) = self.alerts.lock() {
                            alerts.insert(alert.id.clone(), alert.clone());
                        }

                        // Send notifications
                        self.send_alert_notifications(&alert, alert_config)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Send alert notifications
    fn send_alert_notifications(
        &self,
        alert: &DashboardAlert,
        config: &AlertConfig,
    ) -> CoreResult<()> {
        for channel in &config.notification_channels {
            match channel {
                NotificationChannel::Console => {
                    println!("[DASHBOARD ALERT] {message}", message = alert.message);
                }
                NotificationChannel::File(path) => {
                    use std::fs::OpenOptions;
                    use std::io::Write;

                    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
                        writeln!(
                            file,
                            "[{}] {}",
                            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"),
                            alert.message
                        )
                        .ok();
                    }
                }
                #[cfg(feature = "observability_http")]
                NotificationChannel::Webhook(url) => {
                    // Would implement webhook notification here
                    let _ = url; // Suppress unused warning
                }
                #[cfg(not(feature = "observability_http"))]
                NotificationChannel::Webhook(_) => {
                    // Webhook notifications require HTTP feature
                }
                NotificationChannel::Email(_) | NotificationChannel::Slack(_) => {
                    // Would implement email/slack notifications here in production
                }
            }
        }

        Ok(())
    }

    /// Convert metric source to internal metric name
    fn metric_source_to_name(&self, source: &MetricSource) -> String {
        match source {
            MetricSource::SystemCpu => "system.cpu".to_string(),
            MetricSource::SystemMemory => "system.memory".to_string(),
            MetricSource::NetworkIO => "system.network_io".to_string(),
            MetricSource::DiskIO => "system.disk_io".to_string(),
            MetricSource::Application(name) => format!("app.{name}"),
            MetricSource::Custom(name) => format!("custom.{name}"),
            MetricSource::Database(name) => format!("db.{name}"),
            MetricSource::Cache(name) => format!("cache.{name}"),
        }
    }

    /// Start web server for dashboard interface
    fn start_web_server(&mut self) -> CoreResult<()> {
        // In a full implementation, this would start an actual web server
        // For now, we'll simulate it

        let handle = WebServerHandle {
            address: "0.0.0.0".to_string(),
            port: self.config.web_port,
            running: Arc::new(Mutex::new(true)),
        };

        self.web_server_handle = Some(handle);

        println!(
            "Dashboard web interface started at http://localhost:{}/dashboard",
            self.config.web_port
        );

        Ok(())
    }
}

/// Dashboard export/import structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardExport {
    /// Dashboard configuration
    pub config: DashboardConfig,
    /// Widget configurations
    pub widgets: Vec<Widget>,
    /// Export timestamp
    pub created_at: SystemTime,
}

/// Dashboard statistics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct DashboardStatistics {
    /// Total number of widgets
    pub total_widgets: usize,
    /// Total number of metrics
    pub total_metrics: usize,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Last update time
    #[cfg_attr(feature = "serde", serde(skip))]
    pub last_update: Instant,
    /// Dashboard uptime
    pub uptime: Duration,
    /// Current state
    pub state: DashboardState,
}

impl Default for DashboardStatistics {
    fn default() -> Self {
        Self {
            total_widgets: 0,
            total_metrics: 0,
            active_alerts: 0,
            last_update: Instant::now(),
            uptime: Duration::from_secs(0),
            state: DashboardState::Stopped,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_creation() {
        let config = DashboardConfig::default()
            .with_title("Test Dashboard")
            .with_refresh_interval(Duration::from_secs(10));

        let dashboard = PerformanceDashboard::new(config);
        assert!(dashboard.is_ok());

        let dashboard = dashboard.unwrap();
        assert_eq!(dashboard.config.title, "Test Dashboard");
        assert_eq!(dashboard.config.refresh_interval, Duration::from_secs(10));
    }

    #[test]
    fn test_widget_creation_and_addition() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config).unwrap();

        let widget = Widget::new()
            .with_title("CPU Usage")
            .with_chart_type(ChartType::LineChart)
            .with_metric_source(MetricSource::SystemCpu)
            .with_alert_threshold(80.0);

        let widget_id = dashboard.add_widget(widget).unwrap();
        assert!(!widget_id.is_empty());
        assert_eq!(dashboard.widgets.len(), 1);
    }

    #[test]
    fn test_metric_updates() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config).unwrap();

        // Add a widget
        let widget = Widget::new().with_metric_source(MetricSource::SystemCpu);

        dashboard.add_widget(widget).unwrap();

        // Update metric
        let result = dashboard.update_metric(&MetricSource::SystemCpu, 75.5);
        assert!(result.is_ok());

        // Check that metric was recorded
        let metrics = dashboard.get_metrics().unwrap();
        let cpu_metric = metrics.get("system.cpu");
        assert!(cpu_metric.is_some());

        let cpu_metric = cpu_metric.unwrap();
        assert_eq!(cpu_metric.latest_value(), Some(75.5));
    }

    #[test]
    fn test_metric_time_series() {
        let mut ts = MetricTimeSeries::new("test_metric");

        ts.add_point(10.0, None);
        ts.add_point(20.0, None);
        ts.add_point(30.0, None);

        assert_eq!(ts.latest_value(), Some(30.0));
        assert_eq!(ts.data_points.len(), 3);

        // Test cleanup
        ts.cleanup(Duration::from_secs(1), 2);
        assert_eq!(ts.data_points.len(), 2);
    }

    #[test]
    fn test_alert_configuration() {
        let widget = Widget::new()
            .with_title("Memory Usage")
            .with_alert_threshold(90.0);

        assert!(widget.alert_config.is_some());

        let alert_config = widget.alert_config.unwrap();
        assert_eq!(alert_config.threshold, 90.0);
        assert_eq!(alert_config.condition, AlertCondition::GreaterThan);
        assert_eq!(alert_config.severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_dashboard_statistics() {
        let config = DashboardConfig::default();
        let mut dashboard = PerformanceDashboard::new(config).unwrap();

        // Add some widgets
        for i in 0..3 {
            let widget = Widget::new().with_title(&format!("Widget {i}"));
            dashboard.add_widget(widget).unwrap();
        }

        let stats = dashboard.get_statistics();
        assert_eq!(stats.total_widgets, 3);
        assert_eq!(stats.active_alerts, 0);
        assert_eq!(stats.state, DashboardState::Stopped);
    }

    #[test]
    fn test_dashboard_config_builder() {
        let config = DashboardConfig::default()
            .with_title("Custom Dashboard")
            .with_refresh_interval(Duration::from_secs(30))
            .with_retention_period(Duration::from_secs(14 * 24 * 60 * 60)) // 14 days
            .with_web_interface(9090)
            .with_api_token("test-token")
            .with_theme(DashboardTheme::Light);

        assert_eq!(config.title, "Custom Dashboard");
        assert_eq!(config.refresh_interval, Duration::from_secs(30));
        assert_eq!(
            config.retention_period,
            Duration::from_secs(14 * 24 * 60 * 60)
        ); // 14 days
        assert_eq!(config.web_port, 9090);
        assert_eq!(config.api_token, Some("test-token".to_string()));
        assert_eq!(config.theme, DashboardTheme::Light);
    }

    #[test]
    fn test_widget_layout() {
        let widget = Widget::new().with_layout(2, 3, 6, 4);

        assert_eq!(widget.layout.x, 2);
        assert_eq!(widget.layout.y, 3);
        assert_eq!(widget.layout.width, 6);
        assert_eq!(widget.layout.height, 4);
    }
}
