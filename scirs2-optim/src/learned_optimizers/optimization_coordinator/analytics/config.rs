//! Configuration and core types for optimization analytics
//!
//! This module contains configuration structures and fundamental types
//! used throughout the analytics system for optimization monitoring,
//! performance analysis, and reporting.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use num_traits::Float;

/// Analytics engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    /// Enable performance monitoring
    pub enable_performance_monitoring: bool,

    /// Enable convergence analysis
    pub enable_convergence_analysis: bool,

    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,

    /// Enable pattern detection
    pub enable_pattern_detection: bool,

    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,

    /// Enable trend analysis
    pub enable_trend_analysis: bool,

    /// Enable real-time dashboard
    pub enable_dashboard: bool,

    /// Performance monitoring configuration
    pub performance_config: PerformanceMonitoringConfig,

    /// Convergence analysis configuration
    pub convergence_config: ConvergenceAnalysisConfig,

    /// Resource monitoring configuration
    pub resource_config: ResourceMonitoringConfig,

    /// Pattern detection configuration
    pub pattern_config: PatternDetectionConfig,

    /// Anomaly detection configuration
    pub anomaly_config: AnomalyDetectionConfig,

    /// Trend analysis configuration
    pub trend_config: TrendAnalysisConfig,

    /// Dashboard configuration
    pub dashboard_config: DashboardConfig,

    /// Report generation configuration
    pub reporting_config: ReportingConfig,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    /// Sampling interval for performance metrics
    pub sampling_interval: Duration,

    /// Maximum number of performance snapshots to retain
    pub max_snapshots: usize,

    /// Enable detailed profiling
    pub enable_detailed_profiling: bool,

    /// Performance thresholds for alerting
    pub performance_thresholds: PerformanceThresholds<f64>,

    /// Benchmark configuration
    pub benchmark_config: BenchmarkConfig,

    /// Model performance tracking
    pub model_config: ModelTrackingConfig,
}

/// Convergence analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysisConfig {
    /// Detection methods to use
    pub detection_methods: Vec<ConvergenceDetectionMethod>,

    /// Parameters for convergence detection
    pub detection_parameters: ConvergenceParameters<f64>,

    /// Window size for convergence analysis
    pub analysis_window_size: usize,

    /// Minimum steps before convergence detection
    pub min_steps_for_detection: usize,

    /// Enable early stopping
    pub enable_early_stopping: bool,

    /// Early stopping configuration
    pub early_stopping_config: EarlyStoppingConfig<f64>,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Monitor CPU usage
    pub monitor_cpu: bool,

    /// Monitor memory usage
    pub monitor_memory: bool,

    /// Monitor GPU usage
    pub monitor_gpu: bool,

    /// Monitor disk I/O
    pub monitor_disk: bool,

    /// Monitor network I/O
    pub monitor_network: bool,

    /// Sampling frequency for resource monitoring
    pub sampling_frequency: Duration,

    /// Resource usage history size
    pub history_size: usize,

    /// Resource thresholds for alerting
    pub resource_thresholds: ResourceThresholds,
}

/// Pattern detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDetectionConfig {
    /// Enable loss pattern detection
    pub enable_loss_patterns: bool,

    /// Enable gradient pattern detection
    pub enable_gradient_patterns: bool,

    /// Enable performance pattern detection
    pub enable_performance_patterns: bool,

    /// Pattern detection sensitivity
    pub detection_sensitivity: f64,

    /// Minimum pattern length
    pub min_pattern_length: usize,

    /// Maximum pattern length
    pub max_pattern_length: usize,

    /// Pattern matching tolerance
    pub pattern_tolerance: f64,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Anomaly detection methods
    pub detection_methods: Vec<AnomalyDetectionMethod>,

    /// Anomaly detection sensitivity
    pub sensitivity: f64,

    /// Window size for anomaly detection
    pub window_size: usize,

    /// Confidence threshold for anomaly reporting
    pub confidence_threshold: f64,

    /// Enable real-time anomaly alerts
    pub enable_real_time_alerts: bool,

    /// Anomaly alert configuration
    pub alert_config: AnomalyAlertConfig,
}

/// Trend analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisConfig {
    /// Forecasting methods to use
    pub forecasting_methods: Vec<ForecastingMethod>,

    /// Prediction horizon (number of steps)
    pub prediction_horizon: usize,

    /// Trend detection window size
    pub trend_window_size: usize,

    /// Seasonal decomposition enabled
    pub enable_seasonal_decomposition: bool,

    /// Confidence intervals for predictions
    pub confidence_intervals: Vec<f64>,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Update frequency for dashboard
    pub update_frequency: Duration,

    /// Maximum data points to display
    pub max_display_points: usize,

    /// Enable interactive features
    pub enable_interactive: bool,

    /// Dashboard layout configuration
    pub layout_config: DashboardLayoutConfig,

    /// Chart configurations
    pub chart_configs: HashMap<String, ChartConfig>,
}

/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingConfig {
    /// Report generation frequency
    pub generation_frequency: Duration,

    /// Report formats to generate
    pub output_formats: Vec<ReportFormat>,

    /// Include detailed analysis
    pub include_detailed_analysis: bool,

    /// Include visualizations
    pub include_visualizations: bool,

    /// Report retention period
    pub retention_period: Duration,

    /// Email notification settings
    pub notification_config: Option<NotificationConfig>,
}

/// Performance thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds<T: Float> {
    /// Maximum acceptable loss value
    pub max_loss: Option<T>,

    /// Minimum acceptable accuracy
    pub min_accuracy: Option<T>,

    /// Maximum training time per epoch
    pub max_training_time: Option<Duration>,

    /// Maximum memory usage
    pub max_memory_usage: Option<f64>,

    /// Minimum convergence rate
    pub min_convergence_rate: Option<T>,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Enable automated benchmarking
    pub enable_auto_benchmark: bool,

    /// Benchmark frequency
    pub benchmark_frequency: Duration,

    /// Reference datasets for benchmarking
    pub reference_datasets: Vec<String>,

    /// Performance metrics to track
    pub tracked_metrics: Vec<String>,

    /// Comparison baselines
    pub baselines: HashMap<String, f64>,
}

/// Model tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrackingConfig {
    /// Track model predictions
    pub track_predictions: bool,

    /// Track model parameters
    pub track_parameters: bool,

    /// Track gradient norms
    pub track_gradients: bool,

    /// Model versioning enabled
    pub enable_versioning: bool,

    /// Model comparison enabled
    pub enable_comparison: bool,
}

/// Convergence detection methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConvergenceDetectionMethod {
    /// Loss-based convergence detection
    LossBased,
    /// Gradient-norm based detection
    GradientNorm,
    /// Parameter change based detection
    ParameterChange,
    /// Validation metric based detection
    ValidationMetric,
    /// Statistical significance test
    StatisticalTest,
    /// Machine learning based detection
    MLBased,
}

/// Convergence detection parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceParameters<T: Float> {
    /// Tolerance for loss change
    pub loss_tolerance: T,

    /// Tolerance for gradient norm
    pub gradient_tolerance: T,

    /// Tolerance for parameter change
    pub parameter_tolerance: T,

    /// Window size for averaging
    pub window_size: usize,

    /// Minimum improvement threshold
    pub min_improvement: T,

    /// Patience for early stopping
    pub patience: usize,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig<T: Float> {
    /// Enable early stopping
    pub enabled: bool,

    /// Metric to monitor for early stopping
    pub monitor_metric: String,

    /// Minimum change to qualify as improvement
    pub min_delta: T,

    /// Number of epochs with no improvement after which training will be stopped
    pub patience: usize,

    /// Whether to restore best weights
    pub restore_best_weights: bool,

    /// Mode for monitoring (min or max)
    pub mode: EarlyStoppingMode,
}

/// Early stopping monitoring mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EarlyStoppingMode {
    /// Monitor for minimum value (e.g., loss)
    Min,
    /// Monitor for maximum value (e.g., accuracy)
    Max,
}

/// Resource monitoring thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceThresholds {
    /// CPU usage threshold (percentage)
    pub cpu_threshold: f64,

    /// Memory usage threshold (percentage)
    pub memory_threshold: f64,

    /// GPU usage threshold (percentage)
    pub gpu_threshold: f64,

    /// Disk usage threshold (percentage)
    pub disk_threshold: f64,

    /// Network bandwidth threshold (MB/s)
    pub network_threshold: f64,
}

/// Anomaly detection methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    /// Statistical outlier detection
    Statistical,
    /// Isolation Forest
    IsolationForest,
    /// One-Class SVM
    OneClassSVM,
    /// Autoencoder-based detection
    Autoencoder,
    /// Time series anomaly detection
    TimeSeries,
    /// Ensemble methods
    Ensemble,
}

/// Anomaly alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlertConfig {
    /// Alert severity levels
    pub severity_levels: Vec<AlertSeverity>,

    /// Alert channels
    pub alert_channels: Vec<AlertChannel>,

    /// Alert frequency limits
    pub frequency_limits: HashMap<AlertSeverity, Duration>,

    /// Alert suppression rules
    pub suppression_rules: Vec<SuppressionRule>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert delivery channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    /// Email notification
    Email { recipients: Vec<String> },
    /// Slack notification
    Slack { webhook_url: String, channel: String },
    /// Discord notification
    Discord { webhook_url: String },
    /// SMS notification
    Sms { phone_numbers: Vec<String> },
    /// System log
    SystemLog,
    /// Dashboard notification
    Dashboard,
}

/// Alert suppression rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    /// Rule name
    pub name: String,

    /// Conditions for suppression
    pub conditions: Vec<SuppressionCondition>,

    /// Suppression duration
    pub duration: Duration,

    /// Rule enabled
    pub enabled: bool,
}

/// Suppression condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionCondition {
    /// Field to check
    pub field: String,

    /// Condition operator
    pub operator: ComparisonOperator,

    /// Value to compare against
    pub value: String,
}

/// Comparison operators for suppression conditions
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Contains,
    StartsWith,
    EndsWith,
}

/// Forecasting methods for trend analysis
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ForecastingMethod {
    /// Linear regression
    LinearRegression,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// ARIMA model
    Arima,
    /// Prophet
    Prophet,
    /// Neural network based
    NeuralNetwork,
    /// Ensemble of methods
    Ensemble,
}

/// Dashboard layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardLayoutConfig {
    /// Grid layout (rows, columns)
    pub grid_layout: (usize, usize),

    /// Panel configurations
    pub panels: Vec<PanelConfig>,

    /// Theme settings
    pub theme: DashboardTheme,

    /// Responsive design enabled
    pub responsive: bool,
}

/// Individual panel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanelConfig {
    /// Panel identifier
    pub id: String,

    /// Panel title
    pub title: String,

    /// Panel type
    pub panel_type: PanelType,

    /// Grid position (row, column)
    pub position: (usize, usize),

    /// Grid span (rows, columns)
    pub span: (usize, usize),

    /// Data source
    pub data_source: String,

    /// Refresh interval
    pub refresh_interval: Duration,
}

/// Dashboard panel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PanelType {
    /// Line chart
    LineChart,
    /// Bar chart
    BarChart,
    /// Scatter plot
    ScatterPlot,
    /// Heatmap
    Heatmap,
    /// Gauge
    Gauge,
    /// Table
    Table,
    /// Text display
    Text,
    /// Metric display
    Metric,
}

/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    /// Chart type
    pub chart_type: ChartType,

    /// X-axis configuration
    pub x_axis: AxisConfig,

    /// Y-axis configuration
    pub y_axis: AxisConfig,

    /// Color scheme
    pub color_scheme: Vec<String>,

    /// Animation settings
    pub animation: AnimationConfig,

    /// Interactive features
    pub interactions: InteractionConfig,
}

/// Chart types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Scatter,
    Area,
    Pie,
    Donut,
    Histogram,
    BoxPlot,
    Violin,
    Candlestick,
}

/// Chart axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxisConfig {
    /// Axis label
    pub label: String,

    /// Scale type
    pub scale: ScaleType,

    /// Minimum value
    pub min: Option<f64>,

    /// Maximum value
    pub max: Option<f64>,

    /// Grid lines enabled
    pub grid: bool,

    /// Tick configuration
    pub ticks: TickConfig,
}

/// Chart scale types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ScaleType {
    Linear,
    Logarithmic,
    Time,
    Category,
}

/// Tick configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickConfig {
    /// Number of ticks
    pub count: Option<usize>,

    /// Tick interval
    pub interval: Option<f64>,

    /// Tick format
    pub format: String,

    /// Rotation angle
    pub rotation: f64,
}

/// Animation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnimationConfig {
    /// Animation enabled
    pub enabled: bool,

    /// Animation duration
    pub duration: Duration,

    /// Easing function
    pub easing: EasingFunction,
}

/// Easing functions for animations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    ElasticIn,
    ElasticOut,
    BounceIn,
    BounceOut,
}

/// Chart interaction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionConfig {
    /// Zoom enabled
    pub zoom: bool,

    /// Pan enabled
    pub pan: bool,

    /// Hover enabled
    pub hover: bool,

    /// Selection enabled
    pub selection: bool,

    /// Brush enabled
    pub brush: bool,
}

/// Dashboard theme
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Primary color
    pub primary_color: String,

    /// Secondary color
    pub secondary_color: String,

    /// Background color
    pub background_color: String,

    /// Text color
    pub text_color: String,

    /// Font family
    pub font_family: String,

    /// Font size
    pub font_size: usize,
}

/// Report output formats
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReportFormat {
    Pdf,
    Html,
    Markdown,
    Json,
    Csv,
    Excel,
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Email settings
    pub email: Option<EmailConfig>,

    /// Slack settings
    pub slack: Option<SlackConfig>,

    /// Discord settings
    pub discord: Option<DiscordConfig>,
}

/// Email notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailConfig {
    /// SMTP server
    pub smtp_server: String,

    /// SMTP port
    pub smtp_port: u16,

    /// Username
    pub username: String,

    /// Password (should be encrypted)
    pub password: String,

    /// From address
    pub from_address: String,

    /// Recipients
    pub recipients: Vec<String>,

    /// Use TLS
    pub use_tls: bool,
}

/// Slack notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlackConfig {
    /// Webhook URL
    pub webhook_url: String,

    /// Default channel
    pub channel: String,

    /// Bot username
    pub username: String,

    /// Bot icon
    pub icon: Option<String>,
}

/// Discord notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordConfig {
    /// Webhook URL
    pub webhook_url: String,

    /// Bot username
    pub username: String,

    /// Bot avatar URL
    pub avatar_url: Option<String>,
}

impl Default for AnalyticsConfig {
    fn default() -> Self {
        Self {
            enable_performance_monitoring: true,
            enable_convergence_analysis: true,
            enable_resource_monitoring: true,
            enable_pattern_detection: true,
            enable_anomaly_detection: true,
            enable_trend_analysis: true,
            enable_dashboard: true,
            performance_config: PerformanceMonitoringConfig::default(),
            convergence_config: ConvergenceAnalysisConfig::default(),
            resource_config: ResourceMonitoringConfig::default(),
            pattern_config: PatternDetectionConfig::default(),
            anomaly_config: AnomalyDetectionConfig::default(),
            trend_config: TrendAnalysisConfig::default(),
            dashboard_config: DashboardConfig::default(),
            reporting_config: ReportingConfig::default(),
        }
    }
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            sampling_interval: Duration::from_secs(10),
            max_snapshots: 1000,
            enable_detailed_profiling: true,
            performance_thresholds: PerformanceThresholds::default(),
            benchmark_config: BenchmarkConfig::default(),
            model_config: ModelTrackingConfig::default(),
        }
    }
}

impl Default for ConvergenceAnalysisConfig {
    fn default() -> Self {
        Self {
            detection_methods: vec![
                ConvergenceDetectionMethod::LossBased,
                ConvergenceDetectionMethod::GradientNorm,
            ],
            detection_parameters: ConvergenceParameters::default(),
            analysis_window_size: 100,
            min_steps_for_detection: 50,
            enable_early_stopping: true,
            early_stopping_config: EarlyStoppingConfig::default(),
        }
    }
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            monitor_cpu: true,
            monitor_memory: true,
            monitor_gpu: true,
            monitor_disk: true,
            monitor_network: true,
            sampling_frequency: Duration::from_secs(5),
            history_size: 1000,
            resource_thresholds: ResourceThresholds::default(),
        }
    }
}

impl Default for PatternDetectionConfig {
    fn default() -> Self {
        Self {
            enable_loss_patterns: true,
            enable_gradient_patterns: true,
            enable_performance_patterns: true,
            detection_sensitivity: 0.8,
            min_pattern_length: 5,
            max_pattern_length: 50,
            pattern_tolerance: 0.1,
        }
    }
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            detection_methods: vec![
                AnomalyDetectionMethod::Statistical,
                AnomalyDetectionMethod::IsolationForest,
            ],
            sensitivity: 0.95,
            window_size: 100,
            confidence_threshold: 0.9,
            enable_real_time_alerts: true,
            alert_config: AnomalyAlertConfig::default(),
        }
    }
}

impl Default for TrendAnalysisConfig {
    fn default() -> Self {
        Self {
            forecasting_methods: vec![
                ForecastingMethod::LinearRegression,
                ForecastingMethod::ExponentialSmoothing,
            ],
            prediction_horizon: 50,
            trend_window_size: 200,
            enable_seasonal_decomposition: true,
            confidence_intervals: vec![0.8, 0.95],
        }
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_frequency: Duration::from_secs(1),
            max_display_points: 1000,
            enable_interactive: true,
            layout_config: DashboardLayoutConfig::default(),
            chart_configs: HashMap::new(),
        }
    }
}

impl Default for ReportingConfig {
    fn default() -> Self {
        Self {
            generation_frequency: Duration::from_secs(3600), // Hourly
            output_formats: vec![ReportFormat::Html, ReportFormat::Pdf],
            include_detailed_analysis: true,
            include_visualizations: true,
            retention_period: Duration::from_secs(30 * 24 * 3600), // 30 days
            notification_config: None,
        }
    }
}

impl<T: Float> Default for PerformanceThresholds<T> {
    fn default() -> Self {
        Self {
            max_loss: None,
            min_accuracy: None,
            max_training_time: Some(Duration::from_secs(3600)),
            max_memory_usage: Some(0.9),
            min_convergence_rate: None,
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_auto_benchmark: true,
            benchmark_frequency: Duration::from_secs(24 * 3600), // Daily
            reference_datasets: vec!["mnist".to_string(), "cifar10".to_string()],
            tracked_metrics: vec!["accuracy".to_string(), "loss".to_string()],
            baselines: HashMap::new(),
        }
    }
}

impl Default for ModelTrackingConfig {
    fn default() -> Self {
        Self {
            track_predictions: true,
            track_parameters: true,
            track_gradients: true,
            enable_versioning: true,
            enable_comparison: true,
        }
    }
}

impl<T: Float> Default for ConvergenceParameters<T> {
    fn default() -> Self {
        Self {
            loss_tolerance: T::from(1e-6).unwrap(),
            gradient_tolerance: T::from(1e-8).unwrap(),
            parameter_tolerance: T::from(1e-6).unwrap(),
            window_size: 10,
            min_improvement: T::from(1e-4).unwrap(),
            patience: 10,
        }
    }
}

impl<T: Float> Default for EarlyStoppingConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            monitor_metric: "val_loss".to_string(),
            min_delta: T::from(1e-4).unwrap(),
            patience: 10,
            restore_best_weights: true,
            mode: EarlyStoppingMode::Min,
        }
    }
}

impl Default for ResourceThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 85.0,
            gpu_threshold: 90.0,
            disk_threshold: 90.0,
            network_threshold: 100.0, // 100 MB/s
        }
    }
}

impl Default for AnomalyAlertConfig {
    fn default() -> Self {
        Self {
            severity_levels: vec![
                AlertSeverity::Low,
                AlertSeverity::Medium,
                AlertSeverity::High,
                AlertSeverity::Critical,
            ],
            alert_channels: vec![AlertChannel::SystemLog, AlertChannel::Dashboard],
            frequency_limits: HashMap::new(),
            suppression_rules: Vec::new(),
        }
    }
}

impl Default for DashboardLayoutConfig {
    fn default() -> Self {
        Self {
            grid_layout: (4, 3),
            panels: Vec::new(),
            theme: DashboardTheme::default(),
            responsive: true,
        }
    }
}

impl Default for DashboardTheme {
    fn default() -> Self {
        Self {
            primary_color: "#007bff".to_string(),
            secondary_color: "#6c757d".to_string(),
            background_color: "#ffffff".to_string(),
            text_color: "#343a40".to_string(),
            font_family: "Arial, sans-serif".to_string(),
            font_size: 14,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analytics_config_default() {
        let config = AnalyticsConfig::default();
        assert!(config.enable_performance_monitoring);
        assert!(config.enable_convergence_analysis);
        assert!(config.enable_anomaly_detection);
    }

    #[test]
    fn test_performance_thresholds() {
        let thresholds = PerformanceThresholds::<f64>::default();
        assert!(thresholds.max_training_time.is_some());
        assert!(thresholds.max_memory_usage.is_some());
    }

    #[test]
    fn test_convergence_parameters() {
        let params = ConvergenceParameters::<f32>::default();
        assert!(params.loss_tolerance > 0.0);
        assert!(params.gradient_tolerance > 0.0);
        assert!(params.window_size > 0);
    }

    #[test]
    fn test_alert_severity_ordering() {
        use std::cmp::Ordering;

        assert!(AlertSeverity::Critical > AlertSeverity::High);
        assert!(AlertSeverity::High > AlertSeverity::Medium);
        assert!(AlertSeverity::Medium > AlertSeverity::Low);
    }

    #[test]
    fn test_dashboard_theme_colors() {
        let theme = DashboardTheme::default();
        assert!(theme.primary_color.starts_with('#'));
        assert!(theme.secondary_color.starts_with('#'));
        assert_eq!(theme.font_size, 14);
    }
}