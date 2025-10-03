//! Real-time Bifurcation Monitoring
//!
//! This module contains structures for real-time monitoring, alerting,
//! and adaptive threshold systems for bifurcation detection.

use crate::analysis::types::BifurcationType;
use scirs2_core::ndarray::Array1;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

/// Real-time bifurcation monitoring system
#[derive(Debug)]
pub struct RealTimeBifurcationMonitor {
    /// Streaming data buffer
    pub data_buffer: Arc<Mutex<VecDeque<Array1<f64>>>>,
    /// Prediction models
    pub prediction_models: Vec<super::neural_network::BifurcationPredictionNetwork>,
    /// Alert system configuration
    pub alert_system: AlertSystemConfig,
    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
    /// Performance tracker
    pub performance_tracker: PerformanceTracker,
    /// Adaptive threshold system
    pub adaptive_thresholds: AdaptiveThresholdSystem,
}

/// Alert system configuration
#[derive(Debug, Clone, Default)]
pub struct AlertSystemConfig {
    /// Alert thresholds for different bifurcation types
    pub alert_thresholds: HashMap<BifurcationType, f64>,
    /// Alert escalation levels
    pub escalation_levels: Vec<EscalationLevel>,
    /// Notification methods
    pub notification_methods: Vec<NotificationMethod>,
    /// Alert suppression configuration
    pub suppression_config: AlertSuppressionConfig,
}

/// Alert escalation levels
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Level name
    pub level_name: String,
    /// Threshold for this level
    pub threshold: f64,
    /// Time delay before escalation
    pub escalation_delay: std::time::Duration,
    /// Actions to take at this level
    pub actions: Vec<AlertAction>,
}

/// Alert actions
#[derive(Debug, Clone)]
pub enum AlertAction {
    /// Log alert to file
    LogToFile(String),
    /// Send email notification
    SendEmail(String),
    /// Trigger system shutdown
    SystemShutdown,
    /// Execute custom script
    ExecuteScript(String),
    /// Update model parameters
    UpdateModel,
}

/// Notification methods
#[derive(Debug, Clone)]
pub enum NotificationMethod {
    /// Email notification
    Email {
        recipients: Vec<String>,
        smtp_config: String,
    },
    /// SMS notification
    SMS {
        phone_numbers: Vec<String>,
        service_config: String,
    },
    /// Webhook notification
    Webhook {
        url: String,
        headers: HashMap<String, String>,
    },
    /// File logging
    FileLog { log_path: String, format: LogFormat },
}

/// Log format options
#[derive(Debug, Clone, Copy)]
pub enum LogFormat {
    JSON,
    CSV,
    PlainText,
    XML,
}

/// Alert suppression configuration
#[derive(Debug, Clone)]
pub struct AlertSuppressionConfig {
    /// Minimum time between alerts of same type
    pub min_interval: std::time::Duration,
    /// Maximum number of alerts per time window
    pub max_alerts_per_window: usize,
    /// Time window for alert counting
    pub time_window: std::time::Duration,
    /// Suppress alerts during maintenance
    pub maintenance_mode: bool,
}

/// Real-time monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Data sampling rate
    pub sampling_rate: f64,
    /// Buffer size for streaming data
    pub buffer_size: usize,
    /// Prediction update frequency
    pub update_frequency: f64,
    /// Model ensemble configuration
    pub ensemble_config: MonitoringEnsembleConfig,
    /// Data preprocessing pipeline
    pub preprocessing: super::preprocessing::PreprocessingPipeline,
}

/// Ensemble configuration for monitoring
#[derive(Debug, Clone)]
pub struct MonitoringEnsembleConfig {
    /// Use multiple models for robustness
    pub use_ensemble: bool,
    /// Voting strategy for ensemble
    pub voting_strategy: VotingStrategy,
    /// Confidence threshold for predictions
    pub confidence_threshold: f64,
    /// Agreement threshold among models
    pub agreement_threshold: f64,
}

/// Voting strategies for ensemble
#[derive(Debug, Clone, Copy)]
pub enum VotingStrategy {
    /// Majority voting
    Majority,
    /// Weighted voting by model performance
    Weighted,
    /// Confidence-based voting
    ConfidenceBased,
    /// Unanimous voting (all models agree)
    Unanimous,
}

/// Performance tracking for monitoring system
#[derive(Debug, Clone)]
pub struct PerformanceTracker {
    /// Latency metrics
    pub latency_metrics: LatencyMetrics,
    /// Accuracy metrics
    pub accuracy_metrics: AccuracyMetrics,
    /// Resource usage metrics
    pub resource_metrics: ResourceMetrics,
    /// Alert performance metrics
    pub alert_metrics: AlertMetrics,
}

/// Latency metrics for real-time monitoring
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// Average prediction latency
    pub avg_prediction_latency: f64,
    /// Maximum prediction latency
    pub max_prediction_latency: f64,
    /// 95th percentile latency
    pub p95_latency: f64,
    /// Data processing latency
    pub processing_latency: f64,
}

/// Accuracy metrics for monitoring
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// True positive rate
    pub true_positive_rate: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Precision score
    pub precision: f64,
    /// Recall score
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: usize,
    /// Network bandwidth usage
    pub network_usage: f64,
    /// Disk I/O operations
    pub disk_io: f64,
}

/// Alert performance metrics
#[derive(Debug, Clone)]
pub struct AlertMetrics {
    /// Total number of alerts generated
    pub total_alerts: usize,
    /// Number of true alerts
    pub true_alerts: usize,
    /// Number of false alerts
    pub false_alerts: usize,
    /// Average alert response time
    pub avg_response_time: f64,
}

/// Adaptive threshold system
#[derive(Debug, Clone)]
pub struct AdaptiveThresholdSystem {
    /// Current threshold values
    pub current_thresholds: HashMap<BifurcationType, f64>,
    /// Threshold adaptation method
    pub adaptation_method: ThresholdAdaptationMethod,
    /// Feedback mechanism for threshold adjustment
    pub feedback_mechanism: FeedbackMechanism,
    /// Historical performance data
    pub performance_history: Vec<f64>,
}

/// Methods for adapting thresholds
#[derive(Debug, Clone, Copy)]
pub enum ThresholdAdaptationMethod {
    /// Fixed thresholds (no adaptation)
    Fixed,
    /// Statistical-based adaptation
    Statistical,
    /// Machine learning-based adaptation
    MachineLearning,
    /// Feedback-based adaptation
    FeedbackBased,
}

/// Feedback mechanisms for threshold adjustment
#[derive(Debug, Clone, Copy)]
pub enum FeedbackMechanism {
    /// Manual feedback from operators
    Manual,
    /// Automated feedback from validation
    Automated,
    /// Hybrid manual and automated
    Hybrid,
}

impl Default for AlertSuppressionConfig {
    fn default() -> Self {
        Self {
            min_interval: std::time::Duration::from_secs(60),
            max_alerts_per_window: 10,
            time_window: std::time::Duration::from_secs(3600),
            maintenance_mode: false,
        }
    }
}

impl Default for AlertMetrics {
    fn default() -> Self {
        Self {
            total_alerts: 0,
            true_alerts: 0,
            false_alerts: 0,
            avg_response_time: 0.0,
        }
    }
}

impl Default for AdaptiveThresholdSystem {
    fn default() -> Self {
        Self {
            current_thresholds: HashMap::new(),
            adaptation_method: ThresholdAdaptationMethod::Statistical,
            feedback_mechanism: FeedbackMechanism::Automated,
            performance_history: Vec::new(),
        }
    }
}
