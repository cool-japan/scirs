//! Core types, configuration, and traits for advanced streaming metrics
//!
//! This module contains the fundamental types and configurations used throughout
//! the advanced streaming metrics system.

#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Configuration for streaming metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    /// Base window size
    pub base_window_size: usize,
    /// Maximum window size
    pub max_window_size: usize,
    /// Minimum window size
    pub min_window_size: usize,
    /// Drift detection sensitivity
    pub drift_sensitivity: f64,
    /// Warning threshold for drift
    pub warning_threshold: f64,
    /// Drift threshold for adaptation
    pub drift_threshold: f64,
    /// Enable adaptive windowing
    pub adaptive_windowing: bool,
    /// Window adaptation strategy
    pub adaptation_strategy: WindowAdaptationStrategy,
    /// Enable concept drift detection
    pub enable_drift_detection: bool,
    /// Drift detection methods
    pub drift_detection_methods: Vec<DriftDetectionMethod>,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Anomaly detection algorithm
    pub anomaly_algorithm: AnomalyDetectionAlgorithm,
    /// Performance monitoring interval
    pub monitoring_interval: Duration,
    /// Enable real-time alerts
    pub enable_alerts: bool,
    /// Alert configuration
    pub alert_config: AlertConfig,
}

/// Window adaptation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAdaptationStrategy {
    /// Fixed window size
    Fixed,
    /// Exponential decay-based adaptation
    ExponentialDecay { decay_rate: f64 },
    /// Performance-based adaptation
    PerformanceBased { target_accuracy: f64 },
    /// Drift-based adaptation
    DriftBased,
    /// Hybrid approach combining multiple strategies
    Hybrid {
        strategies: Vec<WindowAdaptationStrategy>,
        weights: Vec<f64>,
    },
    /// Machine learning-based adaptation
    MLBased { model_type: String },
}

/// Drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    /// ADWIN (Adaptive Windowing)
    Adwin { confidence: f64 },
    /// DDM (Drift Detection Method)
    Ddm {
        warning_level: f64,
        drift_level: f64,
    },
    /// EDDM (Early Drift Detection Method)
    Eddm { alpha: f64, beta: f64 },
    /// Page-Hinkley Test
    PageHinkley { threshold: f64, alpha: f64 },
    /// CUSUM (Cumulative Sum)
    Cusum {
        threshold: f64,
        drift_threshold: f64,
    },
    /// Kolmogorov-Smirnov Test
    KolmogorovSmirnov { p_value_threshold: f64 },
    /// Ensemble of multiple methods
    Ensemble { methods: Vec<DriftDetectionMethod> },
    /// Custom drift detection
    Custom {
        name: String,
        parameters: HashMap<String, f64>,
    },
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical z-score based
    ZScore { threshold: f64 },
    /// Isolation Forest
    IsolationForest { contamination: f64 },
    /// One-Class SVM
    OneClassSvm { nu: f64 },
    /// Local Outlier Factor
    LocalOutlierFactor { n_neighbors: usize },
    /// DBSCAN-based anomaly detection
    Dbscan { eps: f64, min_samples: usize },
    /// Autoencoder-based
    Autoencoder { threshold: f64 },
    /// Ensemble of multiple algorithms
    Ensemble {
        algorithms: Vec<AnomalyDetectionAlgorithm>,
    },
}

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Enable email alerts
    pub email_enabled: bool,
    /// Email addresses for alerts
    pub email_addresses: Vec<String>,
    /// Enable webhook alerts
    pub webhook_enabled: bool,
    /// Webhook URLs
    pub webhook_urls: Vec<String>,
    /// Enable log alerts
    pub log_enabled: bool,
    /// Log file path
    pub log_file: Option<String>,
    /// Alert severity levels
    pub severity_levels: HashMap<String, AlertSeverity>,
    /// Alert rate limiting
    pub rate_limit: Duration,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Concept drift detector trait
pub trait ConceptDriftDetector<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum>:
    std::fmt::Debug
{
    /// Update detector with new prediction
    fn update(&mut self, prediction_correct: bool, error: F) -> Result<DriftDetectionResult>;

    /// Get current detection status
    fn get_status(&self) -> DriftStatus;

    /// Reset detector state
    fn reset(&mut self);

    /// Get detector configuration
    fn get_config(&self) -> HashMap<String, f64>;

    /// Get detection statistics
    fn get_statistics(&self) -> DriftStatistics<F>;
}

/// Drift detection result
#[derive(Debug, Clone)]
pub struct DriftDetectionResult {
    pub status: DriftStatus,
    pub confidence: f64,
    pub change_point: Option<usize>,
    pub statistics: HashMap<String, f64>,
}

/// Drift status
#[derive(Debug, Clone, PartialEq)]
pub enum DriftStatus {
    Stable,
    Warning,
    Drift,
    Unknown,
}

/// Drift detection statistics
#[derive(Debug, Clone)]
pub struct DriftStatistics<F: Float + std::fmt::Debug> {
    pub samples_since_reset: usize,
    pub warnings_count: usize,
    pub drifts_count: usize,
    pub current_error_rate: F,
    pub baseline_error_rate: F,
    pub drift_score: F,
    pub last_detection_time: Option<SystemTime>,
}

/// Streaming metric trait
pub trait StreamingMetric<F: Float> {
    fn update(&mut self, true_value: F, predicted_value: F) -> Result<()>;
    fn get_value(&self) -> F;
    fn reset(&mut self);
    fn get_name(&self) -> &str;
    fn get_confidence(&self) -> F;
}

/// Ensemble aggregation strategies
#[derive(Debug, Clone)]
pub enum EnsembleAggregation {
    WeightedAverage,
    Majority,
    Maximum,
    Minimum,
    Median,
    Stacking { meta_learner: String },
}

/// Data point in the history buffer
#[derive(Debug, Clone)]
pub struct DataPoint<F: Float + std::fmt::Debug> {
    pub true_value: F,
    pub predicted_value: F,
    pub error: F,
    pub confidence: F,
    pub features: Option<Vec<F>>,
}

/// Alert message
#[derive(Debug, Clone)]
pub struct Alert {
    pub id: String,
    pub timestamp: Instant,
    pub severity: AlertSeverity,
    pub title: String,
    pub message: String,
    pub data: HashMap<String, String>,
    pub tags: Vec<String>,
}

/// Sent alert record
#[derive(Debug, Clone)]
pub struct SentAlert {
    pub alert: Alert,
    pub sent_at: Instant,
    pub channels: Vec<String>,
    pub success: bool,
    pub error_message: Option<String>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            base_window_size: 1000,
            max_window_size: 10000,
            min_window_size: 100,
            drift_sensitivity: 0.05,
            warning_threshold: 0.5,
            drift_threshold: 0.8,
            adaptive_windowing: true,
            adaptation_strategy: WindowAdaptationStrategy::DriftBased,
            enable_drift_detection: true,
            drift_detection_methods: vec![
                DriftDetectionMethod::Adwin { confidence: 0.95 },
                DriftDetectionMethod::Ddm {
                    warning_level: 2.0,
                    drift_level: 3.0,
                },
            ],
            enable_anomaly_detection: true,
            anomaly_algorithm: AnomalyDetectionAlgorithm::ZScore { threshold: 3.0 },
            monitoring_interval: Duration::from_secs(60),
            enable_alerts: false,
            alert_config: AlertConfig::default(),
        }
    }
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            email_enabled: false,
            email_addresses: Vec::new(),
            webhook_enabled: false,
            webhook_urls: Vec::new(),
            log_enabled: true,
            log_file: None,
            severity_levels: HashMap::new(),
            rate_limit: Duration::from_secs(300),
        }
    }
}
