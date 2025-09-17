//! Advanced streaming metrics with concept drift detection and adaptive windowing
//!
//! This module provides sophisticated streaming evaluation capabilities including:
//! - Concept drift detection using statistical tests
//! - Adaptive windowing strategies
//! - Online anomaly detection
//! - Real-time performance monitoring
//! - Ensemble-based drift detection

pub mod alerts;
pub mod anomaly;
pub mod core;
pub mod drift_detection;
pub mod ensemble;
pub mod history;
pub mod neural;
pub mod performance;
pub mod window_management;

// Re-export main types and traits
pub use alerts::{AlertStatistics, AlertsManager};
pub use anomaly::{Anomaly, AnomalyDetector, AnomalyStatistics, AnomalyType};
pub use core::{
    Alert, AlertConfig, AlertSeverity, AnomalyDetectionAlgorithm, ConceptDriftDetector, DataPoint,
    DriftDetectionMethod, DriftDetectionResult, DriftStatistics, DriftStatus, EnsembleAggregation,
    SentAlert, StreamingConfig, StreamingMetric, WindowAdaptationStrategy,
};
pub use drift_detection::{AdwinDetector, DdmDetector, PageHinkleyDetector};
pub use ensemble::MetricEnsemble;
pub use history::HistoryBuffer;
pub use neural::{
    ActivationFunction, AdaptiveLearningScheduler, AttentionMechanism, AttentionType,
    AutoencoderNetwork, BanditAlgorithm, FeatureConfig, FeatureExtractionMethod,
    FeatureNormalization, FeatureSelectionNetwork, MultiArmedBandit, NetworkConfig,
    NeuralFeatureExtractor, ParameterConfiguration, ParameterType, RegretTracker, SchedulerType,
};
pub use performance::{PerformanceDegradation, PerformanceMonitor, PerformanceSnapshot};
pub use window_management::{
    AdaptationTrigger, AdaptiveWindowManager, StreamingStatistics, WindowAdaptation,
};

use crate::error::{MetricsError, Result};
use num_traits::Float;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Advanced streaming metrics with concept drift detection
#[derive(Debug)]
pub struct AdaptiveStreamingMetrics<F: Float + std::fmt::Debug + Send + Sync> {
    /// Configuration for the streaming system
    config: StreamingConfig,
    /// Drift detection algorithms
    drift_detectors: Vec<Box<dyn ConceptDriftDetector<F> + Send + Sync>>,
    /// Adaptive window manager
    window_manager: AdaptiveWindowManager<F>,
    /// Performance monitor
    performance_monitor: PerformanceMonitor<F>,
    /// Anomaly detector
    anomaly_detector: AnomalyDetector<F>,
    /// Ensemble of base metrics
    metric_ensemble: MetricEnsemble<F>,
    /// Historical data buffer
    history_buffer: HistoryBuffer<F>,
    /// Current statistics
    current_stats: StreamingStatistics<F>,
    /// Alerts manager
    alerts_manager: AlertsManager,
}

/// Result of updating metrics
#[derive(Debug, Clone)]
pub struct UpdateResult<F: Float + std::fmt::Debug> {
    pub drift_detected: bool,
    pub drift_results: Vec<DriftDetectionResult>,
    pub anomaly_detected: bool,
    pub anomaly_result: Option<Anomaly<F>>,
    pub window_adapted: bool,
    pub adaptation_result: Option<WindowAdaptation>,
    pub processing_time: Duration,
    pub current_performance: HashMap<String, F>,
}

/// Anomaly detection summary
#[derive(Debug, Clone)]
pub struct AnomalySummary<F: Float + std::fmt::Debug> {
    pub total_anomalies: usize,
    pub recent_anomalies: Vec<Anomaly<F>>,
    pub anomaly_rate: F,
    pub statistics: AnomalyStatistics<F>,
}

impl<F: Float + std::fmt::Debug + Send + Sync + std::iter::Sum + std::ops::AddAssign + 'static>
    AdaptiveStreamingMetrics<F>
{
    /// Create new adaptive streaming metrics with configuration
    pub fn new(config: StreamingConfig) -> Result<Self> {
        let mut drift_detectors: Vec<Box<dyn ConceptDriftDetector<F> + Send + Sync>> = Vec::new();

        // Initialize drift detectors based on configuration
        for method in &config.drift_detection_methods {
            match method {
                DriftDetectionMethod::Adwin { confidence } => {
                    drift_detectors.push(Box::new(AdwinDetector::new(*confidence)?));
                }
                DriftDetectionMethod::Ddm {
                    warning_level,
                    drift_level,
                } => {
                    drift_detectors.push(Box::new(DdmDetector::new(*warning_level, *drift_level)));
                }
                DriftDetectionMethod::PageHinkley { threshold, alpha } => {
                    drift_detectors.push(Box::new(PageHinkleyDetector::new(*threshold, *alpha)));
                }
                _ => {
                    // For other methods, default to ADWIN
                    drift_detectors.push(Box::new(AdwinDetector::new(0.95)?));
                }
            }
        }

        let window_manager = AdaptiveWindowManager::new(
            config.base_window_size,
            config.min_window_size,
            config.max_window_size,
            config.adaptation_strategy.clone(),
        );

        let performance_monitor = PerformanceMonitor::new(config.monitoring_interval);
        let anomaly_detector = AnomalyDetector::new(config.anomaly_algorithm.clone())?;
        let metric_ensemble = MetricEnsemble::new();
        let history_buffer = HistoryBuffer::new(config.max_window_size);
        let current_stats = StreamingStatistics {
            total_samples: 0,
            correct_predictions: 0,
            current_accuracy: F::zero(),
            moving_average_accuracy: F::zero(),
            error_rate: F::zero(),
            drift_detected: false,
            anomalies_detected: 0,
            processing_rate: F::zero(),
            memory_usage: 0,
            last_update: Instant::now(),
        };
        let alerts_manager = AlertsManager::new(config.alert_config.clone());

        Ok(Self {
            config,
            drift_detectors,
            window_manager,
            performance_monitor,
            anomaly_detector,
            metric_ensemble,
            history_buffer,
            current_stats,
            alerts_manager,
        })
    }

    /// Update metrics with new prediction
    pub fn update(
        &mut self,
        true_value: F,
        predicted_value: F,
        confidence: Option<F>,
    ) -> Result<UpdateResult<F>> {
        let start_time = Instant::now();
        let prediction_correct = (true_value - predicted_value).abs() < F::from(0.001).unwrap();
        let error = (true_value - predicted_value).abs();

        // Update basic statistics
        self.current_stats.total_samples += 1;
        if prediction_correct {
            self.current_stats.correct_predictions += 1;
        }

        self.current_stats.current_accuracy = if self.current_stats.total_samples > 0 {
            F::from(self.current_stats.correct_predictions).unwrap()
                / F::from(self.current_stats.total_samples).unwrap()
        } else {
            F::zero()
        };

        self.current_stats.error_rate = F::one() - self.current_stats.current_accuracy;

        // Update moving average
        let alpha = F::from(0.1).unwrap();
        self.current_stats.moving_average_accuracy = alpha * self.current_stats.current_accuracy
            + (F::one() - alpha) * self.current_stats.moving_average_accuracy;

        // Add to history
        let data_point = DataPoint {
            true_value,
            predicted_value,
            error,
            confidence: confidence.unwrap_or(F::one()),
            features: None,
        };
        self.history_buffer.add_data_point(data_point);

        // Drift detection
        let mut drift_results = Vec::new();
        let mut drift_detected = false;

        if self.config.enable_drift_detection {
            for detector in &mut self.drift_detectors {
                let result = detector.update(prediction_correct, error)?;
                if result.status == DriftStatus::Drift {
                    drift_detected = true;
                }
                drift_results.push(result);
            }
        }

        self.current_stats.drift_detected = drift_detected;

        // Anomaly detection
        let mut anomaly_detected = false;
        let mut anomaly_result = None;

        if self.config.enable_anomaly_detection {
            match self.anomaly_detector.detect(error) {
                Ok(anomaly) => {
                    anomaly_detected = true;
                    anomaly_result = Some(anomaly);
                    self.current_stats.anomalies_detected += 1;
                }
                Err(_) => {
                    // No anomaly detected
                }
            }
        }

        // Window adaptation
        let mut window_adapted = false;
        let mut adaptation_result = None;

        if self.config.adaptive_windowing {
            if let Some(adaptation) = self.window_manager.consider_adaptation(
                &self.current_stats,
                drift_detected,
                anomaly_result.as_ref(),
            )? {
                window_adapted = true;
                adaptation_result = Some(adaptation);
            }
        }

        // Performance monitoring
        if self.performance_monitor.should_monitor() {
            self.performance_monitor
                .take_snapshot(&self.current_stats)?;
        }

        // Update ensemble metrics
        self.metric_ensemble.update(true_value, predicted_value)?;

        self.current_stats.last_update = Instant::now();

        let processing_time = start_time.elapsed();
        let current_performance = self.get_current_performance();

        Ok(UpdateResult {
            drift_detected,
            drift_results,
            anomaly_detected,
            anomaly_result,
            window_adapted,
            adaptation_result,
            processing_time,
            current_performance,
        })
    }

    /// Get current performance metrics
    pub fn get_current_performance(&self) -> HashMap<String, F> {
        let mut performance = HashMap::new();
        performance.insert("accuracy".to_string(), self.current_stats.current_accuracy);
        performance.insert("error_rate".to_string(), self.current_stats.error_rate);
        performance.insert(
            "moving_average_accuracy".to_string(),
            self.current_stats.moving_average_accuracy,
        );
        performance.insert(
            "total_samples".to_string(),
            F::from(self.current_stats.total_samples).unwrap(),
        );
        performance
    }

    /// Get anomaly summary
    pub fn get_anomaly_summary(&self) -> AnomalySummary<F> {
        AnomalySummary {
            total_anomalies: self.anomaly_detector.detected_anomalies.len(),
            recent_anomalies: self
                .anomaly_detector
                .detected_anomalies
                .iter()
                .rev()
                .take(10)
                .cloned()
                .collect(),
            anomaly_rate: if self.current_stats.total_samples > 0 {
                F::from(self.anomaly_detector.detected_anomalies.len()).unwrap()
                    / F::from(self.current_stats.total_samples).unwrap()
            } else {
                F::zero()
            },
            statistics: self.anomaly_detector.statistics.clone(),
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) -> Result<()> {
        self.current_stats.reset();
        self.history_buffer.clear();

        for detector in &mut self.drift_detectors {
            detector.reset();
        }

        self.anomaly_detector.reset();
        self.window_manager.reset();
        self.performance_monitor.reset();
        self.metric_ensemble.reset();

        Ok(())
    }

    /// Get configuration
    pub fn get_config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Get current window size
    pub fn get_current_window_size(&self) -> usize {
        self.window_manager.get_current_size()
    }

    /// Get performance monitor
    pub fn get_performance_monitor(&self) -> &PerformanceMonitor<F> {
        &self.performance_monitor
    }

    /// Get alerts manager
    pub fn get_alerts_manager(&self) -> &AlertsManager {
        &self.alerts_manager
    }

    /// Get mutable alerts manager
    pub fn get_alerts_manager_mut(&mut self) -> &mut AlertsManager {
        &mut self.alerts_manager
    }
}
