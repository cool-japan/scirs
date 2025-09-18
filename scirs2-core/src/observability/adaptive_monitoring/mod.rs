//! Adaptive performance monitoring and optimization
//!
//! This module provides intelligent performance monitoring with adaptive
//! optimization capabilities, real-time tuning, and predictive performance
//! management for production 1.0 deployments.

use crate::error::{CoreError, CoreResult, ErrorContext};
#[allow(unused_imports)]
use crate::performance::{OptimizationSettings, PerformanceProfile, WorkloadType};
#[allow(unused_imports)]
use crate::resource::auto_tuning::{ResourceManager, ResourceMetrics};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

// Module declarations
pub mod types;
pub mod metrics;
pub mod alerting;
pub mod prediction;
pub mod strategies;

// Re-exports
pub use types::*;
pub use metrics::MetricsCollector;
pub use alerting::{AlertingSystem, AlertingStats};
pub use prediction::{PredictionEngine, PredictionStats, PerformanceRisk, RiskType, RiskSeverity};
pub use strategies::{OptimizationEngine, OptimizationStats, OptimizationSummary};

/// Global adaptive monitoring system
static GLOBAL_MONITORING: std::sync::OnceLock<Arc<AdaptiveMonitoringSystem>> =
    std::sync::OnceLock::new();

/// Comprehensive adaptive monitoring and optimization system
#[allow(dead_code)]
#[derive(Debug)]
pub struct AdaptiveMonitoringSystem {
    performancemonitor: Arc<RwLock<PerformanceMonitor>>,
    optimization_engine: Arc<RwLock<OptimizationEngine>>,
    prediction_engine: Arc<RwLock<PredictionEngine>>,
    alerting_system: Arc<Mutex<AlertingSystem>>,
    configuration: Arc<RwLock<MonitoringConfiguration>>,
    metrics_collector: Arc<Mutex<MetricsCollector>>,
}

impl AdaptiveMonitoringSystem {
    /// Create new adaptive monitoring system
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            performancemonitor: Arc::new(RwLock::new(PerformanceMonitor::new()?)),
            optimization_engine: Arc::new(RwLock::new(OptimizationEngine::new()?)),
            prediction_engine: Arc::new(RwLock::new(PredictionEngine::new()?)),
            alerting_system: Arc::new(Mutex::new(AlertingSystem::new()?)),
            configuration: Arc::new(RwLock::new(MonitoringConfiguration::default())),
            metrics_collector: Arc::new(Mutex::new(MetricsCollector::new()?)),
        })
    }

    /// Get global monitoring system instance
    pub fn global() -> CoreResult<Arc<Self>> {
        Ok(GLOBAL_MONITORING
            .get_or_init(|| Arc::new(Self::new().unwrap()))
            .clone())
    }

    /// Start adaptive monitoring and optimization
    pub fn start(&self) -> CoreResult<()> {
        // Start performance monitoring thread
        let monitor = self.performancemonitor.clone();
        let config = self.configuration.clone();
        let metrics_collector = self.metrics_collector.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::monitoring_loop(&monitor, &config, &metrics_collector) {
                eprintln!("Monitoring error: {e:?}");
            }
            thread::sleep(Duration::from_secs(1));
        });

        // Start optimization engine thread
        let optimization = self.optimization_engine.clone();
        let monitor_clone = self.performancemonitor.clone();
        let prediction = self.prediction_engine.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::optimization_loop(&optimization, &monitor_clone, &prediction) {
                eprintln!("Optimization error: {e:?}");
            }
            thread::sleep(Duration::from_secs(10));
        });

        // Start prediction engine thread
        let prediction_clone = self.prediction_engine.clone();
        let monitor_clone2 = self.performancemonitor.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::prediction_loop(&prediction_clone, &monitor_clone2) {
                eprintln!("Prediction error: {e:?}");
            }
            thread::sleep(Duration::from_secs(30));
        });

        // Start alerting system thread
        let alerting = self.alerting_system.clone();
        let monitor_clone3 = self.performancemonitor.clone();

        thread::spawn(move || loop {
            if let Err(e) = Self::alerting_loop(&alerting, &monitor_clone3) {
                eprintln!("Alerting error: {e:?}");
            }
            thread::sleep(Duration::from_secs(5));
        });

        Ok(())
    }

    fn collect_metrics(
        collector: &Arc<Mutex<MetricsCollector>>,
        config: &Arc<RwLock<MonitoringConfiguration>>,
        monitor: &Arc<RwLock<PerformanceMonitor>>,
    ) -> CoreResult<()> {
        let config_read = config.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire config lock".to_string(),
            ))
        })?;

        if !config_read.monitoring_enabled {
            return Ok(());
        }

        // Collect current metrics
        let mut collector_lock = collector.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire collector lock".to_string(),
            ))
        })?;
        let metrics = collector_lock.collect_comprehensive_metrics()?;

        // Update performance monitor
        let mut monitor_write = monitor.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire monitor lock".to_string(),
            ))
        })?;
        monitor_write.record_metrics(metrics)?;

        Ok(())
    }

    fn optimization_loop(
        optimization: &Arc<RwLock<OptimizationEngine>>,
        monitor: &Arc<RwLock<PerformanceMonitor>>,
        prediction: &Arc<RwLock<PredictionEngine>>,
    ) -> CoreResult<()> {
        let current_metrics = {
            let monitor_read = monitor.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new(
                    "Failed to acquire monitor lock".to_string(),
                ))
            })?;
            monitor_read.get_current_performance()?
        };

        let predictions = {
            let prediction_read = prediction.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new(
                    "Failed to acquire prediction lock".to_string(),
                ))
            })?;
            prediction_read.get_current_predictions()?
        };

        let mut optimization_write = optimization.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire optimization lock".to_string(),
            ))
        })?;
        optimization_write.adaptive_optimize(&current_metrics, &predictions)?;

        Ok(())
    }

    fn prediction_loop(
        prediction: &Arc<RwLock<PredictionEngine>>,
        monitor: &Arc<RwLock<PerformanceMonitor>>,
    ) -> CoreResult<()> {
        let historical_data = {
            let monitor_read = monitor.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new(
                    "Failed to acquire monitor lock".to_string(),
                ))
            })?;
            monitor_read.get_historical_data()?
        };

        let mut prediction_write = prediction.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire prediction lock".to_string(),
            ))
        })?;
        prediction_write.update_with_data(&historical_data)?;

        Ok(())
    }

    fn alerting_loop(
        alerting: &Arc<Mutex<AlertingSystem>>,
        monitor: &Arc<RwLock<PerformanceMonitor>>,
    ) -> CoreResult<()> {
        let current_performance = {
            let monitor_read = monitor.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new(
                    "Failed to acquire monitor lock".to_string(),
                ))
            })?;
            monitor_read.get_current_performance()?
        };

        let mut alerting_write = alerting.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire alerting lock".to_string(),
            ))
        })?;
        alerting_write.check_and_trigger_alerts(&current_performance)?;

        Ok(())
    }

    /// Get current system performance metrics
    pub fn get_performance_metrics(&self) -> CoreResult<ComprehensivePerformanceMetrics> {
        let monitor = self.performancemonitor.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire monitor lock".to_string(),
            ))
        })?;
        monitor.get_current_performance()
    }

    /// Get optimization recommendations
    pub fn get_optimization_recommendations(&self) -> CoreResult<Vec<OptimizationRecommendation>> {
        let optimization = self.optimization_engine.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire optimization lock".to_string(),
            ))
        })?;
        optimization.get_recommendations()
    }

    /// Get performance predictions
    pub fn get_performance_predictions(&self) -> CoreResult<PerformancePredictions> {
        let prediction = self.prediction_engine.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire prediction lock".to_string(),
            ))
        })?;
        prediction.get_current_predictions()
    }

    /// Update monitoring configuration
    pub fn update_config(&self, new_config: MonitoringConfiguration) -> CoreResult<()> {
        let mut config = self.configuration.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire config lock".to_string(),
            ))
        })?;
        *config = new_config;
        Ok(())
    }

    /// Get monitoring dashboard data
    pub fn get_dashboard_data(&self) -> CoreResult<MonitoringDashboard> {
        let performance = self.get_performance_metrics()?;
        let recommendations = self.get_optimization_recommendations()?;
        let predictions = self.get_performance_predictions()?;

        let alerts = {
            let alerting = self.alerting_system.lock().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new(
                    "Failed to acquire alerting lock".to_string(),
                ))
            })?;
            alerting.get_active_alerts()?
        };

        Ok(MonitoringDashboard {
            performance,
            recommendations,
            predictions,
            alerts,
            timestamp: Instant::now(),
        })
    }

    /// Main monitoring loop for performance tracking
    fn monitoring_loop(
        monitor: &Arc<RwLock<PerformanceMonitor>>,
        config: &Arc<RwLock<MonitoringConfiguration>>,
        metrics_collector: &Arc<Mutex<MetricsCollector>>,
    ) -> CoreResult<()> {
        // Collect metrics and update performance monitor
        Self::collect_metrics(metrics_collector, config, monitor)?;

        // Update performance trends
        let mut monitor_write = monitor.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new(
                "Failed to acquire monitor lock".to_string(),
            ))
        })?;

        // Get latest metrics for trend analysis
        let current_metrics = ComprehensivePerformanceMetrics::default();
        monitor_write.update_performance_trends(&current_metrics)?;

        Ok(())
    }
}

/// Advanced performance monitoring with adaptive capabilities
#[allow(dead_code)]
#[derive(Debug)]
pub struct PerformanceMonitor {
    metrics_history: VecDeque<ComprehensivePerformanceMetrics>,
    performance_trends: HashMap<String, PerformanceTrend>,
    anomaly_detector: AnomalyDetector,
    baseline_performance: Option<PerformanceBaseline>,
    max_history_size: usize,
}

impl PerformanceMonitor {
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            metrics_history: VecDeque::with_capacity(10000),
            performance_trends: HashMap::new(),
            anomaly_detector: AnomalyDetector::new()?,
            baseline_performance: None,
            max_history_size: 10000,
        })
    }

    pub fn record_metrics(&mut self, metrics: ComprehensivePerformanceMetrics) -> CoreResult<()> {
        // Detect anomalies
        if let Some(anomalies) = self.anomaly_detector.detect_anomalies(&metrics)? {
            // Handle anomalies
            self.handle_anomalies(anomalies)?;
        }

        // Update trends
        self.update_performance_trends(&metrics)?;

        // Update baseline if needed
        if self.baseline_performance.is_none() || self.should_update_baseline(&metrics)? {
            self.baseline_performance = Some(PerformanceBaseline::from_metrics(&metrics));
        }

        // Add to history
        self.metrics_history.push_back(metrics);

        // Maintain history size
        while self.metrics_history.len() > self.max_history_size {
            self.metrics_history.pop_front();
        }

        Ok(())
    }

    pub fn get_current_performance(&self) -> CoreResult<ComprehensivePerformanceMetrics> {
        self.metrics_history.back().cloned().ok_or_else(|| {
            CoreError::InvalidState(ErrorContext::new(
                "No performance metrics available".to_string(),
            ))
        })
    }

    pub fn get_historical_data(&self) -> CoreResult<Vec<ComprehensivePerformanceMetrics>> {
        Ok(self.metrics_history.iter().cloned().collect())
    }

    pub fn update_performance_trends(
        &mut self,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<()> {
        // Update CPU trend
        let cpu_trend = self
            .performance_trends
            .entry("cpu".to_string())
            .or_default();
        cpu_trend.add_data_point(metrics.cpu_utilization, metrics.timestamp);

        // Update memory trend
        let memory_trend = self
            .performance_trends
            .entry("memory".to_string())
            .or_default();
        memory_trend.add_data_point(metrics.memory_utilization, metrics.timestamp);

        // Update throughput trend
        let throughput_trend = self
            .performance_trends
            .entry("throughput".to_string())
            .or_default();
        throughput_trend.add_data_point(metrics.operations_per_second, metrics.timestamp);

        // Update latency trend
        let latency_trend = self
            .performance_trends
            .entry("latency".to_string())
            .or_default();
        latency_trend.add_data_point(metrics.average_latency_ms, metrics.timestamp);

        Ok(())
    }

    fn handle_anomalies(&mut self, anomalies: Vec<PerformanceAnomaly>) -> CoreResult<()> {
        for anomaly in anomalies {
            match anomaly.severity {
                AnomalySeverity::Critical => {
                    // Trigger immediate response
                    eprintln!("CRITICAL ANOMALY DETECTED: {}", anomaly.description);
                }
                AnomalySeverity::Warning => {
                    // Log warning
                    println!("Performance warning: {}", anomaly.description);
                }
                AnomalySeverity::Info => {
                    // Log info
                    println!("Performance info: {}", anomaly.description);
                }
            }
        }
        Ok(())
    }

    fn should_update_baseline(
        &self,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<bool> {
        if let Some(baseline) = &self.baseline_performance {
            // Update baseline if performance has significantly improved
            let improvement_threshold = 0.2; // 20% improvement
            let cpu_improvement =
                (baseline.cpu_utilization - metrics.cpu_utilization) / baseline.cpu_utilization;
            let throughput_improvement = (metrics.operations_per_second
                - baseline.operations_per_second)
                / baseline.operations_per_second;

            Ok(cpu_improvement > improvement_threshold
                || throughput_improvement > improvement_threshold)
        } else {
            Ok(true)
        }
    }

    /// Get performance trends for specified metrics
    pub fn get_performance_trends(&self) -> &HashMap<String, PerformanceTrend> {
        &self.performance_trends
    }

    /// Get current baseline performance
    pub fn get_baseline_performance(&self) -> Option<&PerformanceBaseline> {
        self.baseline_performance.as_ref()
    }

    /// Get metrics history size
    pub fn get_history_size(&self) -> usize {
        self.metrics_history.len()
    }

    /// Clear metrics history
    pub fn clear_history(&mut self) -> CoreResult<()> {
        self.metrics_history.clear();
        self.performance_trends.clear();
        self.baseline_performance = None;
        Ok(())
    }
}

/// Initialize adaptive monitoring system
#[allow(dead_code)]
pub fn initialize_adaptive_monitoring() -> CoreResult<()> {
    let monitoring_system = AdaptiveMonitoringSystem::global()?;
    monitoring_system.start()?;
    Ok(())
}

/// Get current monitoring dashboard
#[allow(dead_code)]
pub fn get_monitoring_dashboard() -> CoreResult<MonitoringDashboard> {
    let monitoring_system = AdaptiveMonitoringSystem::global()?;
    monitoring_system.get_dashboard_data()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_system_creation() {
        let _system = AdaptiveMonitoringSystem::new().unwrap();
        // Basic functionality test
    }

    #[test]
    fn test_metrics_collection() {
        let mut collector = MetricsCollector::new().unwrap();
        let metrics = collector.collect_comprehensive_metrics().unwrap();

        assert!(metrics.cpu_utilization >= 0.0);
        assert!(metrics.memory_utilization >= 0.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let detector = AnomalyDetector::new().unwrap();
        let metrics = ComprehensivePerformanceMetrics {
            timestamp: Instant::now(),
            cpu_utilization: 0.99, // Anomalously high
            memory_utilization: 0.5,
            operations_per_second: 1000.0,
            average_latency_ms: 50.0,
            cache_miss_rate: 0.05,
            thread_count: 8,
            heap_size: 1024 * 1024 * 1024,
            gc_pressure: 0.1,
            network_utilization: 0.2,
            disk_io_rate: 100.0,
            custom_metrics: HashMap::new(),
        };

        let anomalies = detector.detect_anomalies(&metrics).unwrap();
        assert!(anomalies.is_some());
    }

    #[test]
    fn test_time_series_prediction() {
        let mut model = TimeSeriesModel::new();
        let data = vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        model.add_data(data).unwrap();

        let predictions = model.predict_next(3);
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_correlation_analysis() {
        let mut analyzer = CorrelationAnalyzer::new();

        // Create test data
        let mut test_data = Vec::new();
        for i in 0..20 {
            test_data.push(ComprehensivePerformanceMetrics {
                timestamp: Instant::now(),
                cpu_utilization: 0.5 + (i as f64) * 0.01,
                memory_utilization: 0.6,
                operations_per_second: 1000.0 - (i as f64) * 10.0, // Inverse correlation
                average_latency_ms: 50.0,
                cache_miss_rate: 0.05,
                thread_count: 8,
                heap_size: 1024 * 1024 * 1024,
                gc_pressure: 0.1,
                network_utilization: 0.2,
                disk_io_rate: 100.0,
                custom_metrics: HashMap::new(),
            });
        }

        analyzer.analyze_correlations(&test_data).unwrap();
        // Note: correlations field is private, so we can't test the specific content
    }
}