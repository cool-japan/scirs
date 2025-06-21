//! # Continuous Performance Monitoring
//!
//! This module provides continuous performance monitoring capabilities for long-running
//! processes with real-time alerts, trend analysis, and adaptive optimization.

use crate::error::{CoreError, CoreResult};
use crate::profiling::hardware_counters::{CounterType, CounterValue, HardwareCounterManager};
use crate::profiling::system_monitor::{SystemMetrics, SystemMonitor, SystemMonitorError};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

/// Error types for continuous monitoring
#[derive(Error, Debug)]
pub enum ContinuousMonitoringError {
    /// Monitor not running
    #[error("Continuous monitor is not running")]
    NotRunning,

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Alert configuration error
    #[error("Alert configuration error: {0}")]
    AlertConfigurationError(String),

    /// Data collection error
    #[error("Data collection error: {0}")]
    DataCollectionError(String),

    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),
}

impl From<ContinuousMonitoringError> for CoreError {
    fn from(err: ContinuousMonitoringError) -> Self {
        CoreError::ComputationError(crate::error::ErrorContext::new(err.to_string()))
    }
}

impl From<SystemMonitorError> for CoreError {
    fn from(err: SystemMonitorError) -> Self {
        CoreError::ComputationError(crate::error::ErrorContext::new(err.to_string()))
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Sampling interval for metrics collection
    pub sampling_interval: Duration,
    /// Maximum number of samples to keep in memory
    pub max_samples: usize,
    /// Enable system resource monitoring
    pub monitor_system: bool,
    /// Enable hardware counter monitoring
    pub monitor_hardware: bool,
    /// Enable application-specific monitoring
    pub monitor_application: bool,
    /// Data retention policy
    pub retention_policy: RetentionPolicy,
    /// Alert configuration
    pub alert_config: AlertConfiguration,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Trend analysis window
    pub trend_window: Duration,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            sampling_interval: Duration::from_secs(1),
            max_samples: 86400, // 24 hours at 1 second intervals
            monitor_system: true,
            monitor_hardware: false, // May require special permissions
            monitor_application: true,
            retention_policy: RetentionPolicy::default(),
            alert_config: AlertConfiguration::default(),
            enable_trend_analysis: true,
            trend_window: Duration::from_secs(300), // 5 minutes
            enable_adaptive_optimization: false,
        }
    }
}

/// Data retention policy
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Keep high-resolution data for this duration
    pub high_resolution_duration: Duration,
    /// Keep medium-resolution data for this duration
    pub medium_resolution_duration: Duration,
    /// Keep low-resolution data for this duration
    pub low_resolution_duration: Duration,
    /// High-resolution sampling interval
    pub high_res_interval: Duration,
    /// Medium-resolution sampling interval
    pub medium_res_interval: Duration,
    /// Low-resolution sampling interval
    pub low_res_interval: Duration,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            high_resolution_duration: Duration::from_secs(3600), // 1 hour
            medium_resolution_duration: Duration::from_secs(86400), // 24 hours
            low_resolution_duration: Duration::from_secs(604800), // 7 days
            high_res_interval: Duration::from_secs(1),
            medium_res_interval: Duration::from_secs(60),
            low_res_interval: Duration::from_secs(300),
        }
    }
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfiguration {
    /// CPU usage threshold for alerts (percentage)
    pub cpu_threshold: f64,
    /// Memory usage threshold for alerts (percentage)
    pub memory_threshold: f64,
    /// Performance degradation threshold (percentage)
    pub performance_degradation_threshold: f64,
    /// Error rate threshold (errors per minute)
    pub error_rate_threshold: f64,
    /// Response time threshold (milliseconds)
    pub response_time_threshold: f64,
    /// Enable email alerts
    pub enable_email_alerts: bool,
    /// Enable webhook alerts
    pub enable_webhook_alerts: bool,
    /// Alert cooldown period
    pub alert_cooldown: Duration,
}

impl Default for AlertConfiguration {
    fn default() -> Self {
        Self {
            cpu_threshold: 80.0,
            memory_threshold: 85.0,
            performance_degradation_threshold: 20.0,
            error_rate_threshold: 10.0,
            response_time_threshold: 1000.0,
            enable_email_alerts: false,
            enable_webhook_alerts: false,
            alert_cooldown: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Performance metrics snapshot
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    /// Timestamp of the snapshot
    pub timestamp: Instant,
    /// System metrics
    pub system_metrics: Option<SystemMetrics>,
    /// Hardware counter values
    pub hardware_counters: HashMap<CounterType, CounterValue>,
    /// Application-specific metrics
    pub application_metrics: ApplicationMetrics,
}

/// Application-specific performance metrics
#[derive(Debug, Clone, Default)]
pub struct ApplicationMetrics {
    /// Request count
    pub request_count: u64,
    /// Error count
    pub error_count: u64,
    /// Average response time (milliseconds)
    pub avg_response_time: f64,
    /// 95th percentile response time
    pub p95_response_time: f64,
    /// 99th percentile response time
    pub p99_response_time: f64,
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Active connections
    pub active_connections: u64,
    /// Custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Metric name
    pub metric_name: String,
    /// Trend direction
    pub trend: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Predicted value for next period
    pub prediction: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Analysis timestamp
    pub timestamp: Instant,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Metric is increasing
    Increasing,
    /// Metric is decreasing
    Decreasing,
    /// Metric is stable
    Stable,
    /// Trend is unclear
    Unknown,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert ID
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Metric that triggered the alert
    pub metric_name: String,
    /// Current value
    pub current_value: f64,
    /// Threshold value
    pub threshold_value: f64,
    /// Timestamp when alert was triggered
    pub timestamp: Instant,
    /// Whether the alert is still active
    pub active: bool,
}

/// Alert types
#[derive(Debug, Clone, PartialEq)]
pub enum AlertType {
    /// CPU usage exceeded threshold
    HighCpuUsage,
    /// Memory usage exceeded threshold
    HighMemoryUsage,
    /// Performance degradation detected
    PerformanceDegradation,
    /// High error rate
    HighErrorRate,
    /// High response time
    HighResponseTime,
    /// System anomaly detected
    SystemAnomaly,
    /// Hardware counter anomaly
    HardwareAnomaly,
    /// Custom metric threshold exceeded
    CustomThreshold(String),
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: String,
    /// Recommendation type
    pub recommendation_type: RecommendationType,
    /// Description
    pub description: String,
    /// Expected impact
    pub expected_impact: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Implementation complexity
    pub complexity: ComplexityLevel,
    /// Timestamp
    pub timestamp: Instant,
}

/// Recommendation types
#[derive(Debug, Clone)]
pub enum RecommendationType {
    /// Scale up resources
    ScaleUp,
    /// Scale down resources
    ScaleDown,
    /// Optimize algorithm
    AlgorithmOptimization,
    /// Adjust configuration
    ConfigurationTuning,
    /// Cache optimization
    CacheOptimization,
    /// Memory optimization
    MemoryOptimization,
    /// I/O optimization
    IoOptimization,
}

/// Implementation complexity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ComplexityLevel {
    /// Simple configuration change
    Low,
    /// Code modification required
    Medium,
    /// Significant refactoring required
    High,
    /// Architecture change required
    VeryHigh,
}

/// Continuous performance monitor
pub struct ContinuousPerformanceMonitor {
    /// Configuration
    config: MonitoringConfig,
    /// System monitor
    system_monitor: Option<Arc<Mutex<SystemMonitor>>>,
    /// Hardware counter manager
    hardware_manager: Option<Arc<Mutex<HardwareCounterManager>>>,
    /// Metrics history
    metrics_history: Arc<RwLock<VecDeque<MetricsSnapshot>>>,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, PerformanceAlert>>>,
    /// Trend analysis results
    trend_analysis: Arc<RwLock<HashMap<String, TrendAnalysis>>>,
    /// Optimization recommendations
    recommendations: Arc<RwLock<Vec<OptimizationRecommendation>>>,
    /// Monitor running state
    running: Arc<Mutex<bool>>,
    /// Background thread handle
    thread_handle: Option<thread::JoinHandle<()>>,
    /// Application metrics provider
    app_metrics_provider: Option<Box<dyn ApplicationMetricsProvider + Send + Sync>>,
}

/// Trait for providing application-specific metrics
pub trait ApplicationMetricsProvider: Send + Sync {
    /// Get current application metrics
    fn get_metrics(&self) -> ApplicationMetrics;

    /// Get custom metric value
    fn get_custom_metric(&self, name: &str) -> Option<f64>;
}

impl ContinuousPerformanceMonitor {
    /// Create a new continuous performance monitor
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            system_monitor: None,
            hardware_manager: None,
            metrics_history: Arc::new(RwLock::new(VecDeque::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            trend_analysis: Arc::new(RwLock::new(HashMap::new())),
            recommendations: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
            thread_handle: None,
            app_metrics_provider: None,
        }
    }

    /// Set application metrics provider
    pub fn set_application_metrics_provider<P>(&mut self, provider: P)
    where
        P: ApplicationMetricsProvider + Send + Sync + 'static,
    {
        self.app_metrics_provider = Some(Box::new(provider));
    }

    /// Start continuous monitoring
    pub fn start(&mut self) -> CoreResult<()> {
        {
            let mut running = self.running.lock().unwrap();
            if *running {
                return Ok(()); // Already running
            }
            *running = true;
        }

        // Initialize monitors based on configuration
        if self.config.monitor_system {
            let mut system_monitor = SystemMonitor::new(Default::default());
            system_monitor.start()?;
            self.system_monitor = Some(Arc::new(Mutex::new(system_monitor)));
        }

        if self.config.monitor_hardware {
            if let Ok(hardware_manager) = HardwareCounterManager::new() {
                self.hardware_manager = Some(Arc::new(Mutex::new(hardware_manager)));
            }
        }

        // Start background monitoring thread
        self.start_monitoring_thread();

        Ok(())
    }

    /// Stop continuous monitoring
    pub fn stop(&mut self) {
        {
            let mut running = self.running.lock().unwrap();
            *running = false;
        }

        // Stop system monitor
        if let Some(system_monitor) = &self.system_monitor {
            let mut monitor = system_monitor.lock().unwrap();
            monitor.stop();
        }

        // Wait for background thread to finish
        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
    }

    /// Start the background monitoring thread
    fn start_monitoring_thread(&mut self) {
        let config = self.config.clone();
        let system_monitor = self.system_monitor.clone();
        let hardware_manager = self.hardware_manager.clone();
        let metrics_history = Arc::clone(&self.metrics_history);
        let active_alerts = Arc::clone(&self.active_alerts);
        let trend_analysis = Arc::clone(&self.trend_analysis);
        let recommendations = Arc::clone(&self.recommendations);
        let running = Arc::clone(&self.running);

        self.thread_handle = Some(thread::spawn(move || {
            Self::monitoring_loop(
                config,
                system_monitor,
                hardware_manager,
                metrics_history,
                active_alerts,
                trend_analysis,
                recommendations,
                running,
            );
        }));
    }

    /// Main monitoring loop
    fn monitoring_loop(
        config: MonitoringConfig,
        system_monitor: Option<Arc<Mutex<SystemMonitor>>>,
        hardware_manager: Option<Arc<Mutex<HardwareCounterManager>>>,
        metrics_history: Arc<RwLock<VecDeque<MetricsSnapshot>>>,
        active_alerts: Arc<RwLock<HashMap<String, PerformanceAlert>>>,
        trend_analysis: Arc<RwLock<HashMap<String, TrendAnalysis>>>,
        recommendations: Arc<RwLock<Vec<OptimizationRecommendation>>>,
        running: Arc<Mutex<bool>>,
    ) {
        let mut last_trend_analysis = Instant::now();
        let mut alert_cooldown_map: HashMap<String, Instant> = HashMap::new();

        while *running.lock().unwrap() {
            let snapshot_start = Instant::now();

            // Collect metrics snapshot
            let snapshot = Self::collect_metrics_snapshot(
                &system_monitor,
                &hardware_manager,
                None, // Would pass app_metrics_provider here
            );

            // Store snapshot
            {
                let mut history = metrics_history.write().unwrap();
                history.push_back(snapshot.clone());

                // Limit history size
                while history.len() > config.max_samples {
                    history.pop_front();
                }
            }

            // Check for alerts
            Self::check_alerts(
                &snapshot,
                &config.alert_config,
                &active_alerts,
                &mut alert_cooldown_map,
            );

            // Perform trend analysis periodically
            if snapshot_start.duration_since(last_trend_analysis) >= config.trend_window {
                if config.enable_trend_analysis {
                    Self::perform_trend_analysis(&metrics_history, &trend_analysis, &config);
                }

                if config.enable_adaptive_optimization {
                    Self::generate_optimization_recommendations(
                        &metrics_history,
                        &trend_analysis,
                        &recommendations,
                    );
                }

                last_trend_analysis = snapshot_start;
            }

            // Sleep until next sampling interval
            let elapsed = snapshot_start.elapsed();
            if elapsed < config.sampling_interval {
                thread::sleep(config.sampling_interval - elapsed);
            }
        }
    }

    /// Collect a metrics snapshot
    fn collect_metrics_snapshot(
        system_monitor: &Option<Arc<Mutex<SystemMonitor>>>,
        hardware_manager: &Option<Arc<Mutex<HardwareCounterManager>>>,
        _app_provider: Option<&(dyn ApplicationMetricsProvider + Send + Sync)>,
    ) -> MetricsSnapshot {
        let timestamp = Instant::now();

        // Collect system metrics
        let system_metrics = if let Some(monitor) = system_monitor {
            monitor.lock().unwrap().get_current_metrics().ok()
        } else {
            None
        };

        // Collect hardware counter values
        let hardware_counters = if let Some(manager) = hardware_manager {
            manager
                .lock()
                .unwrap()
                .sample_counters()
                .unwrap_or_default()
        } else {
            HashMap::new()
        };

        // Collect application metrics
        let application_metrics = ApplicationMetrics::default(); // Would use app_provider here

        MetricsSnapshot {
            timestamp,
            system_metrics,
            hardware_counters,
            application_metrics,
        }
    }

    /// Check for performance alerts
    fn check_alerts(
        snapshot: &MetricsSnapshot,
        alert_config: &AlertConfiguration,
        active_alerts: &Arc<RwLock<HashMap<String, PerformanceAlert>>>,
        cooldown_map: &mut HashMap<String, Instant>,
    ) {
        let now = Instant::now();

        // Check system metrics alerts
        if let Some(sys_metrics) = &snapshot.system_metrics {
            // CPU usage alert
            if sys_metrics.cpu_usage > alert_config.cpu_threshold {
                let alert_key = "high_cpu_usage".to_string();
                if Self::should_trigger_alert(
                    &alert_key,
                    cooldown_map,
                    alert_config.alert_cooldown,
                    now,
                ) {
                    let alert = PerformanceAlert {
                        id: format!("cpu_{}_{}", now.elapsed().as_secs(), rand::random::<u32>()),
                        alert_type: AlertType::HighCpuUsage,
                        severity: if sys_metrics.cpu_usage > alert_config.cpu_threshold * 1.2 {
                            AlertSeverity::Critical
                        } else {
                            AlertSeverity::Warning
                        },
                        message: format!("High CPU usage: {:.1}%", sys_metrics.cpu_usage),
                        metric_name: "cpu_usage".to_string(),
                        current_value: sys_metrics.cpu_usage,
                        threshold_value: alert_config.cpu_threshold,
                        timestamp: now,
                        active: true,
                    };

                    active_alerts
                        .write()
                        .unwrap()
                        .insert(alert.id.clone(), alert);
                    cooldown_map.insert(alert_key, now);
                }
            }

            // Memory usage alert
            if sys_metrics.memory_total > 0 {
                let memory_usage_percent =
                    (sys_metrics.memory_usage as f64 / sys_metrics.memory_total as f64) * 100.0;
                if memory_usage_percent > alert_config.memory_threshold {
                    let alert_key = "high_memory_usage".to_string();
                    if Self::should_trigger_alert(
                        &alert_key,
                        cooldown_map,
                        alert_config.alert_cooldown,
                        now,
                    ) {
                        let alert = PerformanceAlert {
                            id: format!(
                                "mem_{}_{}",
                                now.elapsed().as_secs(),
                                rand::random::<u32>()
                            ),
                            alert_type: AlertType::HighMemoryUsage,
                            severity: if memory_usage_percent > alert_config.memory_threshold * 1.1
                            {
                                AlertSeverity::Critical
                            } else {
                                AlertSeverity::Warning
                            },
                            message: format!("High memory usage: {:.1}%", memory_usage_percent),
                            metric_name: "memory_usage".to_string(),
                            current_value: memory_usage_percent,
                            threshold_value: alert_config.memory_threshold,
                            timestamp: now,
                            active: true,
                        };

                        active_alerts
                            .write()
                            .unwrap()
                            .insert(alert.id.clone(), alert);
                        cooldown_map.insert(alert_key, now);
                    }
                }
            }
        }

        // Check application metrics alerts
        let app_metrics = &snapshot.application_metrics;
        if app_metrics.avg_response_time > alert_config.response_time_threshold {
            let alert_key = "high_response_time".to_string();
            if Self::should_trigger_alert(
                &alert_key,
                cooldown_map,
                alert_config.alert_cooldown,
                now,
            ) {
                let alert = PerformanceAlert {
                    id: format!("resp_{}_{}", now.elapsed().as_secs(), rand::random::<u32>()),
                    alert_type: AlertType::HighResponseTime,
                    severity: AlertSeverity::Warning,
                    message: format!("High response time: {:.1}ms", app_metrics.avg_response_time),
                    metric_name: "avg_response_time".to_string(),
                    current_value: app_metrics.avg_response_time,
                    threshold_value: alert_config.response_time_threshold,
                    timestamp: now,
                    active: true,
                };

                active_alerts
                    .write()
                    .unwrap()
                    .insert(alert.id.clone(), alert);
                cooldown_map.insert(alert_key, now);
            }
        }
    }

    /// Check if an alert should be triggered based on cooldown
    fn should_trigger_alert(
        alert_key: &str,
        cooldown_map: &HashMap<String, Instant>,
        cooldown_duration: Duration,
        now: Instant,
    ) -> bool {
        if let Some(last_alert) = cooldown_map.get(alert_key) {
            now.duration_since(*last_alert) >= cooldown_duration
        } else {
            true
        }
    }

    /// Perform trend analysis on metrics
    fn perform_trend_analysis(
        metrics_history: &Arc<RwLock<VecDeque<MetricsSnapshot>>>,
        trend_analysis: &Arc<RwLock<HashMap<String, TrendAnalysis>>>,
        config: &MonitoringConfig,
    ) {
        let history = metrics_history.read().unwrap();
        let window_start = Instant::now() - config.trend_window;

        // Filter samples within the trend window
        let recent_samples: Vec<_> = history
            .iter()
            .filter(|snapshot| snapshot.timestamp >= window_start)
            .collect();

        if recent_samples.len() < 2 {
            return; // Not enough data for trend analysis
        }

        let mut analysis_results = HashMap::new();

        // Analyze CPU usage trend
        if let Some(trend) = Self::analyze_metric_trend(&recent_samples, |snapshot| {
            snapshot.system_metrics.as_ref().map(|m| m.cpu_usage)
        }) {
            analysis_results.insert("cpu_usage".to_string(), trend);
        }

        // Analyze memory usage trend
        if let Some(trend) = Self::analyze_metric_trend(&recent_samples, |snapshot| {
            snapshot.system_metrics.as_ref().and_then(|m| {
                if m.memory_total > 0 {
                    Some((m.memory_usage as f64 / m.memory_total as f64) * 100.0)
                } else {
                    None
                }
            })
        }) {
            analysis_results.insert("memory_usage".to_string(), trend);
        }

        // Analyze response time trend
        if let Some(trend) = Self::analyze_metric_trend(&recent_samples, |snapshot| {
            Some(snapshot.application_metrics.avg_response_time)
        }) {
            analysis_results.insert("avg_response_time".to_string(), trend);
        }

        // Update trend analysis results
        *trend_analysis.write().unwrap() = analysis_results;
    }

    /// Analyze trend for a specific metric
    fn analyze_metric_trend<F>(
        samples: &[&MetricsSnapshot],
        metric_extractor: F,
    ) -> Option<TrendAnalysis>
    where
        F: Fn(&MetricsSnapshot) -> Option<f64>,
    {
        let values: Vec<(f64, f64)> = samples
            .iter()
            .enumerate()
            .filter_map(|(i, snapshot)| metric_extractor(snapshot).map(|value| (i as f64, value)))
            .collect();

        if values.len() < 2 {
            return None;
        }

        // Calculate linear regression for trend
        let n = values.len() as f64;
        let sum_x: f64 = values.iter().map(|(x, _)| x).sum();
        let sum_y: f64 = values.iter().map(|(_, y)| y).sum();
        let sum_xy: f64 = values.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = values.iter().map(|(x, _)| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Determine trend direction and strength
        let (trend, strength) = if slope.abs() < 0.01 {
            (TrendDirection::Stable, 0.0)
        } else if slope > 0.0 {
            (TrendDirection::Increasing, slope.min(1.0))
        } else {
            (TrendDirection::Decreasing, (-slope).min(1.0))
        };

        // Predict next value
        let next_x = values.len() as f64;
        let prediction = slope * next_x + intercept;

        // Calculate confidence interval (simplified)
        let last_value = values.last().unwrap().1;
        let confidence_range = (last_value - prediction).abs() * 0.2;
        let confidence_interval = (prediction - confidence_range, prediction + confidence_range);

        Some(TrendAnalysis {
            metric_name: "unknown".to_string(), // Would be set by caller
            trend,
            strength,
            prediction,
            confidence_interval,
            timestamp: Instant::now(),
        })
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(
        metrics_history: &Arc<RwLock<VecDeque<MetricsSnapshot>>>,
        trend_analysis: &Arc<RwLock<HashMap<String, TrendAnalysis>>>,
        recommendations: &Arc<RwLock<Vec<OptimizationRecommendation>>>,
    ) {
        let _history = metrics_history.read().unwrap();
        let trends = trend_analysis.read().unwrap();
        let mut new_recommendations = Vec::new();

        // Analyze CPU usage trend
        if let Some(cpu_trend) = trends.get("cpu_usage") {
            if cpu_trend.trend == TrendDirection::Increasing && cpu_trend.strength > 0.5 {
                new_recommendations.push(OptimizationRecommendation {
                    id: format!("cpu_opt_{}", Instant::now().elapsed().as_secs()),
                    recommendation_type: RecommendationType::ScaleUp,
                    description: "CPU usage is trending upward. Consider scaling up resources or optimizing CPU-intensive operations.".to_string(),
                    expected_impact: cpu_trend.strength * 20.0, // Estimated percentage improvement
                    confidence: cpu_trend.strength,
                    complexity: ComplexityLevel::Medium,
                    timestamp: Instant::now(),
                });
            }
        }

        // Analyze memory usage trend
        if let Some(memory_trend) = trends.get("memory_usage") {
            if memory_trend.trend == TrendDirection::Increasing && memory_trend.strength > 0.3 {
                new_recommendations.push(OptimizationRecommendation {
                    id: format!("mem_opt_{}", Instant::now().elapsed().as_secs()),
                    recommendation_type: RecommendationType::MemoryOptimization,
                    description: "Memory usage is increasing. Consider implementing memory pooling or optimizing data structures.".to_string(),
                    expected_impact: memory_trend.strength * 15.0,
                    confidence: memory_trend.strength,
                    complexity: ComplexityLevel::High,
                    timestamp: Instant::now(),
                });
            }
        }

        // Analyze response time patterns
        if let Some(response_trend) = trends.get("avg_response_time") {
            if response_trend.trend == TrendDirection::Increasing && response_trend.strength > 0.4 {
                new_recommendations.push(OptimizationRecommendation {
                    id: format!("resp_opt_{}", Instant::now().elapsed().as_secs()),
                    recommendation_type: RecommendationType::CacheOptimization,
                    description: "Response times are increasing. Consider implementing caching or optimizing database queries.".to_string(),
                    expected_impact: response_trend.strength * 25.0,
                    confidence: response_trend.strength,
                    complexity: ComplexityLevel::Medium,
                    timestamp: Instant::now(),
                });
            }
        }

        // Update recommendations (keep only recent ones)
        let mut recs = recommendations.write().unwrap();
        recs.extend(new_recommendations);

        // Remove old recommendations (keep last 10)
        if recs.len() > 10 {
            let len = recs.len();
            recs.drain(0..len - 10);
        }
    }

    /// Get current active alerts
    pub fn get_active_alerts(&self) -> Vec<PerformanceAlert> {
        self.active_alerts
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect()
    }

    /// Get current trend analysis
    pub fn get_trend_analysis(&self) -> HashMap<String, TrendAnalysis> {
        self.trend_analysis.read().unwrap().clone()
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<OptimizationRecommendation> {
        self.recommendations.read().unwrap().clone()
    }

    /// Get recent metrics history
    pub fn get_metrics_history(&self, duration: Duration) -> Vec<MetricsSnapshot> {
        let history = self.metrics_history.read().unwrap();
        let cutoff = Instant::now() - duration;

        history
            .iter()
            .filter(|snapshot| snapshot.timestamp >= cutoff)
            .cloned()
            .collect()
    }

    /// Generate monitoring report
    pub fn generate_report(&self) -> MonitoringReport {
        let active_alerts = self.get_active_alerts();
        let trends = self.get_trend_analysis();
        let recommendations = self.get_recommendations();
        let recent_metrics = self.get_metrics_history(Duration::from_secs(3600)); // Last hour

        MonitoringReport {
            timestamp: Instant::now(),
            monitoring_duration: self.config.sampling_interval,
            total_samples: recent_metrics.len(),
            active_alerts: active_alerts.len(),
            critical_alerts: active_alerts
                .iter()
                .filter(|a| a.severity == AlertSeverity::Critical)
                .count(),
            trends_detected: trends.len(),
            recommendations_generated: recommendations.len(),
            summary: self.generate_summary(&recent_metrics, &active_alerts, &trends),
        }
    }

    /// Generate monitoring summary
    fn generate_summary(
        &self,
        metrics: &[MetricsSnapshot],
        alerts: &[PerformanceAlert],
        trends: &HashMap<String, TrendAnalysis>,
    ) -> String {
        let mut summary = String::new();

        summary.push_str(&format!("Monitoring Summary ({} samples)\n", metrics.len()));

        if !alerts.is_empty() {
            summary.push_str(&format!("- {} active alerts\n", alerts.len()));
            let critical_count = alerts
                .iter()
                .filter(|a| a.severity == AlertSeverity::Critical)
                .count();
            if critical_count > 0 {
                summary.push_str(&format!(
                    "- {} critical alerts require immediate attention\n",
                    critical_count
                ));
            }
        } else {
            summary.push_str("- No active alerts\n");
        }

        if !trends.is_empty() {
            summary.push_str(&format!("- {} trends detected:\n", trends.len()));
            for (metric, trend) in trends {
                summary.push_str(&format!(
                    "  - {}: {:?} (strength: {:.1})\n",
                    metric, trend.trend, trend.strength
                ));
            }
        }

        if let Some(latest) = metrics.last() {
            if let Some(sys_metrics) = &latest.system_metrics {
                summary.push_str(&format!(
                    "- Current CPU usage: {:.1}%\n",
                    sys_metrics.cpu_usage
                ));
                if sys_metrics.memory_total > 0 {
                    let memory_percent =
                        (sys_metrics.memory_usage as f64 / sys_metrics.memory_total as f64) * 100.0;
                    summary.push_str(&format!("- Current memory usage: {:.1}%\n", memory_percent));
                }
            }
        }

        summary
    }
}

impl Drop for ContinuousPerformanceMonitor {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Monitoring report
#[derive(Debug, Clone)]
pub struct MonitoringReport {
    /// Report timestamp
    pub timestamp: Instant,
    /// Monitoring duration
    pub monitoring_duration: Duration,
    /// Total number of samples collected
    pub total_samples: usize,
    /// Number of active alerts
    pub active_alerts: usize,
    /// Number of critical alerts
    pub critical_alerts: usize,
    /// Number of trends detected
    pub trends_detected: usize,
    /// Number of recommendations generated
    pub recommendations_generated: usize,
    /// Summary text
    pub summary: String,
}

/// Simple application metrics provider for testing
pub struct SimpleApplicationMetricsProvider {
    metrics: Arc<Mutex<ApplicationMetrics>>,
}

impl SimpleApplicationMetricsProvider {
    /// Create a new simple provider
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(ApplicationMetrics::default())),
        }
    }

    /// Update metrics
    pub fn update_metrics<F>(&self, updater: F)
    where
        F: FnOnce(&mut ApplicationMetrics),
    {
        let mut metrics = self.metrics.lock().unwrap();
        updater(&mut metrics);
    }
}

impl Default for SimpleApplicationMetricsProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl ApplicationMetricsProvider for SimpleApplicationMetricsProvider {
    fn get_metrics(&self) -> ApplicationMetrics {
        self.metrics.lock().unwrap().clone()
    }

    fn get_custom_metric(&self, name: &str) -> Option<f64> {
        self.metrics
            .lock()
            .unwrap()
            .custom_metrics
            .get(name)
            .copied()
    }
}

/// Global continuous performance monitor instance
static GLOBAL_MONITOR: std::sync::OnceLock<Arc<Mutex<ContinuousPerformanceMonitor>>> =
    std::sync::OnceLock::new();

/// Get the global continuous performance monitor
pub fn global_monitor() -> Arc<Mutex<ContinuousPerformanceMonitor>> {
    GLOBAL_MONITOR
        .get_or_init(|| {
            Arc::new(Mutex::new(ContinuousPerformanceMonitor::new(
                MonitoringConfig::default(),
            )))
        })
        .clone()
}

/// Convenience functions for continuous monitoring
pub mod utils {
    use super::*;

    /// Start basic continuous monitoring
    pub fn start_basic_monitoring() -> CoreResult<()> {
        let monitor = global_monitor();
        let mut monitor = monitor.lock().unwrap();
        monitor.start()
    }

    /// Stop continuous monitoring
    pub fn stop_monitoring() {
        let monitor = global_monitor();
        let mut monitor = monitor.lock().unwrap();
        monitor.stop();
    }

    /// Get current performance status
    pub fn get_performance_status() -> (usize, usize, usize) {
        let monitor = global_monitor();
        let monitor = monitor.lock().unwrap();

        let alerts = monitor.get_active_alerts();
        let trends = monitor.get_trend_analysis();
        let recommendations = monitor.get_recommendations();

        (alerts.len(), trends.len(), recommendations.len())
    }

    /// Check if there are any critical alerts
    pub fn has_critical_alerts() -> bool {
        let monitor = global_monitor();
        let monitor = monitor.lock().unwrap();

        monitor
            .get_active_alerts()
            .iter()
            .any(|alert| alert.severity == AlertSeverity::Critical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitoring_config() {
        let config = MonitoringConfig::default();
        assert_eq!(config.sampling_interval, Duration::from_secs(1));
        assert_eq!(config.max_samples, 86400);
        assert!(config.monitor_system);
        assert!(!config.monitor_hardware);
    }

    #[test]
    fn test_alert_configuration() {
        let config = AlertConfiguration::default();
        assert_eq!(config.cpu_threshold, 80.0);
        assert_eq!(config.memory_threshold, 85.0);
        assert_eq!(config.alert_cooldown, Duration::from_secs(300));
    }

    #[test]
    fn test_metrics_snapshot() {
        let snapshot = MetricsSnapshot {
            timestamp: Instant::now(),
            system_metrics: None,
            hardware_counters: HashMap::new(),
            application_metrics: ApplicationMetrics::default(),
        };

        assert!(snapshot.hardware_counters.is_empty());
        assert_eq!(snapshot.application_metrics.request_count, 0);
    }

    #[test]
    fn test_performance_alert() {
        let alert = PerformanceAlert {
            id: "test_alert".to_string(),
            alert_type: AlertType::HighCpuUsage,
            severity: AlertSeverity::Warning,
            message: "Test alert".to_string(),
            metric_name: "cpu_usage".to_string(),
            current_value: 85.0,
            threshold_value: 80.0,
            timestamp: Instant::now(),
            active: true,
        };

        assert_eq!(alert.alert_type, AlertType::HighCpuUsage);
        assert_eq!(alert.severity, AlertSeverity::Warning);
        assert!(alert.active);
    }

    #[test]
    fn test_trend_analysis() {
        let trend = TrendAnalysis {
            metric_name: "cpu_usage".to_string(),
            trend: TrendDirection::Increasing,
            strength: 0.7,
            prediction: 85.0,
            confidence_interval: (80.0, 90.0),
            timestamp: Instant::now(),
        };

        assert_eq!(trend.trend, TrendDirection::Increasing);
        assert_eq!(trend.strength, 0.7);
        assert_eq!(trend.prediction, 85.0);
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            id: "opt_1".to_string(),
            recommendation_type: RecommendationType::ScaleUp,
            description: "Scale up resources".to_string(),
            expected_impact: 20.0,
            confidence: 0.8,
            complexity: ComplexityLevel::Medium,
            timestamp: Instant::now(),
        };

        assert_eq!(recommendation.complexity, ComplexityLevel::Medium);
        assert_eq!(recommendation.expected_impact, 20.0);
    }

    #[test]
    fn test_application_metrics_provider() {
        let provider = SimpleApplicationMetricsProvider::new();

        provider.update_metrics(|metrics| {
            metrics.request_count = 100;
            metrics.avg_response_time = 250.0;
            metrics
                .custom_metrics
                .insert("custom_metric".to_string(), 42.0);
        });

        let metrics = provider.get_metrics();
        assert_eq!(metrics.request_count, 100);
        assert_eq!(metrics.avg_response_time, 250.0);

        let custom_value = provider.get_custom_metric("custom_metric");
        assert_eq!(custom_value, Some(42.0));
    }

    #[test]
    fn test_continuous_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = ContinuousPerformanceMonitor::new(config);

        assert!(!*monitor.running.lock().unwrap());
        assert!(monitor.get_active_alerts().is_empty());
        assert!(monitor.get_recommendations().is_empty());
    }

    #[test]
    fn test_monitoring_report() {
        let report = MonitoringReport {
            timestamp: Instant::now(),
            monitoring_duration: Duration::from_secs(1),
            total_samples: 100,
            active_alerts: 2,
            critical_alerts: 1,
            trends_detected: 3,
            recommendations_generated: 1,
            summary: "Test summary".to_string(),
        };

        assert_eq!(report.total_samples, 100);
        assert_eq!(report.active_alerts, 2);
        assert_eq!(report.critical_alerts, 1);
    }

    #[test]
    fn test_global_monitor() {
        let monitor = global_monitor();

        // Should return the same instance
        let monitor2 = global_monitor();
        assert!(Arc::ptr_eq(&monitor, &monitor2));
    }

    #[test]
    fn test_utils_functions() {
        // Test performance status
        let (alerts, trends, recommendations) = utils::get_performance_status();
        assert_eq!(alerts, 0);
        assert_eq!(trends, 0);
        assert_eq!(recommendations, 0);

        // Test critical alerts check
        let has_critical = utils::has_critical_alerts();
        assert!(!has_critical);
    }
}
