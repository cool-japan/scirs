//! # Production-Level Metrics Collection and Monitoring
//!
//! This module provides comprehensive metrics collection, health checks, and monitoring
//! capabilities for production deployments of ``SciRS2`` Core.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Metric types supported by the system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Counter metric (monotonically increasing)
    Counter,
    /// Gauge metric (can go up or down)
    Gauge,
    /// Histogram metric (distribution of values)
    Histogram,
    /// Timer metric (duration measurements)
    Timer,
    /// Summary metric (quantiles over sliding time window)
    Summary,
}

/// Metric value representation
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// Duration value
    Duration(Duration),
    /// Boolean value
    Boolean(bool),
    /// String value
    String(String),
}

impl fmt::Display for MetricValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricValue::Integer(v) => write!(f, "{v}"),
            MetricValue::Float(v) => write!(f, "{v}"),
            MetricValue::Duration(v) => write!(f, "{v:?}"),
            MetricValue::Boolean(v) => write!(f, "{v}"),
            MetricValue::String(v) => write!(f, "{v}"),
        }
    }
}

/// A metric data point with timestamp and labels
#[derive(Debug, Clone)]
pub struct MetricPoint {
    /// Metric name
    pub name: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Metric value
    pub value: MetricValue,
    /// Timestamp when metric was recorded
    pub timestamp: SystemTime,
    /// Labels/tags for the metric
    pub labels: HashMap<String, String>,
    /// Help text describing the metric
    pub help: Option<String>,
}

/// High-performance counter metric
pub struct Counter {
    value: AtomicU64,
    name: String,
    labels: HashMap<String, String>,
}

impl Counter {
    /// Create a new counter
    pub fn new(name: String) -> Self {
        Self {
            value: AtomicU64::new(0),
            name,
            labels: HashMap::new(),
        }
    }

    /// Create a counter with labels
    pub fn with_labels(name: String, labels: HashMap<String, String>) -> Self {
        Self {
            value: AtomicU64::new(0),
            name,
            labels,
        }
    }

    /// Increment the counter by 1
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment the counter by a specific amount
    pub fn add(&self, amount: u64) {
        self.value.fetch_add(amount, Ordering::Relaxed);
    }

    /// Get the current value
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Get metric point
    pub fn to_metric_point(&self) -> MetricPoint {
        MetricPoint {
            name: self.name.clone(),
            metric_type: MetricType::Counter,
            value: MetricValue::Integer(self.get() as i64),
            timestamp: SystemTime::now(),
            labels: self.labels.clone(),
            help: None,
        }
    }
}

/// High-performance gauge metric
pub struct Gauge {
    value: AtomicU64, // Store as bits of f64
    name: String,
    labels: HashMap<String, String>,
}

impl Gauge {
    /// Create a new gauge
    pub fn new(name: String) -> Self {
        Self {
            value: AtomicU64::new(0),
            name,
            labels: HashMap::new(),
        }
    }

    /// Create a gauge with labels
    pub fn with_labels(name: String, labels: HashMap<String, String>) -> Self {
        Self {
            value: AtomicU64::new(0),
            name,
            labels,
        }
    }

    /// Set the gauge value
    pub fn set(&self, value: f64) {
        self.value.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Increment the gauge
    pub fn inc(&self) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed));
        self.set(current + 1.0);
    }

    /// Decrement the gauge
    pub fn dec(&self) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed));
        self.set(current - 1.0);
    }

    /// Add to the gauge
    pub fn add(&self, amount: f64) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed));
        self.set(current + amount);
    }

    /// Subtract from the gauge
    pub fn sub(&self, amount: f64) {
        let current = f64::from_bits(self.value.load(Ordering::Relaxed));
        self.set(current - amount);
    }

    /// Get the current value
    pub fn get(&self) -> f64 {
        f64::from_bits(self.value.load(Ordering::Relaxed))
    }

    /// Get metric point
    pub fn to_metric_point(&self) -> MetricPoint {
        MetricPoint {
            name: self.name.clone(),
            metric_type: MetricType::Gauge,
            value: MetricValue::Float(self.get()),
            timestamp: SystemTime::now(),
            labels: self.labels.clone(),
            help: None,
        }
    }
}

/// Histogram metric for tracking distributions
pub struct Histogram {
    buckets: Vec<(f64, AtomicU64)>, // (upper_bound, count)
    sum: AtomicU64,                 // Store as bits of f64
    count: AtomicU64,
    name: String,
    labels: HashMap<String, String>,
}

impl Histogram {
    /// Create a new histogram with default buckets
    pub fn new(name: String) -> Self {
        let default_buckets = vec![
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            f64::INFINITY,
        ];
        Self::with_buckets(name, default_buckets)
    }

    /// Create a histogram with custom buckets
    pub fn with_buckets(name: String, buckets: Vec<f64>) -> Self {
        let bucket_pairs = buckets
            .into_iter()
            .map(|b| (b, AtomicU64::new(0)))
            .collect();

        Self {
            buckets: bucket_pairs,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
            name,
            labels: HashMap::new(),
        }
    }

    /// Observe a value
    pub fn observe(&self, value: f64) {
        // Update count and sum
        self.count.fetch_add(1, Ordering::Relaxed);
        let current_sum = f64::from_bits(self.sum.load(Ordering::Relaxed));
        self.sum
            .store((current_sum + value).to_bits(), Ordering::Relaxed);

        // Update buckets
        for (upper_bound, count) in &self.buckets {
            if value <= *upper_bound {
                count.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Get histogram statistics
    pub fn get_stats(&self) -> HistogramStats {
        let count = self.count.load(Ordering::Relaxed);
        let sum = f64::from_bits(self.sum.load(Ordering::Relaxed));
        let mean = if count > 0 { sum / count as f64 } else { 0.0 };

        let bucket_counts: Vec<(f64, u64)> = self
            .buckets
            .iter()
            .map(|(bound, count)| (*bound, count.load(Ordering::Relaxed)))
            .collect();

        HistogramStats {
            count,
            sum,
            mean,
            buckets: bucket_counts,
        }
    }
}

/// Histogram statistics
#[derive(Debug, Clone)]
pub struct HistogramStats {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub buckets: Vec<(f64, u64)>,
}

/// Timer metric for measuring durations
pub struct Timer {
    histogram: Histogram,
}

impl Timer {
    /// Create a new timer
    pub fn new(name: String) -> Self {
        // Use smaller buckets for timing measurements (in seconds)
        let timing_buckets = vec![
            0.001,
            0.005,
            0.01,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            f64::INFINITY,
        ];
        Self {
            histogram: Histogram::with_buckets(name, timing_buckets),
        }
    }

    /// Start timing an operation
    pub fn start(&self) -> TimerGuard {
        TimerGuard {
            timer: self,
            start_time: Instant::now(),
        }
    }

    /// Observe a duration
    pub fn observe(&self, duration: Duration) {
        self.histogram.observe(duration.as_secs_f64());
    }

    /// Get timing statistics
    pub fn get_stats(&self) -> HistogramStats {
        self.histogram.get_stats()
    }
}

/// Guard for automatic timing
pub struct TimerGuard<'a> {
    timer: &'a Timer,
    start_time: Instant,
}

impl Drop for TimerGuard<'_> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        self.timer.observe(duration);
    }
}

/// Metrics registry for managing all metrics
pub struct MetricsRegistry {
    metrics: RwLock<HashMap<String, Box<dyn MetricProvider + Send + Sync>>>,
}

/// Trait for types that can provide metric points
pub trait MetricProvider {
    /// Get all metric points from this provider
    fn get_metric_points(&self) -> Vec<MetricPoint>;
}

impl MetricProvider for Counter {
    fn get_metric_points(&self) -> Vec<MetricPoint> {
        vec![self.to_metric_point()]
    }
}

impl MetricProvider for Gauge {
    fn get_metric_points(&self) -> Vec<MetricPoint> {
        vec![self.to_metric_point()]
    }
}

impl MetricProvider for Histogram {
    fn get_metric_points(&self) -> Vec<MetricPoint> {
        let stats = self.get_stats();
        let mut points = Vec::new();

        // Count metric
        points.push(MetricPoint {
            name: format!("{}_count", self.name),
            metric_type: MetricType::Counter,
            value: MetricValue::Integer(stats.count as i64),
            timestamp: SystemTime::now(),
            labels: self.labels.clone(),
            help: Some(format!("Total count for {}", self.name)),
        });

        // Sum metric
        points.push(MetricPoint {
            name: format!("{}_sum", self.name),
            metric_type: MetricType::Counter,
            value: MetricValue::Float(stats.sum),
            timestamp: SystemTime::now(),
            labels: self.labels.clone(),
            help: Some(format!("Total sum for {}", self.name)),
        });

        // Bucket metrics
        for (bucket, count) in stats.buckets {
            let mut bucket_labels = self.labels.clone();
            bucket_labels.insert("le".to_string(), bucket.to_string());

            points.push(MetricPoint {
                name: format!("{}_bucket", self.name),
                metric_type: MetricType::Counter,
                value: MetricValue::Integer(count as i64),
                timestamp: SystemTime::now(),
                labels: bucket_labels,
                help: Some(format!("Histogram bucket for {}", self.name)),
            });
        }

        points
    }
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        Self {
            metrics: RwLock::new(HashMap::new()),
        }
    }

    /// Register a metric
    pub fn register<T>(&self, name: String, metric: T) -> CoreResult<()>
    where
        T: MetricProvider + Send + Sync + 'static,
    {
        let mut metrics = self.metrics.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire metrics lock"))
        })?;

        metrics.insert(name, Box::new(metric));
        Ok(())
    }

    /// Get all metric points
    pub fn get_all_metrics(&self) -> CoreResult<Vec<MetricPoint>> {
        let metrics = self.metrics.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire metrics lock"))
        })?;

        let mut all_points = Vec::new();
        for provider in metrics.values() {
            all_points.extend(provider.get_metric_points());
        }

        Ok(all_points)
    }

    /// Export metrics in Prometheus format
    pub fn export_prometheus(&self) -> CoreResult<String> {
        let metrics = self.get_all_metrics()?;
        let mut output = String::new();

        for metric in metrics {
            // Add help text if available
            if let Some(help) = &metric.help {
                output.push_str(&format!("# HELP {} {}\n", metric.name, help));
            }

            // Add type information
            let type_str = match metric.metric_type {
                MetricType::Counter => "counter",
                MetricType::Gauge => "gauge",
                MetricType::Histogram => "histogram",
                MetricType::Timer => "histogram",
                MetricType::Summary => "summary",
            };
            output.push_str(&format!("# TYPE {} {}\n", metric.name, type_str));

            // Format labels
            let labels_str = if metric.labels.is_empty() {
                String::new()
            } else {
                let label_pairs: Vec<String> = metric
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, v))
                    .collect();
                format!("{{{}}}", label_pairs.join(","))
            };

            // Add metric line
            let timestamp = metric
                .timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();

            output.push_str(&format!(
                "{}{} {} {}\n",
                metric.name, labels_str, metric.value, timestamp
            ));
        }

        Ok(output)
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check status
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System has warnings but is operational
    Warning,
    /// System is unhealthy
    Unhealthy,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Warning => write!(f, "warning"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Name of the health check
    pub name: String,
    /// Status of the health check
    pub status: HealthStatus,
    /// Message describing the status
    pub message: String,
    /// Timestamp of the check
    pub timestamp: SystemTime,
    /// Duration of the check
    pub duration: Duration,
}

/// Health monitoring system
pub struct HealthMonitor {
    checks: RwLock<HashMap<String, Box<dyn HealthChecker + Send + Sync>>>,
    results_cache: RwLock<HashMap<String, HealthCheck>>,
    #[allow(dead_code)]
    cache_duration: Duration,
}

/// Trait for health check implementations
pub trait HealthChecker {
    /// Perform the health check
    fn check(&self) -> CoreResult<HealthCheck>;

    /// Get the name of this health check
    fn name(&self) -> &str;
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new() -> Self {
        Self {
            checks: RwLock::new(HashMap::new()),
            results_cache: RwLock::new(HashMap::new()),
            cache_duration: Duration::from_secs(30), // 30 second cache
        }
    }

    /// Register a health check
    pub fn register_check<T>(&self, checker: T) -> CoreResult<()>
    where
        T: HealthChecker + Send + Sync + 'static,
    {
        let mut checks = self.checks.write().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire health checks lock"))
        })?;

        checks.insert(checker.name().to_string(), Box::new(checker));
        Ok(())
    }

    /// Run all health checks
    pub fn check_all(&self) -> CoreResult<Vec<HealthCheck>> {
        let checks = self.checks.read().map_err(|_| {
            CoreError::ComputationError(ErrorContext::new("Failed to acquire health checks lock"))
        })?;

        let mut results = Vec::new();
        for checker in checks.values() {
            match checker.check() {
                Ok(result) => results.push(result),
                Err(error) => {
                    results.push(HealthCheck {
                        name: checker.name().to_string(),
                        status: HealthStatus::Unhealthy,
                        message: format!("Health check failed: {}", error),
                        timestamp: SystemTime::now(),
                        duration: Duration::ZERO,
                    });
                }
            }
        }

        // Update cache
        if let Ok(mut cache) = self.results_cache.write() {
            cache.clear();
            for result in &results {
                cache.insert(result.name.clone(), result.clone());
            }
        }

        Ok(results)
    }

    /// Get overall health status
    pub fn overall_status(&self) -> CoreResult<HealthStatus> {
        let results = self.check_all()?;

        if results.iter().any(|r| r.status == HealthStatus::Unhealthy) {
            Ok(HealthStatus::Unhealthy)
        } else if results.iter().any(|r| r.status == HealthStatus::Warning) {
            Ok(HealthStatus::Warning)
        } else {
            Ok(HealthStatus::Healthy)
        }
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Built-in health checks
/// Memory usage health check
pub struct MemoryHealthCheck {
    warning_threshold: f64,
    critical_threshold: f64,
}

impl MemoryHealthCheck {
    /// Create a new memory health check
    pub fn new(warning_threshold: f64, critical_threshold: f64) -> Self {
        Self {
            warning_threshold,
            critical_threshold,
        }
    }
}

impl HealthChecker for MemoryHealthCheck {
    fn check(&self) -> CoreResult<HealthCheck> {
        let start_time = Instant::now();

        // Get memory usage from safety tracker
        #[cfg(feature = "memory_management")]
        let pressure = {
            let tracker = crate::memory::safety::global_safety_tracker();
            tracker.memory_pressure()
        };

        #[cfg(not(feature = "memory_management"))]
        let pressure = 0.0; // Fallback when memory management is not available

        let (status, message) = if pressure >= self.critical_threshold {
            (
                HealthStatus::Unhealthy,
                format!("Memory usage critical: {:.1}%", pressure * 100.0),
            )
        } else if pressure >= self.warning_threshold {
            (
                HealthStatus::Warning,
                format!("Memory usage high: {:.1}%", pressure * 100.0),
            )
        } else {
            (
                HealthStatus::Healthy,
                format!("Memory usage normal: {:.1}%", pressure * 100.0),
            )
        };

        Ok(HealthCheck {
            name: "memory".to_string(),
            status,
            message,
            timestamp: SystemTime::now(),
            duration: start_time.elapsed(),
        })
    }

    fn name(&self) -> &str {
        "memory"
    }
}

/// Global metrics registry instance
static GLOBAL_METRICS_REGISTRY: std::sync::LazyLock<MetricsRegistry> =
    std::sync::LazyLock::new(MetricsRegistry::new);

/// Global health monitor instance
static GLOBAL_HEALTH_MONITOR: std::sync::LazyLock<HealthMonitor> = std::sync::LazyLock::new(|| {
    let monitor = HealthMonitor::new();

    // Register built-in health checks
    let _ = monitor.register_check(MemoryHealthCheck::new(0.8, 0.95));

    monitor
});

/// Get the global metrics registry
pub fn global_metrics_registry() -> &'static MetricsRegistry {
    &GLOBAL_METRICS_REGISTRY
}

/// Get the global health monitor
pub fn global_health_monitor() -> &'static HealthMonitor {
    &GLOBAL_HEALTH_MONITOR
}

/// Convenience macros for metrics
/// Create and register a counter metric
#[macro_export]
macro_rules! counter {
    ($name:expr) => {{
        let counter = $crate::metrics::Counter::new($name.to_string());
        let _ = $crate::metrics::global_metrics_registry().register($name.to_string(), counter);
        counter
    }};
    ($name:expr, $labels:expr) => {{
        let counter = $crate::metrics::Counter::with_labels($name.to_string(), $labels);
        let _ = $crate::metrics::global_metrics_registry().register($name.to_string(), counter);
        counter
    }};
}

/// Create and register a gauge metric
#[macro_export]
macro_rules! gauge {
    ($name:expr) => {{
        let gauge = $crate::metrics::Gauge::new($name.to_string());
        let _ = $crate::metrics::global_metrics_registry().register($name.to_string(), gauge);
        gauge
    }};
    ($name:expr, $labels:expr) => {{
        let gauge = $crate::metrics::Gauge::with_labels($name.to_string(), $labels);
        let _ = $crate::metrics::global_metrics_registry().register($name.to_string(), gauge);
        gauge
    }};
}

/// Create and register a histogram metric
#[macro_export]
macro_rules! histogram {
    ($name:expr) => {{
        let histogram = $crate::metrics::Histogram::new($name.to_string());
        let _ = $crate::metrics::global_metrics_registry().register($name.to_string(), histogram);
        histogram
    }};
    ($name:expr, $buckets:expr) => {{
        let histogram = $crate::metrics::Histogram::with_buckets($name.to_string(), $buckets);
        let _ = $crate::metrics::global_metrics_registry().register($name.to_string(), histogram);
        histogram
    }};
}

/// Create and register a timer metric
#[macro_export]
macro_rules! timer {
    ($name:expr) => {{
        let timer = $crate::metrics::Timer::new($name.to_string());
        let _ = $crate::metrics::global_metrics_registry().register($name.to_string(), timer);
        timer
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let counter = Counter::new("test_counter".to_string());
        assert_eq!(counter.get(), 0);

        counter.inc();
        assert_eq!(counter.get(), 1);

        counter.add(5);
        assert_eq!(counter.get(), 6);
    }

    #[test]
    fn test_gauge() {
        let gauge = Gauge::new("test_gauge".to_string());
        assert_eq!(gauge.get(), 0.0);

        gauge.set(std::f64::consts::PI);
        assert!((gauge.get() - std::f64::consts::PI).abs() < f64::EPSILON);

        gauge.inc();
        assert!((gauge.get() - (std::f64::consts::PI + 1.0)).abs() < 1e-10);

        gauge.dec();
        assert!((gauge.get() - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_histogram() {
        let histogram = Histogram::new("test_histogram".to_string());

        histogram.observe(0.5);
        histogram.observe(1.5);
        histogram.observe(2.5);

        let stats = histogram.get_stats();
        assert_eq!(stats.count, 3);
        assert!((stats.sum - 4.5).abs() < f64::EPSILON);
        assert!((stats.mean - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_timer() {
        let timer = Timer::new("test_timer".to_string());

        {
            let _guard = timer.start();
            std::thread::sleep(Duration::from_millis(10));
        }

        let stats = timer.get_stats();
        assert_eq!(stats.count, 1);
        assert!(stats.sum > 0.0);
    }

    #[test]
    fn test_metrics_registry() {
        let registry = MetricsRegistry::new();
        let counter = Counter::new("test_counter".to_string());

        registry
            .register("test_counter".to_string(), counter)
            .unwrap();

        let metrics = registry.get_all_metrics().unwrap();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].name, "test_counter");
    }

    #[test]
    fn test_health_monitor() {
        let monitor = HealthMonitor::new();

        // Register memory health check
        let memory_check = MemoryHealthCheck::new(0.8, 0.95);
        monitor.register_check(memory_check).unwrap();

        let results = monitor.check_all().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "memory");
    }

    #[test]
    fn test_prometheus_export() {
        let registry = MetricsRegistry::new();
        let counter = Counter::new("test_counter".to_string());
        counter.inc();

        registry
            .register("test_counter".to_string(), counter)
            .unwrap();

        let prometheus_output = registry.export_prometheus().unwrap();
        assert!(prometheus_output.contains("test_counter"));
        assert!(prometheus_output.contains("counter"));
    }
}
