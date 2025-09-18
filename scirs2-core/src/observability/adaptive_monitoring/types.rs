//! Shared types and enums for adaptive monitoring system

use crate::error::{CoreError, CoreResult};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ComprehensivePerformanceMetrics {
    pub timestamp: Instant,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub operations_per_second: f64,
    pub average_latency_ms: f64,
    pub cache_miss_rate: f64,
    pub thread_count: usize,
    pub heap_size: usize,
    pub gc_pressure: f64,
    pub network_utilization: f64,
    pub disk_io_rate: f64,
    pub custom_metrics: HashMap<String, f64>,
}

impl Default for ComprehensivePerformanceMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            operations_per_second: 0.0,
            average_latency_ms: 0.0,
            cache_miss_rate: 0.0,
            thread_count: 1,
            heap_size: 0,
            gc_pressure: 0.0,
            network_utilization: 0.0,
            disk_io_rate: 0.0,
            custom_metrics: HashMap::new(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MonitoringConfiguration {
    pub monitoring_enabled: bool,
    pub collection_interval: Duration,
    pub optimization_enabled: bool,
    pub prediction_enabled: bool,
    pub alerting_enabled: bool,
    pub adaptive_tuning_enabled: bool,
    pub max_history_size: usize,
}

impl Default for MonitoringConfiguration {
    fn default() -> Self {
        Self {
            monitoring_enabled: true,
            collection_interval: Duration::from_secs(1),
            optimization_enabled: true,
            prediction_enabled: true,
            alerting_enabled: true,
            adaptive_tuning_enabled: true,
            max_history_size: 10000,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct MonitoringDashboard {
    pub performance: ComprehensivePerformanceMetrics,
    pub recommendations: Vec<OptimizationRecommendation>,
    pub predictions: PerformancePredictions,
    pub alerts: Vec<PerformanceAlert>,
    pub timestamp: Instant,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    data_points: VecDeque<(f64, Instant)>,
    slope: f64,
    direction: TrendDirection,
}

impl Default for PerformanceTrend {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceTrend {
    pub fn new() -> Self {
        Self {
            data_points: VecDeque::with_capacity(100),
            slope: 0.0,
            direction: TrendDirection::Stable,
        }
    }

    pub fn add_data_point(&mut self, value: f64, timestamp: Instant) {
        self.data_points.push_back((value, timestamp));

        // Keep only recent data points
        while self.data_points.len() > 100 {
            self.data_points.pop_front();
        }

        // Update trend analysis
        self.update_trend_analysis();
    }

    fn update_trend_analysis(&mut self) {
        if self.data_points.len() < 2 {
            return;
        }

        // Simple linear regression for slope calculation
        let n = self.data_points.len() as f64;
        let sum_x: f64 = (0..self.data_points.len()).map(|i| i as f64).sum();
        let sum_y: f64 = self.data_points.iter().map(|value| value.0).sum();
        let sum_xy: f64 = self
            .data_points
            .iter()
            .enumerate()
            .map(|(i, value)| i as f64 * value.0)
            .sum();
        let sum_x_squared: f64 = (0..self.data_points.len())
            .map(|i| (i as f64).powi(2))
            .sum();

        self.slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x.powi(2));

        self.direction = if self.slope > 0.01 {
            TrendDirection::Increasing
        } else if self.slope < -0.01 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct AnomalyDetector {
    #[allow(dead_code)]
    detection_window: Duration,
    #[allow(dead_code)]
    sensitivity: f64,
}

impl AnomalyDetector {
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            detection_window: Duration::from_secs(300), // 5 minutes
            sensitivity: 2.0,                           // 2 standard deviations
        })
    }

    pub fn detect_anomalies(
        &self,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<Option<Vec<PerformanceAnomaly>>> {
        let mut anomalies = Vec::new();

        // CPU anomaly detection
        if metrics.cpu_utilization > 0.95 {
            anomalies.push(PerformanceAnomaly {
                metricname: "cpu_utilization".to_string(),
                current_value: metrics.cpu_utilization,
                expected_range: (0.0, 0.8),
                severity: AnomalySeverity::Critical,
                description: "Extremely high CPU utilization detected".to_string(),
                detected_at: metrics.timestamp,
            });
        }

        // Memory anomaly detection
        if metrics.memory_utilization > 0.95 {
            anomalies.push(PerformanceAnomaly {
                metricname: "memory_utilization".to_string(),
                current_value: metrics.memory_utilization,
                expected_range: (0.0, 0.8),
                severity: AnomalySeverity::Critical,
                description: "Extremely high memory utilization detected".to_string(),
                detected_at: metrics.timestamp,
            });
        }

        // Latency anomaly detection
        if metrics.average_latency_ms > 5000.0 {
            anomalies.push(PerformanceAnomaly {
                metricname: "average_latency_ms".to_string(),
                current_value: metrics.average_latency_ms,
                expected_range: (0.0, 1000.0),
                severity: AnomalySeverity::Warning,
                description: "High latency detected".to_string(),
                detected_at: metrics.timestamp,
            });
        }

        if anomalies.is_empty() {
            Ok(None)
        } else {
            Ok(Some(anomalies))
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    pub metricname: String,
    pub current_value: f64,
    pub expected_range: (f64, f64),
    pub severity: AnomalySeverity,
    pub description: String,
    pub detected_at: Instant,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalySeverity {
    Info,
    Warning,
    Critical,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub operations_per_second: f64,
    pub average_latency_ms: f64,
    pub established_at: Instant,
}

impl PerformanceBaseline {
    pub fn from_metrics(metrics: &ComprehensivePerformanceMetrics) -> Self {
        Self {
            cpu_utilization: metrics.cpu_utilization,
            memory_utilization: metrics.memory_utilization,
            operations_per_second: metrics.operations_per_second,
            average_latency_ms: metrics.average_latency_ms,
            established_at: metrics.timestamp,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct PerformanceLearningModel {
    learned_patterns: Vec<PerformancePattern>,
    #[allow(dead_code)]
    model_accuracy: f64,
}

impl PerformanceLearningModel {
    pub fn new() -> CoreResult<Self> {
        Ok(Self {
            learned_patterns: Vec::new(),
            model_accuracy: 0.5,
        })
    }

    pub fn update_with_metrics(
        &mut self,
        metrics: &ComprehensivePerformanceMetrics,
    ) -> CoreResult<()> {
        // Simple learning logic - in a real implementation this would be more sophisticated
        let pattern = PerformancePattern {
            cpu_range: (metrics.cpu_utilization - 0.1, metrics.cpu_utilization + 0.1),
            memory_range: (
                metrics.memory_utilization - 0.1,
                metrics.memory_utilization + 0.1,
            ),
            expected_throughput: metrics.operations_per_second,
            confidence: 0.7,
        };

        self.learned_patterns.push(pattern);

        // Keep only recent patterns
        if self.learned_patterns.len() > 1000 {
            self.learned_patterns.drain(0..100);
        }

        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformancePattern {
    pub cpu_range: (f64, f64),
    pub memory_range: (f64, f64),
    pub expected_throughput: f64,
    pub confidence: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationStrategy {
    Conservative,
    Balanced,
    Aggressive,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptimizationAction {
    pub actions: Vec<OptimizationActionType>,
    pub timestamp: Instant,
    pub reason: String,
    pub priority: OptimizationPriority,
    pub expected_impact: ImpactLevel,
    pub success: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizationActionType {
    ReduceThreads,
    IncreaseParallelism,
    ReduceMemoryUsage,
    OptimizeCacheUsage,
    PreemptiveCpuOptimization,
    PreemptiveMemoryOptimization,
    ReduceCpuUsage,
    OptimizePerformance,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationPriority {
    Low,
    Medium,
    High,
    Critical,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub category: RecommendationCategory,
    pub title: String,
    pub description: String,
    pub priority: RecommendationPriority,
    pub estimated_impact: ImpactLevel,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationCategory {
    Optimization,
    Strategy,
    Resource,
    Performance,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformancePredictions {
    pub predicted_cpu_spike: bool,
    pub predicted_memory_pressure: bool,
    pub predicted_throughput_drop: bool,
    pub cpu_forecast: Vec<f64>,
    pub memory_forecast: Vec<f64>,
    pub throughput_forecast: Vec<f64>,
    pub confidence: f64,
    pub time_horizon_minutes: u32,
    pub generated_at: Instant,
    pub predicted_performance_change: f64,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct TimeSeriesModel {
    data: VecDeque<f64>,
    trend: f64,
    #[allow(dead_code)]
    seasonal_component: Vec<f64>,
}

impl Default for TimeSeriesModel {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeSeriesModel {
    pub fn new() -> Self {
        Self {
            data: VecDeque::with_capacity(1000),
            trend: 0.0,
            seasonal_component: Vec::new(),
        }
    }

    pub fn add_data(&mut self, newdata: Vec<f64>) -> CoreResult<()> {
        for value in newdata {
            self.data.push_back(value);
        }

        // Keep only recent data
        while self.data.len() > 1000 {
            self.data.pop_front();
        }

        // Update trend analysis
        self.update_trend()?;

        Ok(())
    }

    fn update_trend(&mut self) -> CoreResult<()> {
        if self.data.len() < 2 {
            return Ok(());
        }

        // Simple trend calculation
        let recent_data: Vec<_> = self.data.iter().rev().take(10).cloned().collect();
        if recent_data.len() >= 2 {
            self.trend =
                (recent_data[0] - recent_data[recent_data.len() - 1]) / recent_data.len() as f64;
        }

        Ok(())
    }

    pub fn predict_next(&self, steps: usize) -> Vec<f64> {
        if self.data.is_empty() {
            return vec![0.0; steps];
        }

        let last_value = *self.data.back().unwrap();
        let mut predictions = Vec::with_capacity(steps);

        for i in 0..steps {
            let predicted_value = last_value + self.trend * (i + 1) as f64;
            predictions.push(predicted_value.clamp(0.0, 1.0)); // Clamp to reasonable range
        }

        predictions
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct CorrelationAnalyzer {
    correlations: HashMap<(String, String), f64>,
}

impl Default for CorrelationAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrelationAnalyzer {
    pub fn new() -> Self {
        Self {
            correlations: HashMap::new(),
        }
    }

    pub fn analyze_correlations(
        &mut self,
        data: &[ComprehensivePerformanceMetrics],
    ) -> CoreResult<()> {
        if data.len() < 10 {
            return Ok(());
        }

        // Calculate correlation between CPU and throughput
        let cpu_data: Vec<f64> = data.iter().map(|m| m.cpu_utilization).collect();
        let throughput_data: Vec<f64> = data.iter().map(|m| m.operations_per_second).collect();
        let cpu_throughput_correlation = self.calculate_correlation(&cpu_data, &throughput_data);
        self.correlations.insert(
            ("cpu".to_string(), "throughput".to_string()),
            cpu_throughput_correlation,
        );

        // Calculate correlation between memory and latency
        let memory_data: Vec<f64> = data.iter().map(|m| m.memory_utilization).collect();
        let latency_data: Vec<f64> = data.iter().map(|m| m.average_latency_ms).collect();
        let memory_latency_correlation = self.calculate_correlation(&memory_data, &latency_data);
        self.correlations.insert(
            ("memory".to_string(), "latency".to_string()),
            memory_latency_correlation,
        );

        Ok(())
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct PatternDetector {
    detected_patterns: Vec<DetectedPattern>,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    pub fn new() -> Self {
        Self {
            detected_patterns: Vec::new(),
        }
    }

    pub fn detect_patterns(&mut self, data: &[ComprehensivePerformanceMetrics]) -> CoreResult<()> {
        // Simple pattern detection logic
        if data.len() < 20 {
            return Ok(());
        }

        // Detect periodic patterns in CPU usage
        let cpu_data: Vec<f64> = data.iter().map(|m| m.cpu_utilization).collect();
        if let Some(period) = self.detect_periodicity(&cpu_data) {
            self.detected_patterns.push(DetectedPattern {
                pattern_type: PatternType::Periodic,
                metric: "cpu_utilization".to_string(),
                period: Some(period),
                confidence: 0.7,
                detected_at: Instant::now(),
            });
        }

        Ok(())
    }

    fn detect_periodicity(&self, data: &[f64]) -> Option<usize> {
        // Simple autocorrelation-based periodicity detection
        let max_period = data.len() / 4;
        let mut best_period = None;
        let mut best_correlation = 0.0;

        for period in 2..=max_period {
            if data.len() < 2 * period {
                continue;
            }

            let first_half = &data[0..period];
            let second_half = &data[period..2 * period];

            let correlation = self.calculate_simple_correlation(first_half, second_half);

            if correlation > best_correlation && correlation > 0.7 {
                best_correlation = correlation;
                best_period = Some(period);
            }
        }

        best_period
    }

    fn calculate_simple_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let mean_x = x.iter().sum::<f64>() / x.len() as f64;
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DetectedPattern {
    pub pattern_type: PatternType,
    pub metric: String,
    pub period: Option<usize>,
    pub confidence: f64,
    pub detected_at: Instant,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    Periodic,
    Trending,
    Seasonal,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub id: String,
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: Instant,
    pub acknowledged: bool,
    pub resolved: bool,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AlertRule {
    pub name: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub duration: Duration,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum AlertCondition {
    Threshold {
        metric: String,
        operator: ComparisonOperator,
        value: f64,
    },
    RateOfChange {
        metric: String,
        threshold: f64,
        timeframe: Duration,
    },
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AlertEvent {
    pub alert: PerformanceAlert,
    pub event_type: AlertEventType,
    pub timestamp: Instant,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertEventType {
    Triggered,
    Acknowledged,
    Resolved,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct NotificationChannel {
    channel_type: NotificationChannelType,
    #[allow(dead_code)]
    endpoint: String,
    enabled: bool,
}

impl NotificationChannel {
    pub fn send_notification(&self, alertname: &str, severity: AlertSeverity) -> CoreResult<()> {
        if !self.enabled {
            return Ok(());
        }

        match &self.channel_type {
            NotificationChannelType::Email => {
                // Send email notification
                println!("EMAIL ALERT: {alertname} - {severity:?}");
            }
            NotificationChannelType::Slack => {
                // Send Slack notification
                println!("SLACK ALERT: {alertname} - {severity:?}");
            }
            NotificationChannelType::Webhook => {
                // Send webhook notification
                println!("WEBHOOK ALERT: {alertname} - {severity:?}");
            }
        }

        Ok(())
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum NotificationChannelType {
    Email,
    Slack,
    Webhook,
}