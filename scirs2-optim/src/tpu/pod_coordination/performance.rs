//! Performance Analysis for TPU Pod Coordination
//!
//! This module provides comprehensive performance analysis functionality for TPU pod coordination,
//! including metrics collection, performance monitoring, and optimization recommendations.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::{DeviceId, PodTopology};
use crate::error::Result;

/// Pod performance metrics
#[derive(Debug, Clone)]
pub struct PodPerformanceMetrics {
    /// Throughput (operations/second)
    pub throughput: f64,

    /// Latency
    pub latency: Duration,

    /// Utilization (0.0 to 1.0)
    pub utilization: f64,

    /// Efficiency (0.0 to 1.0)
    pub efficiency: f64,

    /// Power consumption (watts)
    pub power_consumption: f64,

    /// Temperature (celsius)
    pub temperature: f64,
}

/// Device performance metrics
#[derive(Debug, Clone)]
pub struct DevicePerformanceMetrics {
    /// Device ID
    pub device_id: DeviceId,

    /// Compute utilization
    pub compute_utilization: f64,

    /// Memory utilization
    pub memory_utilization: f64,

    /// Communication utilization
    pub communication_utilization: f64,

    /// Power consumption
    pub power_consumption: f64,

    /// Temperature
    pub temperature: f64,

    /// Throughput
    pub throughput: f64,

    /// Error rate
    pub error_rate: f64,

    /// Queue depth
    pub queue_depth: usize,
}

/// Performance benchmark
#[derive(Debug, Clone)]
pub struct PerformanceBenchmark {
    /// Benchmark name
    pub name: String,

    /// Expected throughput
    pub expected_throughput: f64,

    /// Expected latency
    pub expected_latency: Duration,

    /// Maximum utilization
    pub max_utilization: f64,

    /// Baseline metrics
    pub baseline_metrics: PodPerformanceMetrics,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    /// Alert type
    pub alert_type: AlertType,

    /// Severity
    pub severity: AlertSeverity,

    /// Device ID (if device-specific)
    pub device_id: Option<DeviceId>,

    /// Message
    pub message: String,

    /// Timestamp
    pub timestamp: Instant,

    /// Metric value
    pub metric_value: f64,

    /// Threshold
    pub threshold: f64,
}

/// Alert types
#[derive(Debug, Clone, Copy)]
pub enum AlertType {
    HighLatency,
    LowThroughput,
    HighUtilization,
    LowUtilization,
    HighTemperature,
    HighPowerConsumption,
    HighErrorRate,
    QueueBacklog,
    PerformanceDegradation,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Performance optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Target devices
    pub target_devices: Vec<DeviceId>,

    /// Expected improvement
    pub expected_improvement: f64,

    /// Implementation effort
    pub implementation_effort: EffortLevel,

    /// Description
    pub description: String,

    /// Priority
    pub priority: RecommendationPriority,
}

/// Recommendation types
#[derive(Debug, Clone, Copy)]
pub enum RecommendationType {
    LoadBalancing,
    ResourceScaling,
    ConfigurationTuning,
    WorkloadOptimization,
    CommunicationOptimization,
    MemoryOptimization,
    ThermalManagement,
    PowerOptimization,
}

/// Implementation effort levels
#[derive(Debug, Clone, Copy)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Recommendation priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecommendationPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance trend
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    /// Metric name
    pub metric_name: String,

    /// Trend direction
    pub trend_direction: TrendDirection,

    /// Trend magnitude
    pub trend_magnitude: f64,

    /// Confidence level
    pub confidence: f64,

    /// Time window
    pub time_window: Duration,
}

/// Trend directions
#[derive(Debug, Clone, Copy)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Coordination strategies
#[derive(Debug, Clone, Copy)]
pub enum CoordinationStrategy {
    Centralized,
    Decentralized,
    Hierarchical,
    Ring,
    Mesh,
    Adaptive,
}

/// Communication patterns
#[derive(Debug, Clone, Copy)]
pub enum CommunicationPattern {
    AllReduce,
    AllGather,
    ReduceScatter,
    Broadcast,
    AllToAll,
    ParameterServer,
    Ring,
    Tree,
    Butterfly,
    Hypercube,
}

/// Synchronization modes
#[derive(Debug, Clone, Copy)]
pub enum SynchronizationMode {
    Synchronous,
    Asynchronous,
    BulkSynchronous,
    Bounded,
    StaleStynchronous,
    Adaptive,
}

/// Batch parallelization strategies
#[derive(Debug, Clone, Copy)]
pub enum BatchParallelizationStrategy {
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Hybrid,
    HybridParallel,
    TensorParallel,
    ExpertParallel,
    Adaptive,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingStrategy {
    Static,
    Dynamic,
    PredictiveDynamic,
    WorkStealing,
    LoadAware,
    LatencyAware,
    BandwidthAware,
    Adaptive,
}

/// Configuration for pod coordination
#[derive(Debug, Clone)]
pub struct PodCoordinationConfig {
    /// Pod topology
    pub topology: PodTopology,

    /// Number of devices
    pub num_devices: usize,

    /// Coordination strategy
    pub coordination_strategy: CoordinationStrategy,

    /// Communication pattern
    pub communication_pattern: CommunicationPattern,

    /// Synchronization mode
    pub synchronization_mode: SynchronizationMode,

    /// Batch strategy
    pub batch_strategy: BatchParallelizationStrategy,

    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,

    /// Enable adaptive optimization
    pub adaptive_optimization: bool,
}

/// Pod performance analyzer
#[derive(Debug)]
pub struct PodPerformanceAnalyzer {
    /// Configuration
    config: PodCoordinationConfig,

    /// Metrics history
    metrics_history: VecDeque<PodPerformanceMetrics>,

    /// Device metrics
    device_metrics: HashMap<DeviceId, VecDeque<DevicePerformanceMetrics>>,

    /// Performance benchmarks
    benchmarks: Vec<PerformanceBenchmark>,

    /// Active alerts
    active_alerts: Vec<PerformanceAlert>,

    /// Start time
    start_time: Instant,

    /// Alert thresholds
    alert_thresholds: HashMap<AlertType, f64>,

    /// Optimization recommendations
    recommendations: Vec<OptimizationRecommendation>,
}

impl PodPerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new(config: &PodCoordinationConfig) -> Result<Self> {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(AlertType::HighLatency, 10.0); // 10ms
        alert_thresholds.insert(AlertType::LowThroughput, 100.0); // 100 ops/sec
        alert_thresholds.insert(AlertType::HighUtilization, 0.9); // 90%
        alert_thresholds.insert(AlertType::LowUtilization, 0.1); // 10%
        alert_thresholds.insert(AlertType::HighTemperature, 80.0); // 80°C
        alert_thresholds.insert(AlertType::HighPowerConsumption, 1000.0); // 1000W
        alert_thresholds.insert(AlertType::HighErrorRate, 0.01); // 1%
        alert_thresholds.insert(AlertType::QueueBacklog, 100.0); // 100 items

        Ok(Self {
            config: config.clone(),
            metrics_history: VecDeque::with_capacity(1000),
            device_metrics: HashMap::new(),
            benchmarks: Self::create_default_benchmarks(),
            active_alerts: Vec::new(),
            start_time: Instant::now(),
            alert_thresholds,
            recommendations: Vec::new(),
        })
    }

    /// Create default performance benchmarks
    fn create_default_benchmarks() -> Vec<PerformanceBenchmark> {
        vec![
            PerformanceBenchmark {
                name: "High Throughput".to_string(),
                expected_throughput: 1000.0,
                expected_latency: Duration::from_millis(5),
                max_utilization: 0.9,
                baseline_metrics: PodPerformanceMetrics {
                    throughput: 1000.0,
                    latency: Duration::from_millis(5),
                    utilization: 0.8,
                    efficiency: 0.9,
                    power_consumption: 800.0,
                    temperature: 65.0,
                },
            },
            PerformanceBenchmark {
                name: "Low Latency".to_string(),
                expected_throughput: 500.0,
                expected_latency: Duration::from_millis(2),
                max_utilization: 0.7,
                baseline_metrics: PodPerformanceMetrics {
                    throughput: 500.0,
                    latency: Duration::from_millis(2),
                    utilization: 0.6,
                    efficiency: 0.85,
                    power_consumption: 600.0,
                    temperature: 55.0,
                },
            },
        ]
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> PodPerformanceMetrics {
        let elapsed = self.start_time.elapsed().as_secs_f64();

        // Calculate dynamic throughput based on configuration and load
        let base_throughput = match self.config.topology {
            PodTopology::Single => 100.0,
            PodTopology::Pod2x2 => 400.0,
            PodTopology::Pod4x4 => 1200.0,
            PodTopology::Pod8x8 => 4800.0,
            PodTopology::Pod16x16 => 19200.0,
            PodTopology::Pod32x32 => 76800.0,
        };

        let efficiency_factor = match self.config.coordination_strategy {
            CoordinationStrategy::Centralized => 0.85,
            CoordinationStrategy::Decentralized => 0.92,
            CoordinationStrategy::Hierarchical => 0.89,
            CoordinationStrategy::Ring => 0.87,
            CoordinationStrategy::Mesh => 0.94,
            CoordinationStrategy::Adaptive => 0.96,
        };

        // Simulate realistic performance variations
        let workload_factor = if self.config.adaptive_optimization { 1.1 } else { 1.0 };
        let time_variation = 1.0 + (elapsed * 0.1).sin() * 0.05; // ±5% variation
        let throughput = base_throughput * efficiency_factor * workload_factor * time_variation;

        // Calculate latency based on throughput and batch strategy
        let base_latency_ms = match self.config.batch_strategy {
            BatchParallelizationStrategy::DataParallel => 3.0,
            BatchParallelizationStrategy::ModelParallel => 8.0,
            BatchParallelizationStrategy::PipelineParallel => 5.0,
            BatchParallelizationStrategy::Hybrid => 4.0,
            BatchParallelizationStrategy::HybridParallel => 4.5,
            BatchParallelizationStrategy::TensorParallel => 6.0,
            BatchParallelizationStrategy::ExpertParallel => 7.0,
            BatchParallelizationStrategy::Adaptive => 3.5,
        };

        let communication_overhead = match self.config.communication_pattern {
            CommunicationPattern::AllReduce => 1.2,
            CommunicationPattern::AllGather => 1.5,
            CommunicationPattern::ReduceScatter => 1.1,
            CommunicationPattern::Broadcast => 0.8,
            CommunicationPattern::AllToAll => 2.0,
            _ => 1.0, // Fallback for other patterns
        };

        let latency = Duration::from_millis(
            (base_latency_ms * communication_overhead * (1.0 + elapsed * 0.001)) as u64,
        );

        // Calculate overall utilization
        let device_count = self.config.num_devices as f64;
        let utilization = (0.7 + device_count / 100.0).min(0.95);

        // Calculate efficiency based on various factors
        let sync_efficiency = match self.config.synchronization_mode {
            SynchronizationMode::Synchronous => 0.95,
            SynchronizationMode::Asynchronous => 0.88,
            SynchronizationMode::BulkSynchronous => 0.92,
            SynchronizationMode::Bounded => 0.90,
            SynchronizationMode::StaleStynchronous => 0.85,
            SynchronizationMode::Adaptive => 0.93,
        };

        let load_balance_efficiency = match self.config.load_balancing_strategy {
            LoadBalancingStrategy::Static => 0.85,
            LoadBalancingStrategy::Dynamic => 0.93,
            LoadBalancingStrategy::WorkStealing => 0.91,
            LoadBalancingStrategy::LoadAware => 0.94,
            LoadBalancingStrategy::LatencyAware => 0.92,
            LoadBalancingStrategy::BandwidthAware => 0.89,
            LoadBalancingStrategy::Adaptive => 0.95,
            LoadBalancingStrategy::PredictiveDynamic => 0.96,
        };

        let efficiency = (sync_efficiency + load_balance_efficiency) / 2.0;

        // Calculate power consumption based on utilization and device count
        let base_power_per_device = 15.0; // Watts per TPU
        let power_consumption = device_count * base_power_per_device * (utilization + 0.2);

        // Temperature simulation based on power and time
        let thermal_factor = 1.0 + (power_consumption / 1000.0) * 0.3;
        let ambient_temp = 25.0; // Base ambient temperature
        let thermal_rise = 30.0 * thermal_factor * utilization;
        let cooling_efficiency = if elapsed > 600.0 { 0.9 } else { 1.0 }; // Cooling degrades over time
        let temperature = ambient_temp + (thermal_rise * cooling_efficiency);

        PodPerformanceMetrics {
            throughput,
            latency,
            utilization,
            efficiency,
            power_consumption,
            temperature,
        }
    }

    /// Record performance metrics
    pub fn record_metrics(&mut self, metrics: PodPerformanceMetrics) {
        self.metrics_history.push_back(metrics.clone());
        if self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }

        // Check for alerts
        self.check_performance_alerts(&metrics);

        // Generate recommendations if needed
        self.generate_recommendations(&metrics);
    }

    /// Record device metrics
    pub fn record_device_metrics(&mut self, device_id: DeviceId, metrics: DevicePerformanceMetrics) {
        let device_history = self.device_metrics.entry(device_id).or_insert_with(|| VecDeque::with_capacity(100));
        device_history.push_back(metrics);
        if device_history.len() > 100 {
            device_history.pop_front();
        }
    }

    /// Check for performance alerts
    fn check_performance_alerts(&mut self, metrics: &PodPerformanceMetrics) {
        // Clear old alerts
        self.active_alerts.retain(|alert| alert.timestamp.elapsed() < Duration::from_secs(300));

        // Check latency
        if let Some(&threshold) = self.alert_thresholds.get(&AlertType::HighLatency) {
            if metrics.latency.as_millis() as f64 > threshold {
                self.create_alert(
                    AlertType::HighLatency,
                    AlertSeverity::Warning,
                    None,
                    format!("High latency detected: {:.2}ms", metrics.latency.as_millis()),
                    metrics.latency.as_millis() as f64,
                    threshold,
                );
            }
        }

        // Check throughput
        if let Some(&threshold) = self.alert_thresholds.get(&AlertType::LowThroughput) {
            if metrics.throughput < threshold {
                self.create_alert(
                    AlertType::LowThroughput,
                    AlertSeverity::Warning,
                    None,
                    format!("Low throughput detected: {:.2} ops/sec", metrics.throughput),
                    metrics.throughput,
                    threshold,
                );
            }
        }

        // Check utilization
        if let Some(&threshold) = self.alert_thresholds.get(&AlertType::HighUtilization) {
            if metrics.utilization > threshold {
                self.create_alert(
                    AlertType::HighUtilization,
                    AlertSeverity::Critical,
                    None,
                    format!("High utilization detected: {:.1}%", metrics.utilization * 100.0),
                    metrics.utilization,
                    threshold,
                );
            }
        }

        // Check temperature
        if let Some(&threshold) = self.alert_thresholds.get(&AlertType::HighTemperature) {
            if metrics.temperature > threshold {
                self.create_alert(
                    AlertType::HighTemperature,
                    AlertSeverity::Critical,
                    None,
                    format!("High temperature detected: {:.1}°C", metrics.temperature),
                    metrics.temperature,
                    threshold,
                );
            }
        }

        // Check power consumption
        if let Some(&threshold) = self.alert_thresholds.get(&AlertType::HighPowerConsumption) {
            if metrics.power_consumption > threshold {
                self.create_alert(
                    AlertType::HighPowerConsumption,
                    AlertSeverity::Warning,
                    None,
                    format!("High power consumption detected: {:.1}W", metrics.power_consumption),
                    metrics.power_consumption,
                    threshold,
                );
            }
        }
    }

    /// Create performance alert
    fn create_alert(
        &mut self,
        alert_type: AlertType,
        severity: AlertSeverity,
        device_id: Option<DeviceId>,
        message: String,
        metric_value: f64,
        threshold: f64,
    ) {
        let alert = PerformanceAlert {
            alert_type,
            severity,
            device_id,
            message,
            timestamp: Instant::now(),
            metric_value,
            threshold,
        };

        self.active_alerts.push(alert);
    }

    /// Generate optimization recommendations
    fn generate_recommendations(&mut self, metrics: &PodPerformanceMetrics) {
        self.recommendations.clear();

        // High utilization recommendation
        if metrics.utilization > 0.85 {
            self.recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::LoadBalancing,
                target_devices: vec![], // Apply to all devices
                expected_improvement: 0.15,
                implementation_effort: EffortLevel::Medium,
                description: "Implement dynamic load balancing to redistribute workload".to_string(),
                priority: RecommendationPriority::High,
            });
        }

        // High temperature recommendation
        if metrics.temperature > 75.0 {
            self.recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::ThermalManagement,
                target_devices: vec![],
                expected_improvement: 0.10,
                implementation_effort: EffortLevel::High,
                description: "Implement thermal throttling or improve cooling".to_string(),
                priority: RecommendationPriority::Critical,
            });
        }

        // Low efficiency recommendation
        if metrics.efficiency < 0.8 {
            self.recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::ConfigurationTuning,
                target_devices: vec![],
                expected_improvement: 0.20,
                implementation_effort: EffortLevel::Low,
                description: "Tune coordination and communication parameters".to_string(),
                priority: RecommendationPriority::Medium,
            });
        }

        // High power consumption recommendation
        if metrics.power_consumption > 800.0 {
            self.recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::PowerOptimization,
                target_devices: vec![],
                expected_improvement: 0.12,
                implementation_effort: EffortLevel::Medium,
                description: "Implement power-aware scheduling and DVFS".to_string(),
                priority: RecommendationPriority::Medium,
            });
        }
    }

    /// Calculate performance trends
    pub fn calculate_trends(&self) -> Vec<PerformanceTrend> {
        let mut trends = Vec::new();

        if self.metrics_history.len() < 10 {
            return trends; // Not enough data for trend analysis
        }

        // Throughput trend
        let throughput_values: Vec<f64> = self.metrics_history.iter().map(|m| m.throughput).collect();
        if let Some(trend) = self.calculate_metric_trend("throughput", &throughput_values) {
            trends.push(trend);
        }

        // Latency trend
        let latency_values: Vec<f64> = self.metrics_history.iter().map(|m| m.latency.as_millis() as f64).collect();
        if let Some(trend) = self.calculate_metric_trend("latency", &latency_values) {
            trends.push(trend);
        }

        // Utilization trend
        let utilization_values: Vec<f64> = self.metrics_history.iter().map(|m| m.utilization).collect();
        if let Some(trend) = self.calculate_metric_trend("utilization", &utilization_values) {
            trends.push(trend);
        }

        // Temperature trend
        let temperature_values: Vec<f64> = self.metrics_history.iter().map(|m| m.temperature).collect();
        if let Some(trend) = self.calculate_metric_trend("temperature", &temperature_values) {
            trends.push(trend);
        }

        trends
    }

    /// Calculate trend for a specific metric
    fn calculate_metric_trend(&self, metric_name: &str, values: &[f64]) -> Option<PerformanceTrend> {
        if values.len() < 10 {
            return None;
        }

        // Simple linear regression
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = x_values.iter().zip(values.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = x_values.iter()
            .map(|x| (x - x_mean).powi(2))
            .sum();

        if denominator == 0.0 {
            return None;
        }

        let slope = numerator / denominator;

        // Determine trend direction and magnitude
        let (trend_direction, trend_magnitude) = if slope.abs() < 0.01 {
            (TrendDirection::Stable, slope.abs())
        } else if slope > 0.0 {
            (TrendDirection::Increasing, slope)
        } else {
            (TrendDirection::Decreasing, slope.abs())
        };

        // Calculate R-squared for confidence
        let y_pred: Vec<f64> = x_values.iter().map(|x| y_mean + slope * (x - x_mean)).collect();
        let ss_res: f64 = values.iter().zip(y_pred.iter())
            .map(|(y, y_p)| (y - y_p).powi(2))
            .sum();
        let ss_tot: f64 = values.iter()
            .map(|y| (y - y_mean).powi(2))
            .sum();

        let confidence = if ss_tot == 0.0 { 1.0 } else { 1.0 - (ss_res / ss_tot) };

        Some(PerformanceTrend {
            metric_name: metric_name.to_string(),
            trend_direction,
            trend_magnitude,
            confidence: confidence.max(0.0).min(1.0),
            time_window: Duration::from_secs(self.metrics_history.len() as u64 * 5), // Assuming 5-second intervals
        })
    }

    /// Compare with benchmarks
    pub fn compare_with_benchmarks(&self, metrics: &PodPerformanceMetrics) -> Vec<(String, f64)> {
        let mut comparisons = Vec::new();

        for benchmark in &self.benchmarks {
            let throughput_ratio = metrics.throughput / benchmark.expected_throughput;
            let latency_ratio = benchmark.expected_latency.as_millis() as f64 / metrics.latency.as_millis() as f64;
            let utilization_ratio = metrics.utilization / benchmark.max_utilization;

            // Overall performance score (higher is better)
            let performance_score = (throughput_ratio + latency_ratio) / 2.0 * utilization_ratio;

            comparisons.push((benchmark.name.clone(), performance_score));
        }

        comparisons
    }

    /// Get performance summary
    pub fn get_performance_summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();

        if let Some(latest_metrics) = self.metrics_history.back() {
            summary.insert("current_throughput".to_string(), latest_metrics.throughput);
            summary.insert("current_latency_ms".to_string(), latest_metrics.latency.as_millis() as f64);
            summary.insert("current_utilization".to_string(), latest_metrics.utilization);
            summary.insert("current_efficiency".to_string(), latest_metrics.efficiency);
            summary.insert("current_power_w".to_string(), latest_metrics.power_consumption);
            summary.insert("current_temperature_c".to_string(), latest_metrics.temperature);
        }

        // Calculate averages
        if !self.metrics_history.is_empty() {
            let avg_throughput = self.metrics_history.iter().map(|m| m.throughput).sum::<f64>() / self.metrics_history.len() as f64;
            let avg_latency = self.metrics_history.iter().map(|m| m.latency.as_millis() as f64).sum::<f64>() / self.metrics_history.len() as f64;
            let avg_utilization = self.metrics_history.iter().map(|m| m.utilization).sum::<f64>() / self.metrics_history.len() as f64;
            let avg_efficiency = self.metrics_history.iter().map(|m| m.efficiency).sum::<f64>() / self.metrics_history.len() as f64;

            summary.insert("avg_throughput".to_string(), avg_throughput);
            summary.insert("avg_latency_ms".to_string(), avg_latency);
            summary.insert("avg_utilization".to_string(), avg_utilization);
            summary.insert("avg_efficiency".to_string(), avg_efficiency);
        }

        summary.insert("active_alerts".to_string(), self.active_alerts.len() as f64);
        summary.insert("recommendations".to_string(), self.recommendations.len() as f64);
        summary.insert("uptime_seconds".to_string(), self.start_time.elapsed().as_secs() as f64);

        summary
    }

    /// Get active alerts
    pub fn get_active_alerts(&self) -> &[PerformanceAlert] {
        &self.active_alerts
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> &[OptimizationRecommendation] {
        &self.recommendations
    }

    /// Update alert threshold
    pub fn set_alert_threshold(&mut self, alert_type: AlertType, threshold: f64) {
        self.alert_thresholds.insert(alert_type, threshold);
    }

    /// Add custom benchmark
    pub fn add_benchmark(&mut self, benchmark: PerformanceBenchmark) {
        self.benchmarks.push(benchmark);
    }

    /// Get metrics history
    pub fn get_metrics_history(&self) -> &VecDeque<PodPerformanceMetrics> {
        &self.metrics_history
    }

    /// Get device metrics
    pub fn get_device_metrics(&self, device_id: DeviceId) -> Option<&VecDeque<DevicePerformanceMetrics>> {
        self.device_metrics.get(&device_id)
    }
}

// Default implementations
impl Default for PodPerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::from_millis(0),
            utilization: 0.0,
            efficiency: 0.0,
            power_consumption: 0.0,
            temperature: 25.0,
        }
    }
}

impl Default for DevicePerformanceMetrics {
    fn default() -> Self {
        Self {
            device_id: DeviceId(0),
            compute_utilization: 0.0,
            memory_utilization: 0.0,
            communication_utilization: 0.0,
            power_consumption: 0.0,
            temperature: 25.0,
            throughput: 0.0,
            error_rate: 0.0,
            queue_depth: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_analyzer_creation() {
        let config = PodCoordinationConfig {
            topology: PodTopology::Pod4x4,
            num_devices: 16,
            coordination_strategy: CoordinationStrategy::Hierarchical,
            communication_pattern: CommunicationPattern::AllReduce,
            synchronization_mode: SynchronizationMode::Synchronous,
            batch_strategy: BatchParallelizationStrategy::DataParallel,
            load_balancing_strategy: LoadBalancingStrategy::Dynamic,
            adaptive_optimization: true,
        };

        let analyzer = PodPerformanceAnalyzer::new(&config);
        assert!(analyzer.is_ok());
    }

    #[test]
    fn test_metrics_recording() {
        let config = PodCoordinationConfig {
            topology: PodTopology::Pod4x4,
            num_devices: 16,
            coordination_strategy: CoordinationStrategy::Hierarchical,
            communication_pattern: CommunicationPattern::AllReduce,
            synchronization_mode: SynchronizationMode::Synchronous,
            batch_strategy: BatchParallelizationStrategy::DataParallel,
            load_balancing_strategy: LoadBalancingStrategy::Dynamic,
            adaptive_optimization: true,
        };

        let mut analyzer = PodPerformanceAnalyzer::new(&config).unwrap();
        let metrics = PodPerformanceMetrics::default();

        analyzer.record_metrics(metrics);
        assert_eq!(analyzer.get_metrics_history().len(), 1);
    }

    #[test]
    fn test_trend_calculation() {
        let config = PodCoordinationConfig {
            topology: PodTopology::Pod4x4,
            num_devices: 16,
            coordination_strategy: CoordinationStrategy::Hierarchical,
            communication_pattern: CommunicationPattern::AllReduce,
            synchronization_mode: SynchronizationMode::Synchronous,
            batch_strategy: BatchParallelizationStrategy::DataParallel,
            load_balancing_strategy: LoadBalancingStrategy::Dynamic,
            adaptive_optimization: true,
        };

        let mut analyzer = PodPerformanceAnalyzer::new(&config).unwrap();

        // Add some test data
        for i in 0..15 {
            let metrics = PodPerformanceMetrics {
                throughput: 100.0 + i as f64 * 10.0, // Increasing trend
                latency: Duration::from_millis(5),
                utilization: 0.5,
                efficiency: 0.8,
                power_consumption: 500.0,
                temperature: 60.0,
            };
            analyzer.record_metrics(metrics);
        }

        let trends = analyzer.calculate_trends();
        assert!(!trends.is_empty());

        // Should detect increasing throughput trend
        let throughput_trend = trends.iter().find(|t| t.metric_name == "throughput");
        assert!(throughput_trend.is_some());
        assert!(matches!(throughput_trend.unwrap().trend_direction, TrendDirection::Increasing));
    }
}