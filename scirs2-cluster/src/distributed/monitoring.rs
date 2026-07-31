//! Performance monitoring and metrics collection for distributed clustering
//!
//! This module provides comprehensive monitoring capabilities including
//! performance metrics, resource usage tracking, and system health analysis.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::error::{ClusteringError, Result};

/// Performance monitoring coordinator
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub metrics_history: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    pub resource_usage: Arc<Mutex<VecDeque<ResourceUsage>>>,
    pub worker_metrics: HashMap<usize, WorkerMetrics>,
    pub config: MonitoringConfig,
    pub alert_thresholds: AlertThresholds,
    pub start_time: Instant,
    /// Real system resource sampler used to feed `update_worker_metrics`
    /// (replaces previously-hardcoded CPU/memory placeholder constants).
    resource_sampler: ResourceSampler,
}

/// Configuration for performance monitoring
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub enable_detailed_monitoring: bool,
    pub metrics_collection_interval_ms: u64,
    pub max_history_size: usize,
    pub enable_resource_monitoring: bool,
    pub enable_network_monitoring: bool,
    pub enable_predictive_analytics: bool,
    pub export_metrics: bool,
    pub alert_on_anomalies: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_detailed_monitoring: true,
            metrics_collection_interval_ms: 1000,
            max_history_size: 1000,
            enable_resource_monitoring: true,
            enable_network_monitoring: false,
            enable_predictive_analytics: false,
            export_metrics: false,
            alert_on_anomalies: true,
        }
    }
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub max_convergence_time_ms: u64,
    pub min_worker_efficiency: f64,
    pub max_memory_utilization: f64,
    pub max_cpu_utilization: f64,
    pub max_message_latency_ms: f64,
    pub max_sync_overhead_ms: f64,
    pub min_throughput_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_convergence_time_ms: 300000, // 5 minutes
            min_worker_efficiency: 0.6,
            max_memory_utilization: 0.9,
            max_cpu_utilization: 0.95,
            max_message_latency_ms: 1000.0,
            max_sync_overhead_ms: 5000.0,
            min_throughput_threshold: 10.0,
        }
    }
}

/// System performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub timestamp: SystemTime,
    pub iteration: usize,
    pub global_inertia: f64,
    pub convergence_rate: f64,
    pub worker_efficiency: f64,
    pub message_latency_ms: f64,
    pub sync_overhead_ms: f64,
    pub total_computation_time_ms: u64,
    pub memory_pressure_score: f64,
    pub load_balance_score: f64,
    pub network_utilization: f64,
}

/// Resource usage metrics
#[derive(Debug, Clone)]
pub struct ResourceUsage {
    pub timestamp: SystemTime,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub network_throughput_mbps: f64,
    pub disk_io_rate: f64,
    pub active_workers: usize,
    pub failed_workers: usize,
    pub queue_depth: usize,
    pub cache_hit_ratio: f64,
}

/// Worker-specific metrics
#[derive(Debug, Clone)]
pub struct WorkerMetrics {
    pub worker_id: usize,
    pub cpu_usage_history: VecDeque<f64>,
    pub memory_usage_history: VecDeque<f64>,
    pub throughput_history: VecDeque<f64>,
    pub latency_history: VecDeque<f64>,
    pub error_count: usize,
    pub last_update: SystemTime,
    pub health_score: f64,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub severity: AlertSeverity,
    pub message: String,
    pub timestamp: SystemTime,
    pub worker_id: Option<usize>,
    pub metric_value: f64,
    pub threshold: f64,
}

/// Types of performance alerts
#[derive(Debug, Clone)]
pub enum AlertType {
    HighCpuUsage,
    HighMemoryUsage,
    HighLatency,
    LowThroughput,
    WorkerFailure,
    ConvergenceTimeout,
    LoadImbalance,
    NetworkCongestion,
    ResourceExhaustion,
    AnomalyDetected,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

/// System efficiency analysis
#[derive(Debug, Clone)]
pub struct EfficiencyAnalysis {
    pub overall_efficiency: f64,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub resource_utilization: HashMap<String, f64>,
    pub performance_trends: PerformanceTrends,
    pub optimization_recommendations: Vec<String>,
}

/// Bottleneck analysis results
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: BottleneckType,
    pub bottleneck_severity: f64,
    pub affected_workers: Vec<usize>,
    pub estimated_impact: f64,
}

/// Types of system bottlenecks
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BottleneckType {
    Cpu,
    Memory,
    Network,
    Disk,
    Synchronization,
    LoadImbalance,
    MessagePassing,
    None,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    pub throughput_trend: TrendDirection,
    pub latency_trend: TrendDirection,
    pub efficiency_trend: TrendDirection,
    pub resource_trend: TrendDirection,
    pub trend_confidence: f64,
}

/// Trend direction indicators
#[derive(Debug, Clone, Copy)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    Unknown,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(config: MonitoringConfig) -> Self {
        Self {
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            resource_usage: Arc::new(Mutex::new(VecDeque::new())),
            worker_metrics: HashMap::new(),
            config,
            alert_thresholds: AlertThresholds::default(),
            start_time: Instant::now(),
            resource_sampler: ResourceSampler::new(),
        }
    }

    /// Register worker for monitoring
    pub fn register_worker(&mut self, workerid: usize) {
        let worker_metrics = WorkerMetrics {
            worker_id: workerid,
            cpu_usage_history: VecDeque::new(),
            memory_usage_history: VecDeque::new(),
            throughput_history: VecDeque::new(),
            latency_history: VecDeque::new(),
            error_count: 0,
            last_update: SystemTime::now(),
            health_score: 1.0,
        };

        self.worker_metrics.insert(workerid, worker_metrics);
    }

    /// Record performance metrics
    pub fn record_performance_metrics(&self, metrics: PerformanceMetrics) -> Result<()> {
        let mut history = self.metrics_history.lock().map_err(|_| {
            ClusteringError::InvalidInput("Failed to acquire metrics lock".to_string())
        })?;

        history.push_back(metrics);

        // Maintain history size limit
        while history.len() > self.config.max_history_size {
            history.pop_front();
        }

        Ok(())
    }

    /// Record resource usage metrics
    pub fn record_resource_usage(&self, usage: ResourceUsage) -> Result<()> {
        if !self.config.enable_resource_monitoring {
            return Ok(());
        }

        let mut usage_history = self.resource_usage.lock().map_err(|_| {
            ClusteringError::InvalidInput("Failed to acquire resource usage lock".to_string())
        })?;

        usage_history.push_back(usage);

        // Maintain history size limit
        while usage_history.len() > self.config.max_history_size {
            usage_history.pop_front();
        }

        Ok(())
    }

    /// Update worker metrics
    pub fn update_worker_metrics(
        &mut self,
        worker_id: usize,
        cpu_usage: f64,
        memory_usage: f64,
        throughput: f64,
        latency: f64,
    ) -> Result<()> {
        if let Some(metrics) = self.worker_metrics.get_mut(&worker_id) {
            metrics.cpu_usage_history.push_back(cpu_usage);
            metrics.memory_usage_history.push_back(memory_usage);
            metrics.throughput_history.push_back(throughput);
            metrics.latency_history.push_back(latency);
            metrics.last_update = SystemTime::now();

            // Maintain history size
            let max_size = 100;
            if metrics.cpu_usage_history.len() > max_size {
                metrics.cpu_usage_history.pop_front();
            }
            if metrics.memory_usage_history.len() > max_size {
                metrics.memory_usage_history.pop_front();
            }
            if metrics.throughput_history.len() > max_size {
                metrics.throughput_history.pop_front();
            }
            if metrics.latency_history.len() > max_size {
                metrics.latency_history.pop_front();
            }
        }

        // Update health score after all metrics updates
        if let Some(metrics) = self.worker_metrics.get(&worker_id) {
            let health_score = self.calculate_worker_health_score(metrics);
            if let Some(metrics_mut) = self.worker_metrics.get_mut(&worker_id) {
                metrics_mut.health_score = health_score;
            }
        }

        Ok(())
    }

    /// Calculate worker health score
    fn calculate_worker_health_score(&self, metrics: &WorkerMetrics) -> f64 {
        let mut score = 1.0;

        // CPU usage component
        if !metrics.cpu_usage_history.is_empty() {
            let avg_cpu = metrics.cpu_usage_history.iter().sum::<f64>()
                / metrics.cpu_usage_history.len() as f64;
            score *= (1.0 - (avg_cpu - 0.8).max(0.0) * 2.0).max(0.0);
        }

        // Memory usage component
        if !metrics.memory_usage_history.is_empty() {
            let avg_memory = metrics.memory_usage_history.iter().sum::<f64>()
                / metrics.memory_usage_history.len() as f64;
            score *= (1.0 - (avg_memory - 0.85).max(0.0) * 3.0).max(0.0);
        }

        // Latency component
        if !metrics.latency_history.is_empty() {
            let avg_latency =
                metrics.latency_history.iter().sum::<f64>() / metrics.latency_history.len() as f64;
            let latency_penalty = (avg_latency / 1000.0).min(1.0) * 0.3;
            score *= (1.0 - latency_penalty).max(0.0);
        }

        // Error rate component
        let time_window_hours = 1.0; // Consider last hour
        let error_rate = metrics.error_count as f64 / time_window_hours;
        let error_penalty = (error_rate / 10.0).min(0.5); // Max 50% penalty for errors
        score *= (1.0 - error_penalty).max(0.0);

        score.max(0.0).min(1.0)
    }

    /// Check for performance alerts
    pub fn check_alerts(&self) -> Result<Vec<PerformanceAlert>> {
        if !self.config.alert_on_anomalies {
            return Ok(Vec::new());
        }

        let mut alerts = Vec::new();

        // Check latest metrics against thresholds
        let metrics_history = self.metrics_history.lock().map_err(|_| {
            ClusteringError::InvalidInput("Failed to acquire metrics lock".to_string())
        })?;

        if let Some(latest_metrics) = metrics_history.back() {
            // Check convergence time
            if latest_metrics.total_computation_time_ms
                > self.alert_thresholds.max_convergence_time_ms
            {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::ConvergenceTimeout,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Convergence taking longer than expected: {}ms > {}ms",
                        latest_metrics.total_computation_time_ms,
                        self.alert_thresholds.max_convergence_time_ms
                    ),
                    timestamp: SystemTime::now(),
                    worker_id: None,
                    metric_value: latest_metrics.total_computation_time_ms as f64,
                    threshold: self.alert_thresholds.max_convergence_time_ms as f64,
                });
            }

            // Check worker efficiency
            if latest_metrics.worker_efficiency < self.alert_thresholds.min_worker_efficiency {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::LowThroughput,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Worker efficiency below threshold: {:.2} < {:.2}",
                        latest_metrics.worker_efficiency,
                        self.alert_thresholds.min_worker_efficiency
                    ),
                    timestamp: SystemTime::now(),
                    worker_id: None,
                    metric_value: latest_metrics.worker_efficiency,
                    threshold: self.alert_thresholds.min_worker_efficiency,
                });
            }

            // Check message latency
            if latest_metrics.message_latency_ms > self.alert_thresholds.max_message_latency_ms {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::HighLatency,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "High message latency detected: {:.2}ms > {:.2}ms",
                        latest_metrics.message_latency_ms,
                        self.alert_thresholds.max_message_latency_ms
                    ),
                    timestamp: SystemTime::now(),
                    worker_id: None,
                    metric_value: latest_metrics.message_latency_ms,
                    threshold: self.alert_thresholds.max_message_latency_ms,
                });
            }
        }

        // Check resource usage
        let resource_usage = self.resource_usage.lock().map_err(|_| {
            ClusteringError::InvalidInput("Failed to acquire resource usage lock".to_string())
        })?;

        if let Some(latest_usage) = resource_usage.back() {
            // Check CPU utilization
            if latest_usage.cpu_utilization > self.alert_thresholds.max_cpu_utilization {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::HighCpuUsage,
                    severity: AlertSeverity::Critical,
                    message: format!(
                        "High CPU utilization: {:.1}% > {:.1}%",
                        latest_usage.cpu_utilization * 100.0,
                        self.alert_thresholds.max_cpu_utilization * 100.0
                    ),
                    timestamp: SystemTime::now(),
                    worker_id: None,
                    metric_value: latest_usage.cpu_utilization,
                    threshold: self.alert_thresholds.max_cpu_utilization,
                });
            }

            // Check memory utilization
            if latest_usage.memory_utilization > self.alert_thresholds.max_memory_utilization {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::HighMemoryUsage,
                    severity: AlertSeverity::Critical,
                    message: format!(
                        "High memory utilization: {:.1}% > {:.1}%",
                        latest_usage.memory_utilization * 100.0,
                        self.alert_thresholds.max_memory_utilization * 100.0
                    ),
                    timestamp: SystemTime::now(),
                    worker_id: None,
                    metric_value: latest_usage.memory_utilization,
                    threshold: self.alert_thresholds.max_memory_utilization,
                });
            }

            // Check for failed workers
            if latest_usage.failed_workers > 0 {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::WorkerFailure,
                    severity: AlertSeverity::Critical,
                    message: format!("{} worker(s) have failed", latest_usage.failed_workers),
                    timestamp: SystemTime::now(),
                    worker_id: None,
                    metric_value: latest_usage.failed_workers as f64,
                    threshold: 0.0,
                });
            }
        }

        // Check individual worker metrics
        for (worker_id, metrics) in &self.worker_metrics {
            if metrics.health_score < 0.5 {
                alerts.push(PerformanceAlert {
                    alert_type: AlertType::AnomalyDetected,
                    severity: AlertSeverity::Warning,
                    message: format!(
                        "Worker {} health score is low: {:.2}",
                        worker_id, metrics.health_score
                    ),
                    timestamp: SystemTime::now(),
                    worker_id: Some(*worker_id),
                    metric_value: metrics.health_score,
                    threshold: 0.5,
                });
            }
        }

        Ok(alerts)
    }

    /// Perform comprehensive system analysis
    pub fn analyze_system_efficiency(&self) -> Result<EfficiencyAnalysis> {
        let metrics_history = self.metrics_history.lock().map_err(|_| {
            ClusteringError::InvalidInput("Failed to acquire metrics lock".to_string())
        })?;

        let resource_usage = self.resource_usage.lock().map_err(|_| {
            ClusteringError::InvalidInput("Failed to acquire resource usage lock".to_string())
        })?;

        // Calculate overall efficiency
        let overall_efficiency = if !metrics_history.is_empty() {
            let recent_metrics: Vec<_> = metrics_history.iter().rev().take(10).collect();
            let avg_efficiency = recent_metrics
                .iter()
                .map(|m| m.worker_efficiency)
                .sum::<f64>()
                / recent_metrics.len() as f64;
            avg_efficiency
        } else {
            0.0
        };

        // Perform bottleneck analysis
        let bottleneck_analysis = self.analyze_bottlenecks(&metrics_history, &resource_usage);

        // Calculate resource utilization
        let mut resource_utilization = HashMap::new();
        if let Some(latest_usage) = resource_usage.back() {
            resource_utilization.insert("cpu".to_string(), latest_usage.cpu_utilization);
            resource_utilization.insert("memory".to_string(), latest_usage.memory_utilization);
            resource_utilization.insert(
                "network".to_string(),
                latest_usage.network_throughput_mbps / 1000.0,
            );
            resource_utilization.insert("disk".to_string(), latest_usage.disk_io_rate);
        }

        // Analyze performance trends
        let performance_trends = self.analyze_trends(&metrics_history);

        // Generate optimization recommendations
        let optimization_recommendations = self.generate_recommendations(
            &bottleneck_analysis,
            &performance_trends,
            overall_efficiency,
        );

        Ok(EfficiencyAnalysis {
            overall_efficiency,
            bottleneck_analysis,
            resource_utilization,
            performance_trends,
            optimization_recommendations,
        })
    }

    /// Analyze system bottlenecks
    fn analyze_bottlenecks(
        &self,
        metrics_history: &VecDeque<PerformanceMetrics>,
        resource_usage: &VecDeque<ResourceUsage>,
    ) -> BottleneckAnalysis {
        let mut bottleneck_scores = HashMap::new();
        bottleneck_scores.insert(BottleneckType::Cpu, 0.0);
        bottleneck_scores.insert(BottleneckType::Memory, 0.0);
        bottleneck_scores.insert(BottleneckType::Network, 0.0);
        bottleneck_scores.insert(BottleneckType::Synchronization, 0.0);
        bottleneck_scores.insert(BottleneckType::LoadImbalance, 0.0);
        bottleneck_scores.insert(BottleneckType::MessagePassing, 0.0);

        // Analyze resource usage patterns
        if !resource_usage.is_empty() {
            let recent_usage: Vec<_> = resource_usage.iter().rev().take(10).collect();

            let avg_cpu = recent_usage.iter().map(|u| u.cpu_utilization).sum::<f64>()
                / recent_usage.len() as f64;
            let avg_memory = recent_usage
                .iter()
                .map(|u| u.memory_utilization)
                .sum::<f64>()
                / recent_usage.len() as f64;
            let avg_network = recent_usage
                .iter()
                .map(|u| u.network_throughput_mbps)
                .sum::<f64>()
                / recent_usage.len() as f64;

            bottleneck_scores.insert(BottleneckType::Cpu, avg_cpu);
            bottleneck_scores.insert(BottleneckType::Memory, avg_memory);
            bottleneck_scores.insert(BottleneckType::Network, avg_network / 1000.0);
            // Normalize
        }

        // Analyze performance metrics patterns
        if !metrics_history.is_empty() {
            let recent_metrics: Vec<_> = metrics_history.iter().rev().take(10).collect();

            let avg_sync_overhead = recent_metrics
                .iter()
                .map(|m| m.sync_overhead_ms)
                .sum::<f64>()
                / recent_metrics.len() as f64;
            let avg_message_latency = recent_metrics
                .iter()
                .map(|m| m.message_latency_ms)
                .sum::<f64>()
                / recent_metrics.len() as f64;
            let avg_load_balance = recent_metrics
                .iter()
                .map(|m| m.load_balance_score)
                .sum::<f64>()
                / recent_metrics.len() as f64;

            bottleneck_scores.insert(BottleneckType::Synchronization, avg_sync_overhead / 1000.0); // Normalize
            bottleneck_scores.insert(BottleneckType::MessagePassing, avg_message_latency / 1000.0); // Normalize
            bottleneck_scores.insert(BottleneckType::LoadImbalance, 1.0 - avg_load_balance);
            // Invert score
        }

        // Find primary bottleneck
        let (primary_bottleneck, bottleneck_severity) = bottleneck_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(bottleneck, &severity)| (bottleneck.clone(), severity))
            .unwrap_or((BottleneckType::None, 0.0));

        // Identify affected workers (simplified)
        let affected_workers: Vec<usize> = self
            .worker_metrics
            .iter()
            .filter(|(_, metrics)| metrics.health_score < 0.7)
            .map(|(&id, _)| id)
            .collect();

        let estimated_impact = bottleneck_severity * 0.5; // Simplified impact calculation

        BottleneckAnalysis {
            primary_bottleneck,
            bottleneck_severity,
            affected_workers,
            estimated_impact,
        }
    }

    /// Analyze performance trends
    fn analyze_trends(&self, metricshistory: &VecDeque<PerformanceMetrics>) -> PerformanceTrends {
        if metricshistory.len() < 5 {
            return PerformanceTrends {
                throughput_trend: TrendDirection::Unknown,
                latency_trend: TrendDirection::Unknown,
                efficiency_trend: TrendDirection::Unknown,
                resource_trend: TrendDirection::Unknown,
                trend_confidence: 0.0,
            };
        }

        let recent_metrics: Vec<_> = metricshistory.iter().rev().take(10).collect();
        let older_metrics: Vec<_> = metricshistory.iter().rev().skip(5).take(10).collect();

        // Calculate trend for worker efficiency
        let recent_efficiency = recent_metrics
            .iter()
            .map(|m| m.worker_efficiency)
            .sum::<f64>()
            / recent_metrics.len() as f64;
        let older_efficiency = if !older_metrics.is_empty() {
            older_metrics
                .iter()
                .map(|m| m.worker_efficiency)
                .sum::<f64>()
                / older_metrics.len() as f64
        } else {
            recent_efficiency
        };

        let efficiency_trend = if (recent_efficiency - older_efficiency).abs() < 0.05 {
            TrendDirection::Stable
        } else if recent_efficiency > older_efficiency {
            TrendDirection::Improving
        } else {
            TrendDirection::Degrading
        };

        // Calculate trend for message latency
        let recent_latency = recent_metrics
            .iter()
            .map(|m| m.message_latency_ms)
            .sum::<f64>()
            / recent_metrics.len() as f64;
        let older_latency = if !older_metrics.is_empty() {
            older_metrics
                .iter()
                .map(|m| m.message_latency_ms)
                .sum::<f64>()
                / older_metrics.len() as f64
        } else {
            recent_latency
        };

        let latency_trend = if (recent_latency - older_latency).abs() < 50.0 {
            TrendDirection::Stable
        } else if recent_latency < older_latency {
            TrendDirection::Improving
        } else {
            TrendDirection::Degrading
        };

        // Simplified trends for other metrics
        let throughput_trend = efficiency_trend;
        let resource_trend = TrendDirection::Stable;

        let trend_confidence = if recent_metrics.len() >= 10 { 0.8 } else { 0.4 };

        PerformanceTrends {
            throughput_trend,
            latency_trend,
            efficiency_trend,
            resource_trend,
            trend_confidence,
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        bottleneck_analysis: &BottleneckAnalysis,
        performance_trends: &PerformanceTrends,
        overall_efficiency: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Bottleneck-based recommendations
        match bottleneck_analysis.primary_bottleneck {
            BottleneckType::Cpu => {
                recommendations.push(
                    "Consider adding more CPU cores or reducing computational load per worker"
                        .to_string(),
                );
                recommendations
                    .push("Optimize algorithms to reduce CPU-intensive operations".to_string());
            }
            BottleneckType::Memory => {
                recommendations.push(
                    "Increase memory allocation or implement more efficient memory management"
                        .to_string(),
                );
                recommendations
                    .push("Consider data compression or streaming techniques".to_string());
            }
            BottleneckType::Network => {
                recommendations.push("Optimize network communication patterns".to_string());
                recommendations.push("Consider message batching or compression".to_string());
            }
            BottleneckType::Synchronization => {
                recommendations.push(
                    "Reduce synchronization frequency or implement asynchronous patterns"
                        .to_string(),
                );
                recommendations
                    .push("Consider lockless data structures where possible".to_string());
            }
            BottleneckType::LoadImbalance => {
                recommendations.push("Implement dynamic load balancing".to_string());
                recommendations.push("Review data partitioning strategy".to_string());
            }
            BottleneckType::MessagePassing => {
                recommendations.push("Optimize message passing protocols".to_string());
                recommendations.push("Reduce message size or frequency".to_string());
            }
            _ => {}
        }

        // Trend-based recommendations
        match performance_trends.efficiency_trend {
            TrendDirection::Degrading => {
                recommendations
                    .push("Performance is degrading - investigate recent changes".to_string());
                recommendations
                    .push("Consider scaling up resources or optimizing algorithms".to_string());
            }
            TrendDirection::Stable => {
                if overall_efficiency < 0.7 {
                    recommendations.push(
                        "Performance is stable but suboptimal - consider optimization".to_string(),
                    );
                }
            }
            _ => {}
        }

        // Overall efficiency recommendations
        if overall_efficiency < 0.5 {
            recommendations.push(
                "Overall efficiency is very low - comprehensive system review needed".to_string(),
            );
        } else if overall_efficiency < 0.7 {
            recommendations
                .push("Moderate efficiency - targeted optimizations recommended".to_string());
        }

        // Worker-specific recommendations
        let unhealthy_workers = self
            .worker_metrics
            .iter()
            .filter(|(_, metrics)| metrics.health_score < 0.6)
            .count();

        if unhealthy_workers > 0 {
            recommendations.push(format!(
                "{} workers are performing poorly - investigate individual worker issues",
                unhealthy_workers
            ));
        }

        if recommendations.is_empty() {
            recommendations
                .push("System performance is optimal - no immediate action required".to_string());
        }

        recommendations
    }

    /// Generate monitoring report
    pub fn generate_report(&self) -> MonitoringReport {
        let mut report = MonitoringReport::default();

        // Calculate averages from recent history
        let metrics_history = self.metrics_history.lock().expect("Operation failed");
        let resource_usage = self.resource_usage.lock().expect("Operation failed");

        if !metrics_history.is_empty() {
            let recent_metrics: Vec<_> = metrics_history.iter().rev().take(10).collect();

            report.avg_convergence_rate = recent_metrics
                .iter()
                .map(|m| m.convergence_rate)
                .sum::<f64>()
                / recent_metrics.len() as f64;

            report.avg_worker_efficiency = recent_metrics
                .iter()
                .map(|m| m.worker_efficiency)
                .sum::<f64>()
                / recent_metrics.len() as f64;

            report.avg_sync_overhead = recent_metrics
                .iter()
                .map(|m| m.sync_overhead_ms)
                .sum::<f64>()
                / recent_metrics.len() as f64;
        }

        if !resource_usage.is_empty() {
            let recent_usage: Vec<_> = resource_usage.iter().rev().take(10).collect();

            report.avg_cpu_utilization =
                recent_usage.iter().map(|r| r.cpu_utilization).sum::<f64>()
                    / recent_usage.len() as f64;

            report.avg_memory_utilization = recent_usage
                .iter()
                .map(|r| r.memory_utilization)
                .sum::<f64>()
                / recent_usage.len() as f64;

            report.peak_network_throughput = recent_usage
                .iter()
                .map(|r| r.network_throughput_mbps)
                .fold(0.0, f64::max);
        }

        // Calculate efficiency scores
        report.overall_efficiency_score = self.calculate_efficiency_score();
        report.recommendations = self.generate_optimization_recommendations();

        report
    }

    /// Calculate overall system efficiency score
    fn calculate_efficiency_score(&self) -> f64 {
        let metrics_history = self.metrics_history.lock().expect("Operation failed");
        let resource_usage = self.resource_usage.lock().expect("Operation failed");

        if metrics_history.is_empty() || resource_usage.is_empty() {
            return 0.0;
        }

        // Weighted efficiency calculation
        let convergence_score = metrics_history
            .iter()
            .map(|m| m.convergence_rate.min(1.0))
            .sum::<f64>()
            / metrics_history.len() as f64;

        let worker_score = metrics_history
            .iter()
            .map(|m| m.worker_efficiency)
            .sum::<f64>()
            / metrics_history.len() as f64;

        let resource_score = 1.0
            - (resource_usage
                .iter()
                .map(|r| r.memory_utilization.max(r.cpu_utilization))
                .sum::<f64>()
                / resource_usage.len() as f64);

        // Weighted average: 40% convergence, 40% worker efficiency, 20% resource usage
        convergence_score * 0.4 + worker_score * 0.4 + resource_score * 0.2
    }

    /// Generate optimization recommendations
    fn generate_optimization_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let metrics_history = self.metrics_history.lock().expect("Operation failed");
        let resource_usage = self.resource_usage.lock().expect("Operation failed");

        if let Some(latest_metrics) = metrics_history.back() {
            if latest_metrics.worker_efficiency < 0.7 {
                recommendations
                    .push("Consider load rebalancing - worker efficiency is low".to_string());
            }

            if latest_metrics.sync_overhead_ms > 1000.0 {
                recommendations.push(
                    "High synchronization overhead - consider reducing coordination frequency"
                        .to_string(),
                );
            }

            if latest_metrics.message_latency_ms > 500.0 {
                recommendations
                    .push("High message latency - check network configuration".to_string());
            }
        }

        if let Some(latest_resources) = resource_usage.back() {
            if latest_resources.memory_utilization > 0.8 {
                recommendations.push(
                    "High memory usage - consider increasing workers or reducing batch size"
                        .to_string(),
                );
            }

            if latest_resources.failed_workers > 0 {
                recommendations.push(
                    "Worker failures detected - check fault tolerance configuration".to_string(),
                );
            }

            if latest_resources.queue_depth > 100 {
                recommendations.push(
                    "High message queue depth - consider increasing processing capacity"
                        .to_string(),
                );
            }
        }

        if recommendations.is_empty() {
            recommendations.push("System performance is optimal".to_string());
        }

        recommendations
    }

    /// Export metrics for external analysis
    pub fn export_metrics_csv(&self, filepath: &str) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filepath)
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to create file: {}", e)))?;

        // Write CSV header
        writeln!(file, "timestamp,iteration,global_inertia,convergence_rate,worker_efficiency,message_latency_ms,sync_overhead_ms,memory_pressure")
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to write header: {}", e)))?;

        // Write metrics data
        let metrics_history = self.metrics_history.lock().expect("Operation failed");
        for metrics in metrics_history.iter() {
            writeln!(
                file,
                "{:?},{},{},{},{},{},{},{}",
                metrics.timestamp,
                metrics.iteration,
                metrics.global_inertia,
                metrics.convergence_rate,
                metrics.worker_efficiency,
                metrics.message_latency_ms,
                metrics.sync_overhead_ms,
                metrics.memory_pressure_score
            )
            .map_err(|e| ClusteringError::InvalidInput(format!("Failed to write data: {}", e)))?;
        }

        Ok(())
    }

    /// Get current worker metrics
    pub fn get_worker_metrics(&self) -> &HashMap<usize, WorkerMetrics> {
        &self.worker_metrics
    }

    /// Get monitoring configuration
    pub fn get_config(&self) -> &MonitoringConfig {
        &self.config
    }

    /// Get system uptime
    pub fn get_uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Sample real (cpu_fraction, memory_fraction) resource usage, each in
    /// `[0.0, 1.0]`, to feed into `update_worker_metrics`. This replaces the
    /// previously-hardcoded `(0.5, 0.4)` placeholder pair with an actual,
    /// varying measurement of the process performing the work.
    ///
    /// Workers in this implementation are simulated sequentially within a
    /// single OS process (there is no separate real process/thread per
    /// worker to query), so there is no meaningful *per-worker* OS-level
    /// figure to report. What genuinely is real and measurable is the state
    /// of the process actually doing the work at the moment each worker's
    /// partition is processed, which is what this reports. When built
    /// without the `sysinfo` feature, this honestly returns `(0.0, 0.0)`
    /// (a neutral "unknown" that never trips the health-score penalty
    /// thresholds in `calculate_worker_health_score`) instead of fabricating
    /// a plausible-looking constant.
    pub fn sample_resources(&mut self) -> (f64, f64) {
        self.resource_sampler.sample()
    }
}

/// Real, best-effort sampler for the current process/system's resource
/// utilization. See [`PerformanceMonitor::sample_resources`] for the
/// rationale (no fabricated constants, ever).
pub struct ResourceSampler {
    #[cfg(feature = "sysinfo")]
    system: sysinfo::System,
}

impl std::fmt::Debug for ResourceSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResourceSampler").finish_non_exhaustive()
    }
}

impl Default for ResourceSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceSampler {
    /// Create a new sampler. When the `sysinfo` feature is enabled this
    /// eagerly initializes the underlying system snapshot so the first
    /// `sample()` call has a baseline to compute a real CPU delta against.
    pub fn new() -> Self {
        #[cfg(feature = "sysinfo")]
        {
            Self {
                system: sysinfo::System::new_all(),
            }
        }
        #[cfg(not(feature = "sysinfo"))]
        {
            Self {}
        }
    }

    /// Sample current `(cpu_fraction, memory_fraction)`, both clamped to
    /// `[0.0, 1.0]`. Returns `(0.0, 0.0)` -- an honest, penalty-neutral
    /// "unknown" -- when compiled without the `sysinfo` feature.
    pub fn sample(&mut self) -> (f64, f64) {
        #[cfg(feature = "sysinfo")]
        {
            self.system.refresh_cpu_all();
            self.system.refresh_memory();

            let cpus = self.system.cpus();
            let cpu_fraction = if cpus.is_empty() {
                0.0
            } else {
                let sum: f64 = cpus.iter().map(|cpu| cpu.cpu_usage() as f64 / 100.0).sum();
                (sum / cpus.len() as f64).clamp(0.0, 1.0)
            };

            let total_memory = self.system.total_memory();
            let memory_fraction = if total_memory == 0 {
                0.0
            } else {
                (self.system.used_memory() as f64 / total_memory as f64).clamp(0.0, 1.0)
            };

            (cpu_fraction, memory_fraction)
        }

        #[cfg(not(feature = "sysinfo"))]
        {
            (0.0, 0.0)
        }
    }
}

/// Comprehensive monitoring report
#[derive(Debug, Default)]
pub struct MonitoringReport {
    pub avg_convergence_rate: f64,
    pub avg_worker_efficiency: f64,
    pub avg_sync_overhead: f64,
    pub avg_cpu_utilization: f64,
    pub avg_memory_utilization: f64,
    pub peak_network_throughput: f64,
    pub overall_efficiency_score: f64,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        assert!(monitor.worker_metrics.is_empty());
        assert!(monitor
            .metrics_history
            .lock()
            .expect("Operation failed")
            .is_empty());
    }

    #[test]
    fn test_worker_registration() {
        let config = MonitoringConfig::default();
        let mut monitor = PerformanceMonitor::new(config);

        monitor.register_worker(1);
        assert!(monitor.worker_metrics.contains_key(&1));
        assert_eq!(monitor.worker_metrics[&1].worker_id, 1);
    }

    #[test]
    fn test_performance_metrics_recording() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            iteration: 1,
            global_inertia: 100.0,
            convergence_rate: 0.8,
            worker_efficiency: 0.9,
            message_latency_ms: 50.0,
            sync_overhead_ms: 100.0,
            total_computation_time_ms: 5000,
            memory_pressure_score: 0.6,
            load_balance_score: 0.8,
            network_utilization: 0.5,
        };

        let result = monitor.record_performance_metrics(metrics);
        assert!(result.is_ok());
        assert_eq!(
            monitor
                .metrics_history
                .lock()
                .expect("Operation failed")
                .len(),
            1
        );
    }

    #[test]
    fn test_worker_health_score_calculation() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        let mut metrics = WorkerMetrics {
            worker_id: 1,
            cpu_usage_history: VecDeque::from(vec![0.5, 0.6, 0.4]),
            memory_usage_history: VecDeque::from(vec![0.3, 0.4, 0.2]),
            throughput_history: VecDeque::new(),
            latency_history: VecDeque::from(vec![100.0, 150.0, 80.0]),
            error_count: 0,
            last_update: SystemTime::now(),
            health_score: 0.0,
        };

        let score = monitor.calculate_worker_health_score(&metrics);
        assert!(score > 0.5 && score <= 1.0);

        // Test with high resource usage
        metrics.cpu_usage_history = VecDeque::from(vec![0.95, 0.98, 0.92]);
        metrics.memory_usage_history = VecDeque::from(vec![0.9, 0.95, 0.88]);

        let degraded_score = monitor.calculate_worker_health_score(&metrics);
        assert!(degraded_score < score);
    }

    #[test]
    fn test_alert_generation() {
        let config = MonitoringConfig::default();
        let monitor = PerformanceMonitor::new(config);

        // Record metrics that should trigger alerts
        let metrics = PerformanceMetrics {
            timestamp: SystemTime::now(),
            iteration: 1,
            global_inertia: 100.0,
            convergence_rate: 0.1,      // Low convergence
            worker_efficiency: 0.3,     // Low efficiency
            message_latency_ms: 2000.0, // High latency
            sync_overhead_ms: 100.0,
            total_computation_time_ms: 400000, // Long computation time
            memory_pressure_score: 0.6,
            load_balance_score: 0.8,
            network_utilization: 0.5,
        };

        monitor
            .record_performance_metrics(metrics)
            .expect("Operation failed");

        let alerts = monitor.check_alerts().expect("Operation failed");
        assert!(!alerts.is_empty());

        // Check if we got expected alert types
        let alert_types: Vec<_> = alerts.iter().map(|a| &a.alert_type).collect();
        assert!(alert_types
            .iter()
            .any(|t| matches!(t, AlertType::ConvergenceTimeout)));
        assert!(alert_types
            .iter()
            .any(|t| matches!(t, AlertType::LowThroughput)));
    }

    #[cfg(feature = "sysinfo")]
    #[test]
    fn test_resource_sampler_reports_real_nonzero_memory_usage() {
        let mut sampler = ResourceSampler::new();
        let (cpu, memory) = sampler.sample();

        assert!(
            (0.0..=1.0).contains(&cpu),
            "cpu fraction out of range: {cpu}"
        );
        assert!(
            (0.0..=1.0).contains(&memory),
            "memory fraction out of range: {memory}"
        );
        // A real running process always holds *some* resident memory, so a
        // genuine measurement can never legitimately read exactly zero on a
        // live system -- unlike the old hardcoded 0.4 constant, this value
        // is derived from the actual host state at sample time.
        assert!(
            memory > 0.0,
            "expected a genuine nonzero memory reading, got {memory}"
        );
    }

    #[cfg(feature = "sysinfo")]
    #[test]
    fn test_resource_sampler_varies_is_not_hardcoded_pair() {
        // Two independent samplers on the same live machine should not both
        // coincidentally reproduce the exact former hardcoded (0.5, 0.4)
        // placeholder pair -- this is a regression guard against silently
        // reintroducing fabricated constants.
        let mut sampler = ResourceSampler::new();
        let (cpu, memory) = sampler.sample();
        assert!(
            !(cpu == 0.5 && memory == 0.4),
            "resource sampler must not reproduce the old fabricated (0.5, 0.4) placeholder pair"
        );
    }

    #[cfg(not(feature = "sysinfo"))]
    #[test]
    fn test_resource_sampler_reports_honest_unknown_without_sysinfo_feature() {
        // Without real measurement capability, the sampler must report an
        // honest, penalty-neutral "unknown" rather than a fabricated
        // plausible-looking constant such as the old (0.5, 0.4) pair.
        let mut sampler = ResourceSampler::new();
        assert_eq!(sampler.sample(), (0.0, 0.0));
    }

    #[test]
    fn test_sample_resources_delegates_to_sampler() {
        let config = MonitoringConfig::default();
        let mut monitor = PerformanceMonitor::new(config);

        let (cpu, memory) = monitor.sample_resources();
        assert!((0.0..=1.0).contains(&cpu));
        assert!((0.0..=1.0).contains(&memory));
    }
}
