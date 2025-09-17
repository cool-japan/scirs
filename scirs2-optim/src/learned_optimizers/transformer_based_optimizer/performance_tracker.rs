//! Performance tracking for transformer-based optimizer

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use crate::error::Result;
use super::config::PerformanceConfig;
use super::meta_learning::MetaLearningResult;
use super::TrainingMetrics;

/// Performance metrics for transformer optimizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Training metrics
    pub training_metrics: TrainingMetricsCollection,

    /// Inference metrics
    pub inference_metrics: InferenceMetricsCollection,

    /// Memory usage metrics
    pub memory_metrics: MemoryMetricsCollection,

    /// Computation time metrics
    pub timing_metrics: TimingMetricsCollection,

    /// Model quality metrics
    pub quality_metrics: QualityMetricsCollection,

    /// Resource utilization metrics
    pub resource_metrics: ResourceMetricsCollection,

    /// Optimization metrics
    pub optimization_metrics: OptimizationMetricsCollection,
}

/// Transformer performance tracker
pub struct TransformerPerformanceTracker<T: Float> {
    /// Configuration
    config: PerformanceConfig,

    /// Performance metrics
    metrics: PerformanceMetrics,

    /// Loss history
    loss_history: VecDeque<f64>,

    /// Training step timings
    step_timings: VecDeque<Duration>,

    /// Meta-learning results history
    meta_results_history: VecDeque<MetaLearningResult<T>>,

    /// Training metrics history
    training_history: VecDeque<TrainingMetrics>,

    /// Memory usage samples
    memory_samples: VecDeque<MemorySample>,

    /// Performance baselines
    baselines: PerformanceBaselines,

    /// Alert thresholds
    alert_thresholds: AlertThresholds,

    /// Performance trends
    trends: PerformanceTrends,

    /// Profiling data
    profiling_data: ProfilingData,

    /// Start time for tracking session
    session_start: Instant,
}

impl<T: Float> TransformerPerformanceTracker<T> {
    /// Create new performance tracker
    pub fn new() -> Self {
        let config = PerformanceConfig::default();

        Self {
            config,
            metrics: PerformanceMetrics::new(),
            loss_history: VecDeque::new(),
            step_timings: VecDeque::new(),
            meta_results_history: VecDeque::new(),
            training_history: VecDeque::new(),
            memory_samples: VecDeque::new(),
            baselines: PerformanceBaselines::new(),
            alert_thresholds: AlertThresholds::default(),
            trends: PerformanceTrends::new(),
            profiling_data: ProfilingData::new(),
            session_start: Instant::now(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: PerformanceConfig) -> Self {
        let mut tracker = Self::new();
        tracker.config = config;
        tracker
    }

    /// Record optimization step performance
    pub fn record_optimization_step(&mut self, duration: Duration, update: &Array1<T>) {
        // Record timing
        self.step_timings.push_back(duration);
        if self.step_timings.len() > self.config.max_history_size {
            self.step_timings.pop_front();
        }

        // Update timing metrics
        self.metrics.timing_metrics.record_step_time(duration);

        // Analyze update quality
        let update_norm = self.compute_array_norm(update);
        self.metrics.optimization_metrics.record_update_norm(update_norm);

        // Update trends
        self.trends.update_step_timing(duration);

        // Check for performance alerts
        self.check_performance_alerts(duration, update_norm);
    }

    /// Record training epoch performance
    pub fn record_training_epoch(&mut self, metrics: TrainingMetrics) {
        self.training_history.push_back(metrics.clone());
        if self.training_history.len() > self.config.max_history_size {
            self.training_history.pop_front();
        }

        // Update training metrics
        self.metrics.training_metrics.record_epoch(
            metrics.loss,
            metrics.training_time,
            metrics.convergence_rate,
        );

        // Update trends
        self.trends.update_training_loss(metrics.loss);
        self.trends.update_convergence_rate(metrics.convergence_rate);

        // Check for convergence alerts
        self.check_convergence_alerts(&metrics);
    }

    /// Record meta-learning step performance
    pub fn record_meta_step(&mut self, result: MetaLearningResult<T>) {
        self.meta_results_history.push_back(result.clone());
        if self.meta_results_history.len() > 100 {
            self.meta_results_history.pop_front();
        }

        // Update meta-learning metrics
        self.metrics.training_metrics.record_meta_step(
            result.meta_loss,
            result.computation_time,
            result.task_adaptations.len(),
        );

        // Update trends
        self.trends.update_meta_loss(result.meta_loss);
        self.trends.update_adaptation_efficiency(result.task_adaptations.len() as f64);
    }

    /// Record inference performance
    pub fn record_inference(&mut self, input_size: usize, duration: Duration, output_quality: f64) {
        self.metrics.inference_metrics.record_inference(
            input_size,
            duration,
            output_quality,
        );

        // Update throughput calculations
        let throughput = input_size as f64 / duration.as_secs_f64();
        self.metrics.inference_metrics.update_throughput(throughput);

        // Update trends
        self.trends.update_inference_latency(duration);
        self.trends.update_inference_quality(output_quality);
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, usage: MemoryUsage) {
        let sample = MemorySample {
            timestamp: std::time::SystemTime::now(),
            usage,
        };

        self.memory_samples.push_back(sample.clone());
        if self.memory_samples.len() > self.config.max_history_size {
            self.memory_samples.pop_front();
        }

        // Update memory metrics
        self.metrics.memory_metrics.record_usage(usage);

        // Update trends
        self.trends.update_memory_usage(usage.total_memory);

        // Check for memory alerts
        self.check_memory_alerts(&usage);
    }

    /// Record loss value
    pub fn record_loss(&mut self, loss: f64) {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > self.config.max_history_size {
            self.loss_history.pop_front();
        }

        // Update quality metrics
        self.metrics.quality_metrics.record_loss(loss);

        // Update trends
        self.trends.update_loss(loss);
    }

    /// Profile operation performance
    pub fn profile_operation<F, R>(&mut self, operation_name: &str, operation: F) -> Result<R>
    where
        F: FnOnce() -> Result<R>,
    {
        let start_time = Instant::now();
        let result = operation();
        let duration = start_time.elapsed();

        // Record profiling data
        self.profiling_data.record_operation(operation_name.to_string(), duration);

        // Update timing metrics
        self.metrics.timing_metrics.record_operation_time(operation_name, duration);

        result
    }

    /// Generate performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let session_duration = self.session_start.elapsed();

        PerformanceReport {
            session_duration,
            total_optimization_steps: self.step_timings.len(),
            total_training_epochs: self.training_history.len(),
            total_meta_steps: self.meta_results_history.len(),

            // Average performance metrics
            average_step_time: self.calculate_average_step_time(),
            average_loss: self.calculate_average_loss(),
            current_convergence_rate: self.calculate_current_convergence_rate(),

            // Memory statistics
            peak_memory_usage: self.calculate_peak_memory_usage(),
            average_memory_usage: self.calculate_average_memory_usage(),

            // Quality metrics
            best_loss: self.calculate_best_loss(),
            loss_improvement: self.calculate_loss_improvement(),

            // Performance trends
            loss_trend: self.trends.get_loss_trend(),
            convergence_trend: self.trends.get_convergence_trend(),
            memory_trend: self.trends.get_memory_trend(),

            // Alert summary
            performance_alerts: self.get_recent_alerts(),

            // Resource utilization
            cpu_utilization: self.metrics.resource_metrics.get_average_cpu_usage(),
            memory_utilization: self.metrics.resource_metrics.get_average_memory_usage(),

            // Model quality assessment
            quality_score: self.calculate_overall_quality_score(),

            // Recommendations
            recommendations: self.generate_recommendations(),
        }
    }

    /// Get loss history
    pub fn get_loss_history(&self) -> &VecDeque<f64> {
        &self.loss_history
    }

    /// Get current performance metrics
    pub fn get_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Get performance trends
    pub fn get_trends(&self) -> &PerformanceTrends {
        &self.trends
    }

    /// Set performance baselines
    pub fn set_baselines(&mut self, baselines: PerformanceBaselines) {
        self.baselines = baselines;
    }

    /// Reset all performance tracking
    pub fn reset(&mut self) {
        self.metrics = PerformanceMetrics::new();
        self.loss_history.clear();
        self.step_timings.clear();
        self.meta_results_history.clear();
        self.training_history.clear();
        self.memory_samples.clear();
        self.trends.reset();
        self.profiling_data.reset();
        self.session_start = Instant::now();
    }

    /// Helper methods
    fn compute_array_norm(&self, array: &Array1<T>) -> f64 {
        let sum_squares: T = array.iter().map(|&x| x * x).fold(T::zero(), |acc, x| acc + x);
        sum_squares.sqrt().to_f64().unwrap_or(0.0)
    }

    fn calculate_average_step_time(&self) -> Duration {
        if self.step_timings.is_empty() {
            Duration::new(0, 0)
        } else {
            let total: Duration = self.step_timings.iter().sum();
            total / self.step_timings.len() as u32
        }
    }

    fn calculate_average_loss(&self) -> f64 {
        if self.loss_history.is_empty() {
            0.0
        } else {
            self.loss_history.iter().sum::<f64>() / self.loss_history.len() as f64
        }
    }

    fn calculate_current_convergence_rate(&self) -> f64 {
        if let Some(recent_training) = self.training_history.back() {
            recent_training.convergence_rate
        } else {
            0.0
        }
    }

    fn calculate_peak_memory_usage(&self) -> usize {
        self.memory_samples
            .iter()
            .map(|sample| sample.usage.total_memory)
            .max()
            .unwrap_or(0)
    }

    fn calculate_average_memory_usage(&self) -> f64 {
        if self.memory_samples.is_empty() {
            0.0
        } else {
            let total: usize = self.memory_samples.iter().map(|s| s.usage.total_memory).sum();
            total as f64 / self.memory_samples.len() as f64
        }
    }

    fn calculate_best_loss(&self) -> f64 {
        self.loss_history
            .iter()
            .fold(f64::INFINITY, |min, &loss| min.min(loss))
    }

    fn calculate_loss_improvement(&self) -> f64 {
        if self.loss_history.len() < 2 {
            return 0.0;
        }

        let initial_loss = self.loss_history[0];
        let final_loss = *self.loss_history.back().unwrap();

        if initial_loss > 0.0 {
            (initial_loss - final_loss) / initial_loss
        } else {
            0.0
        }
    }

    fn calculate_overall_quality_score(&self) -> f64 {
        // Weighted combination of various quality metrics
        let loss_score = 1.0 - (self.calculate_average_loss() / 10.0).min(1.0);
        let convergence_score = self.calculate_current_convergence_rate();
        let stability_score = 1.0 - self.trends.get_loss_volatility().min(1.0);

        (loss_score * 0.4 + convergence_score * 0.3 + stability_score * 0.3).max(0.0).min(1.0)
    }

    fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Loss-based recommendations
        if self.trends.get_loss_trend() > 0.0 {
            recommendations.push("Loss is increasing. Consider reducing learning rate.".to_string());
        }

        // Memory-based recommendations
        if self.calculate_peak_memory_usage() > 1024 * 1024 * 1024 { // 1GB
            recommendations.push("High memory usage detected. Consider enabling memory compression.".to_string());
        }

        // Performance-based recommendations
        if self.calculate_average_step_time() > Duration::from_millis(100) {
            recommendations.push("Slow optimization steps. Consider reducing model size or batch size.".to_string());
        }

        // Convergence-based recommendations
        if self.calculate_current_convergence_rate() < 0.1 {
            recommendations.push("Low convergence rate. Consider adjusting meta-learning parameters.".to_string());
        }

        recommendations
    }

    fn check_performance_alerts(&mut self, duration: Duration, update_norm: f64) {
        // Check for slow steps
        if duration > self.alert_thresholds.max_step_time {
            self.record_alert(AlertType::SlowStep, format!("Step took {:?}", duration));
        }

        // Check for gradient explosion
        if update_norm > self.alert_thresholds.max_gradient_norm {
            self.record_alert(AlertType::GradientExplosion, format!("Update norm: {:.6}", update_norm));
        }

        // Check for gradient vanishing
        if update_norm < self.alert_thresholds.min_gradient_norm {
            self.record_alert(AlertType::GradientVanishing, format!("Update norm: {:.6}", update_norm));
        }
    }

    fn check_convergence_alerts(&mut self, metrics: &TrainingMetrics) {
        // Check for training stagnation
        if metrics.convergence_rate < self.alert_thresholds.min_convergence_rate {
            self.record_alert(AlertType::TrainingStagnation,
                format!("Convergence rate: {:.6}", metrics.convergence_rate));
        }

        // Check for loss increase
        if let Some(previous) = self.training_history.iter().rev().nth(1) {
            if metrics.loss > previous.loss * 1.1 { // 10% increase
                self.record_alert(AlertType::LossIncrease,
                    format!("Loss increased from {:.6} to {:.6}", previous.loss, metrics.loss));
            }
        }
    }

    fn check_memory_alerts(&mut self, usage: &MemoryUsage) {
        // Check for high memory usage
        if usage.total_memory > self.alert_thresholds.max_memory_usage {
            self.record_alert(AlertType::HighMemoryUsage,
                format!("Memory usage: {} bytes", usage.total_memory));
        }

        // Check for memory leaks
        if let Some(previous) = self.memory_samples.back() {
            let memory_increase = usage.total_memory.saturating_sub(previous.usage.total_memory);
            if memory_increase > 1024 * 1024 * 100 { // 100MB increase
                self.record_alert(AlertType::PossibleMemoryLeak,
                    format!("Memory increased by {} bytes", memory_increase));
            }
        }
    }

    fn record_alert(&mut self, alert_type: AlertType, message: String) {
        let alert = PerformanceAlert {
            alert_type,
            message,
            timestamp: std::time::SystemTime::now(),
        };

        self.metrics.quality_metrics.record_alert(alert);
    }

    fn get_recent_alerts(&self) -> Vec<PerformanceAlert> {
        self.metrics.quality_metrics.get_recent_alerts(10)
    }
}

/// Performance metrics collections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetricsCollection {
    pub total_epochs: usize,
    pub total_training_time: Duration,
    pub average_loss: f64,
    pub best_loss: f64,
    pub convergence_rate: f64,
    pub meta_steps: usize,
    pub task_adaptations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetricsCollection {
    pub total_inferences: usize,
    pub average_latency: Duration,
    pub peak_throughput: f64,
    pub average_quality: f64,
    pub cache_hit_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetricsCollection {
    pub peak_usage: usize,
    pub average_usage: f64,
    pub total_allocations: usize,
    pub compression_ratio: f64,
    pub fragmentation_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMetricsCollection {
    pub average_step_time: Duration,
    pub total_computation_time: Duration,
    pub operation_timings: HashMap<String, Duration>,
    pub profiling_overhead: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetricsCollection {
    pub loss_statistics: LossStatistics,
    pub convergence_statistics: ConvergenceStatistics,
    pub stability_metrics: StabilityMetrics,
    pub performance_alerts: VecDeque<PerformanceAlert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetricsCollection {
    pub cpu_usage_history: VecDeque<f64>,
    pub memory_usage_history: VecDeque<f64>,
    pub disk_io_metrics: DiskIOMetrics,
    pub network_metrics: NetworkMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetricsCollection {
    pub update_norm_history: VecDeque<f64>,
    pub parameter_change_rate: f64,
    pub optimization_efficiency: f64,
    pub adaptive_learning_metrics: AdaptiveLearningMetrics,
}

/// Supporting data structures
#[derive(Debug, Clone)]
pub struct MemorySample {
    pub timestamp: std::time::SystemTime,
    pub usage: MemoryUsage,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MemoryUsage {
    pub total_memory: usize,
    pub model_memory: usize,
    pub cache_memory: usize,
    pub temporary_memory: usize,
}

#[derive(Debug, Clone)]
pub struct PerformanceBaselines {
    pub baseline_loss: f64,
    pub baseline_step_time: Duration,
    pub baseline_memory_usage: usize,
    pub baseline_convergence_rate: f64,
}

impl PerformanceBaselines {
    pub fn new() -> Self {
        Self {
            baseline_loss: f64::INFINITY,
            baseline_step_time: Duration::new(0, 0),
            baseline_memory_usage: 0,
            baseline_convergence_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub max_step_time: Duration,
    pub min_convergence_rate: f64,
    pub max_memory_usage: usize,
    pub max_gradient_norm: f64,
    pub min_gradient_norm: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_step_time: Duration::from_secs(1),
            min_convergence_rate: 0.01,
            max_memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
            max_gradient_norm: 10.0,
            min_gradient_norm: 1e-8,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    loss_trend: TrendAnalyzer,
    convergence_trend: TrendAnalyzer,
    memory_trend: TrendAnalyzer,
    timing_trend: TrendAnalyzer,
}

impl PerformanceTrends {
    pub fn new() -> Self {
        Self {
            loss_trend: TrendAnalyzer::new(100),
            convergence_trend: TrendAnalyzer::new(50),
            memory_trend: TrendAnalyzer::new(100),
            timing_trend: TrendAnalyzer::new(100),
        }
    }

    pub fn update_loss(&mut self, loss: f64) {
        self.loss_trend.add_sample(loss);
    }

    pub fn update_training_loss(&mut self, loss: f64) {
        self.loss_trend.add_sample(loss);
    }

    pub fn update_meta_loss(&mut self, loss: f64) {
        self.loss_trend.add_sample(loss);
    }

    pub fn update_convergence_rate(&mut self, rate: f64) {
        self.convergence_trend.add_sample(rate);
    }

    pub fn update_adaptation_efficiency(&mut self, efficiency: f64) {
        self.convergence_trend.add_sample(efficiency);
    }

    pub fn update_memory_usage(&mut self, usage: usize) {
        self.memory_trend.add_sample(usage as f64);
    }

    pub fn update_step_timing(&mut self, duration: Duration) {
        self.timing_trend.add_sample(duration.as_secs_f64());
    }

    pub fn update_inference_latency(&mut self, duration: Duration) {
        self.timing_trend.add_sample(duration.as_secs_f64());
    }

    pub fn update_inference_quality(&mut self, _quality: f64) {
        // Could be used for quality trend analysis
    }

    pub fn get_loss_trend(&self) -> f64 {
        self.loss_trend.get_trend()
    }

    pub fn get_convergence_trend(&self) -> f64 {
        self.convergence_trend.get_trend()
    }

    pub fn get_memory_trend(&self) -> f64 {
        self.memory_trend.get_trend()
    }

    pub fn get_loss_volatility(&self) -> f64 {
        self.loss_trend.get_volatility()
    }

    pub fn reset(&mut self) {
        self.loss_trend.reset();
        self.convergence_trend.reset();
        self.memory_trend.reset();
        self.timing_trend.reset();
    }
}

#[derive(Debug, Clone)]
pub struct TrendAnalyzer {
    samples: VecDeque<f64>,
    max_samples: usize,
}

impl TrendAnalyzer {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            max_samples,
        }
    }

    pub fn add_sample(&mut self, value: f64) {
        self.samples.push_back(value);
        if self.samples.len() > self.max_samples {
            self.samples.pop_front();
        }
    }

    pub fn get_trend(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }

        // Simple linear trend calculation
        let n = self.samples.len() as f64;
        let x_sum = (0..self.samples.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = self.samples.iter().sum::<f64>();
        let xy_sum = self.samples.iter().enumerate()
            .map(|(i, &y)| i as f64 * y).sum::<f64>();
        let x_sq_sum = (0..self.samples.len()).map(|i| (i as f64).powi(2)).sum::<f64>();

        let denominator = n * x_sq_sum - x_sum * x_sum;
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            (n * xy_sum - x_sum * y_sum) / denominator
        }
    }

    pub fn get_volatility(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }

        let mean = self.samples.iter().sum::<f64>() / self.samples.len() as f64;
        let variance = self.samples.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / self.samples.len() as f64;

        variance.sqrt()
    }

    pub fn reset(&mut self) {
        self.samples.clear();
    }
}

#[derive(Debug, Clone)]
pub struct ProfilingData {
    operation_timings: HashMap<String, VecDeque<Duration>>,
    total_operations: usize,
}

impl ProfilingData {
    pub fn new() -> Self {
        Self {
            operation_timings: HashMap::new(),
            total_operations: 0,
        }
    }

    pub fn record_operation(&mut self, operation: String, duration: Duration) {
        let timings = self.operation_timings.entry(operation).or_insert_with(VecDeque::new);
        timings.push_back(duration);
        if timings.len() > 1000 {
            timings.pop_front();
        }
        self.total_operations += 1;
    }

    pub fn get_average_time(&self, operation: &str) -> Option<Duration> {
        self.operation_timings.get(operation).map(|timings| {
            let total: Duration = timings.iter().sum();
            total / timings.len() as u32
        })
    }

    pub fn reset(&mut self) {
        self.operation_timings.clear();
        self.total_operations = 0;
    }
}

/// Alert system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub message: String,
    pub timestamp: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    SlowStep,
    GradientExplosion,
    GradientVanishing,
    TrainingStagnation,
    LossIncrease,
    HighMemoryUsage,
    PossibleMemoryLeak,
    ConvergenceFailure,
}

/// Performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub session_duration: Duration,
    pub total_optimization_steps: usize,
    pub total_training_epochs: usize,
    pub total_meta_steps: usize,
    pub average_step_time: Duration,
    pub average_loss: f64,
    pub current_convergence_rate: f64,
    pub peak_memory_usage: usize,
    pub average_memory_usage: f64,
    pub best_loss: f64,
    pub loss_improvement: f64,
    pub loss_trend: f64,
    pub convergence_trend: f64,
    pub memory_trend: f64,
    pub performance_alerts: Vec<PerformanceAlert>,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub quality_score: f64,
    pub recommendations: Vec<String>,
}

/// Additional metric structures (simplified implementations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossStatistics {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceStatistics {
    pub average_rate: f64,
    pub best_rate: f64,
    pub convergence_episodes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityMetrics {
    pub loss_volatility: f64,
    pub gradient_stability: f64,
    pub convergence_stability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOMetrics {
    pub total_reads: usize,
    pub total_writes: usize,
    pub total_bytes_read: usize,
    pub total_bytes_written: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetrics {
    pub total_requests: usize,
    pub total_bytes_sent: usize,
    pub total_bytes_received: usize,
    pub average_latency: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveLearningMetrics {
    pub learning_rate_adjustments: usize,
    pub batch_size_adjustments: usize,
    pub architecture_modifications: usize,
}

/// Implementation of basic functionality for metric collections
impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            training_metrics: TrainingMetricsCollection::new(),
            inference_metrics: InferenceMetricsCollection::new(),
            memory_metrics: MemoryMetricsCollection::new(),
            timing_metrics: TimingMetricsCollection::new(),
            quality_metrics: QualityMetricsCollection::new(),
            resource_metrics: ResourceMetricsCollection::new(),
            optimization_metrics: OptimizationMetricsCollection::new(),
        }
    }
}

impl TrainingMetricsCollection {
    pub fn new() -> Self {
        Self {
            total_epochs: 0,
            total_training_time: Duration::new(0, 0),
            average_loss: 0.0,
            best_loss: f64::INFINITY,
            convergence_rate: 0.0,
            meta_steps: 0,
            task_adaptations: 0,
        }
    }

    pub fn record_epoch(&mut self, loss: f64, duration: Duration, convergence: f64) {
        self.total_epochs += 1;
        self.total_training_time += duration;
        self.average_loss = (self.average_loss * (self.total_epochs - 1) as f64 + loss) / self.total_epochs as f64;
        self.best_loss = self.best_loss.min(loss);
        self.convergence_rate = convergence;
    }

    pub fn record_meta_step(&mut self, loss: f64, _duration: Duration, adaptations: usize) {
        self.meta_steps += 1;
        self.task_adaptations += adaptations;
        self.best_loss = self.best_loss.min(loss);
    }
}

impl InferenceMetricsCollection {
    pub fn new() -> Self {
        Self {
            total_inferences: 0,
            average_latency: Duration::new(0, 0),
            peak_throughput: 0.0,
            average_quality: 0.0,
            cache_hit_rate: 0.0,
        }
    }

    pub fn record_inference(&mut self, _input_size: usize, duration: Duration, quality: f64) {
        self.total_inferences += 1;
        self.average_latency = (self.average_latency * (self.total_inferences - 1) as u32 + duration) / self.total_inferences as u32;
        self.average_quality = (self.average_quality * (self.total_inferences - 1) as f64 + quality) / self.total_inferences as f64;
    }

    pub fn update_throughput(&mut self, throughput: f64) {
        self.peak_throughput = self.peak_throughput.max(throughput);
    }
}

impl MemoryMetricsCollection {
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            average_usage: 0.0,
            total_allocations: 0,
            compression_ratio: 1.0,
            fragmentation_rate: 0.0,
        }
    }

    pub fn record_usage(&mut self, usage: MemoryUsage) {
        self.peak_usage = self.peak_usage.max(usage.total_memory);
        self.total_allocations += 1;
        self.average_usage = (self.average_usage * (self.total_allocations - 1) as f64 + usage.total_memory as f64) / self.total_allocations as f64;
    }
}

impl TimingMetricsCollection {
    pub fn new() -> Self {
        Self {
            average_step_time: Duration::new(0, 0),
            total_computation_time: Duration::new(0, 0),
            operation_timings: HashMap::new(),
            profiling_overhead: Duration::new(0, 0),
        }
    }

    pub fn record_step_time(&mut self, duration: Duration) {
        self.total_computation_time += duration;
        // Update average step time logic would go here
    }

    pub fn record_operation_time(&mut self, operation: &str, duration: Duration) {
        self.operation_timings.insert(operation.to_string(), duration);
    }
}

impl QualityMetricsCollection {
    pub fn new() -> Self {
        Self {
            loss_statistics: LossStatistics { mean: 0.0, std_dev: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY },
            convergence_statistics: ConvergenceStatistics { average_rate: 0.0, best_rate: 0.0, convergence_episodes: 0 },
            stability_metrics: StabilityMetrics { loss_volatility: 0.0, gradient_stability: 0.0, convergence_stability: 0.0 },
            performance_alerts: VecDeque::new(),
        }
    }

    pub fn record_loss(&mut self, loss: f64) {
        self.loss_statistics.min = self.loss_statistics.min.min(loss);
        self.loss_statistics.max = self.loss_statistics.max.max(loss);
    }

    pub fn record_alert(&mut self, alert: PerformanceAlert) {
        self.performance_alerts.push_back(alert);
        if self.performance_alerts.len() > 100 {
            self.performance_alerts.pop_front();
        }
    }

    pub fn get_recent_alerts(&self, count: usize) -> Vec<PerformanceAlert> {
        self.performance_alerts.iter().rev().take(count).cloned().collect()
    }
}

impl ResourceMetricsCollection {
    pub fn new() -> Self {
        Self {
            cpu_usage_history: VecDeque::new(),
            memory_usage_history: VecDeque::new(),
            disk_io_metrics: DiskIOMetrics { total_reads: 0, total_writes: 0, total_bytes_read: 0, total_bytes_written: 0 },
            network_metrics: NetworkMetrics { total_requests: 0, total_bytes_sent: 0, total_bytes_received: 0, average_latency: Duration::new(0, 0) },
        }
    }

    pub fn get_average_cpu_usage(&self) -> f64 {
        if self.cpu_usage_history.is_empty() {
            0.0
        } else {
            self.cpu_usage_history.iter().sum::<f64>() / self.cpu_usage_history.len() as f64
        }
    }

    pub fn get_average_memory_usage(&self) -> f64 {
        if self.memory_usage_history.is_empty() {
            0.0
        } else {
            self.memory_usage_history.iter().sum::<f64>() / self.memory_usage_history.len() as f64
        }
    }
}

impl OptimizationMetricsCollection {
    pub fn new() -> Self {
        Self {
            update_norm_history: VecDeque::new(),
            parameter_change_rate: 0.0,
            optimization_efficiency: 0.0,
            adaptive_learning_metrics: AdaptiveLearningMetrics { learning_rate_adjustments: 0, batch_size_adjustments: 0, architecture_modifications: 0 },
        }
    }

    pub fn record_update_norm(&mut self, norm: f64) {
        self.update_norm_history.push_back(norm);
        if self.update_norm_history.len() > 1000 {
            self.update_norm_history.pop_front();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_tracker_creation() {
        let tracker = TransformerPerformanceTracker::<f32>::new();
        assert_eq!(tracker.loss_history.len(), 0);
        assert_eq!(tracker.step_timings.len(), 0);
    }

    #[test]
    fn test_record_loss() {
        let mut tracker = TransformerPerformanceTracker::<f32>::new();
        tracker.record_loss(1.5);
        tracker.record_loss(1.2);
        tracker.record_loss(0.9);

        assert_eq!(tracker.loss_history.len(), 3);
        assert_eq!(tracker.calculate_best_loss(), 0.9);
    }

    #[test]
    fn test_trend_analyzer() {
        let mut analyzer = TrendAnalyzer::new(10);

        for i in 0..5 {
            analyzer.add_sample(i as f64);
        }

        let trend = analyzer.get_trend();
        assert!(trend > 0.0); // Should be positive for increasing sequence
    }

    #[test]
    fn test_performance_report_generation() {
        let mut tracker = TransformerPerformanceTracker::<f32>::new();
        tracker.record_loss(2.0);
        tracker.record_loss(1.5);
        tracker.record_loss(1.0);

        let report = tracker.generate_report();
        assert!(report.loss_improvement > 0.0);
        assert_eq!(report.best_loss, 1.0);
    }

    #[test]
    fn test_profiling() {
        let mut tracker = TransformerPerformanceTracker::<f32>::new();

        let result = tracker.profile_operation("test_op", || {
            std::thread::sleep(Duration::from_millis(10));
            Ok(42)
        });

        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
}