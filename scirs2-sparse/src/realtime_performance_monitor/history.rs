//! Performance History Management
//!
//! This module manages the storage, retrieval, and analysis of performance
//! history data for real-time monitoring.

use super::metrics::{AggregatedMetrics, PerformanceSample, ProcessorType};
use std::collections::{HashMap, VecDeque};

/// Performance history tracking
#[derive(Debug)]
pub struct PerformanceHistory {
    pub samples: VecDeque<PerformanceSample>,
    pub aggregated_metrics: AggregatedMetrics,
    pub performance_baselines: HashMap<String, f64>,
    max_samples: usize,
    processor_metrics: HashMap<String, ProcessorMetrics>,
}

/// Per-processor metrics tracking
#[derive(Debug)]
struct ProcessorMetrics {
    samples: VecDeque<PerformanceSample>,
    aggregated: AggregatedMetrics,
    baseline_established: bool,
    last_update: u64,
}

impl PerformanceHistory {
    /// Create new performance history
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_samples),
            aggregated_metrics: AggregatedMetrics::new(),
            performance_baselines: HashMap::new(),
            max_samples,
            processor_metrics: HashMap::new(),
        }
    }

    /// Add a new performance sample
    pub fn add_sample(&mut self, sample: PerformanceSample) {
        // Add to global history
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(sample.clone());

        // Update global aggregated metrics
        self.aggregated_metrics.update_with_sample(&sample);

        // Update processor-specific metrics
        let processor_key = format!("{}:{}", sample.processor_type, sample.processor_id);
        let should_establish_baseline = {
            let processor_metrics = self
                .processor_metrics
                .entry(processor_key.clone())
                .or_insert_with(|| ProcessorMetrics {
                    samples: VecDeque::with_capacity(self.max_samples / 4),
                    aggregated: AggregatedMetrics::new(),
                    baseline_established: false,
                    last_update: sample.timestamp,
                });

            if processor_metrics.samples.len() >= self.max_samples / 4 {
                processor_metrics.samples.pop_front();
            }
            processor_metrics.samples.push_back(sample.clone());
            processor_metrics.aggregated.update_with_sample(&sample);
            processor_metrics.last_update = sample.timestamp;

            // Check if we should establish baseline
            !processor_metrics.baseline_established && processor_metrics.samples.len() >= 10
        };

        // Establish baseline if needed
        if should_establish_baseline {
            self.establish_baseline(&processor_key);
            if let Some(processor_metrics) = self.processor_metrics.get_mut(&processor_key) {
                processor_metrics.baseline_established = true;
            }
        }
    }

    /// Get recent samples for a processor
    pub fn get_processor_samples(
        &self,
        processor_type: ProcessorType,
        processor_id: &str,
    ) -> Vec<&PerformanceSample> {
        let key = format!("{}:{}", processor_type, processor_id);
        self.processor_metrics
            .get(&key)
            .map(|metrics| metrics.samples.iter().collect())
            .unwrap_or_default()
    }

    /// Get aggregated metrics for a processor
    pub fn get_processor_metrics(
        &self,
        processor_type: ProcessorType,
        processor_id: &str,
    ) -> Option<&AggregatedMetrics> {
        let key = format!("{}:{}", processor_type, processor_id);
        self.processor_metrics
            .get(&key)
            .map(|metrics| &metrics.aggregated)
    }

    /// Get all recent samples (last N samples)
    pub fn get_recent_samples(&self, count: usize) -> Vec<&PerformanceSample> {
        self.samples
            .iter()
            .rev()
            .take(count)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    /// Get samples within time range
    pub fn get_samples_in_range(&self, start_time: u64, end_time: u64) -> Vec<&PerformanceSample> {
        self.samples
            .iter()
            .filter(|sample| sample.timestamp >= start_time && sample.timestamp <= end_time)
            .collect()
    }

    /// Get samples for specific processor type
    pub fn get_samples_by_type(&self, processor_type: ProcessorType) -> Vec<&PerformanceSample> {
        self.samples
            .iter()
            .filter(|sample| sample.processor_type == processor_type)
            .collect()
    }

    /// Establish performance baseline for a processor
    fn establish_baseline(&mut self, processor_key: &str) {
        if let Some(metrics) = self.processor_metrics.get(processor_key) {
            let baseline_execution_time = metrics.aggregated.avg_execution_time;
            let baseline_throughput = metrics.aggregated.avg_throughput;
            let baseline_efficiency = metrics.aggregated.efficiency_score;

            self.performance_baselines.insert(
                format!("{}_execution_time", processor_key),
                baseline_execution_time,
            );
            self.performance_baselines
                .insert(format!("{}_throughput", processor_key), baseline_throughput);
            self.performance_baselines
                .insert(format!("{}_efficiency", processor_key), baseline_efficiency);
        }
    }

    /// Get baseline value for a metric
    pub fn get_baseline(
        &self,
        processor_type: ProcessorType,
        processor_id: &str,
        metric: &str,
    ) -> Option<f64> {
        let key = format!("{}:{}_{}", processor_type, processor_id, metric);
        self.performance_baselines.get(&key).copied()
    }

    /// Check if performance has degraded compared to baseline
    pub fn check_performance_degradation(
        &self,
        processor_type: ProcessorType,
        processor_id: &str,
        threshold: f64,
    ) -> Vec<String> {
        let mut degradations = Vec::new();
        let processor_key = format!("{}:{}", processor_type, processor_id);

        if let Some(metrics) = self.processor_metrics.get(&processor_key) {
            // Check execution time degradation
            if let Some(baseline) =
                self.get_baseline(processor_type, processor_id, "execution_time")
            {
                let current = metrics.aggregated.avg_execution_time;
                if current > baseline * (1.0 + threshold) {
                    degradations.push(format!(
                        "Execution time increased by {:.1}%",
                        ((current - baseline) / baseline) * 100.0
                    ));
                }
            }

            // Check throughput degradation
            if let Some(baseline) = self.get_baseline(processor_type, processor_id, "throughput") {
                let current = metrics.aggregated.avg_throughput;
                if current < baseline * (1.0 - threshold) {
                    degradations.push(format!(
                        "Throughput decreased by {:.1}%",
                        ((baseline - current) / baseline) * 100.0
                    ));
                }
            }

            // Check efficiency degradation
            if let Some(baseline) = self.get_baseline(processor_type, processor_id, "efficiency") {
                let current = metrics.aggregated.efficiency_score;
                if current < baseline * (1.0 - threshold) {
                    degradations.push(format!(
                        "Efficiency decreased by {:.1}%",
                        ((baseline - current) / baseline) * 100.0
                    ));
                }
            }
        }

        degradations
    }

    /// Get performance trend for a metric
    pub fn get_performance_trend(
        &self,
        processor_type: ProcessorType,
        processor_id: &str,
        metric: &str,
    ) -> PerformanceTrend {
        let samples = self.get_processor_samples(processor_type, processor_id);

        if samples.len() < 3 {
            return PerformanceTrend::Insufficient;
        }

        let values: Vec<f64> = samples
            .iter()
            .filter_map(|sample| sample.get_metric(metric))
            .collect();

        if values.len() < 3 {
            return PerformanceTrend::Insufficient;
        }

        // Simple trend analysis using linear regression
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let sum_x: f64 = x_values.iter().sum();
        let sum_y: f64 = values.iter().sum();
        let sum_xy: f64 = x_values.iter().zip(&values).map(|(x, y)| x * y).sum();
        let sum_x2: f64 = x_values.iter().map(|x| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);

        if slope > 0.01 {
            PerformanceTrend::Improving
        } else if slope < -0.01 {
            PerformanceTrend::Degrading
        } else {
            PerformanceTrend::Stable
        }
    }

    /// Get processor activity summary
    pub fn get_processor_summary(&self) -> Vec<ProcessorSummary> {
        self.processor_metrics
            .iter()
            .map(|(key, metrics)| {
                let parts: Vec<&str> = key.split(':').collect();
                let processor_type = match parts[0] {
                    "QuantumInspired" => ProcessorType::QuantumInspired,
                    "NeuralAdaptive" => ProcessorType::NeuralAdaptive,
                    "QuantumNeuralHybrid" => ProcessorType::QuantumNeuralHybrid,
                    "MemoryCompression" => ProcessorType::MemoryCompression,
                    _ => ProcessorType::QuantumInspired, // Default fallback
                };
                let processor_id = parts.get(1).unwrap_or(&"unknown").to_string();

                ProcessorSummary {
                    processor_type,
                    processor_id,
                    sample_count: metrics.samples.len(),
                    avg_execution_time: metrics.aggregated.avg_execution_time,
                    avg_throughput: metrics.aggregated.avg_throughput,
                    efficiency_score: metrics.aggregated.efficiency_score,
                    last_update: metrics.last_update,
                    baseline_established: metrics.baseline_established,
                }
            })
            .collect()
    }

    /// Clear old samples beyond retention period
    pub fn cleanup_old_samples(&mut self, retention_time_ms: u64) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let cutoff_time = current_time.saturating_sub(retention_time_ms);

        // Clean global samples
        while let Some(sample) = self.samples.front() {
            if sample.timestamp < cutoff_time {
                self.samples.pop_front();
            } else {
                break;
            }
        }

        // Clean processor-specific samples
        for metrics in self.processor_metrics.values_mut() {
            while let Some(sample) = metrics.samples.front() {
                if sample.timestamp < cutoff_time {
                    metrics.samples.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Get memory usage of history data
    pub fn memory_usage(&self) -> usize {
        let sample_size = std::mem::size_of::<PerformanceSample>();
        self.samples.len() * sample_size
            + self
                .processor_metrics
                .values()
                .map(|metrics| metrics.samples.len() * sample_size)
                .sum::<usize>()
    }

    /// Reset all history data
    pub fn clear(&mut self) {
        self.samples.clear();
        self.aggregated_metrics.reset();
        self.performance_baselines.clear();
        self.processor_metrics.clear();
    }
}

/// Performance trend indicator
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerformanceTrend {
    Improving,
    Stable,
    Degrading,
    Insufficient, // Not enough data for trend analysis
}

impl std::fmt::Display for PerformanceTrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PerformanceTrend::Improving => write!(f, "Improving"),
            PerformanceTrend::Stable => write!(f, "Stable"),
            PerformanceTrend::Degrading => write!(f, "Degrading"),
            PerformanceTrend::Insufficient => write!(f, "Insufficient Data"),
        }
    }
}

/// Summary information for a processor
#[derive(Debug, Clone)]
pub struct ProcessorSummary {
    pub processor_type: ProcessorType,
    pub processor_id: String,
    pub sample_count: usize,
    pub avg_execution_time: f64,
    pub avg_throughput: f64,
    pub efficiency_score: f64,
    pub last_update: u64,
    pub baseline_established: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_history_creation() {
        let history = PerformanceHistory::new(1000);
        assert_eq!(history.samples.len(), 0);
        assert_eq!(history.max_samples, 1000);
    }

    #[test]
    fn test_add_sample() {
        let mut history = PerformanceHistory::new(10);

        let sample =
            PerformanceSample::new(ProcessorType::QuantumInspired, "test-processor".to_string())
                .with_execution_time(100.0);

        history.add_sample(sample);
        assert_eq!(history.samples.len(), 1);
        assert_eq!(history.aggregated_metrics.sample_count, 1);
    }

    #[test]
    fn test_sample_capacity_limit() {
        let mut history = PerformanceHistory::new(3);

        for i in 0..5 {
            let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
                .with_execution_time(i as f64 * 10.0);
            history.add_sample(sample);
        }

        assert_eq!(history.samples.len(), 3);
        // Should contain the last 3 samples
        assert_eq!(history.samples[0].execution_time_ms, 20.0);
        assert_eq!(history.samples[1].execution_time_ms, 30.0);
        assert_eq!(history.samples[2].execution_time_ms, 40.0);
    }

    #[test]
    fn test_processor_specific_samples() {
        let mut history = PerformanceHistory::new(100);

        let sample1 =
            PerformanceSample::new(ProcessorType::QuantumInspired, "processor1".to_string())
                .with_execution_time(100.0);

        let sample2 =
            PerformanceSample::new(ProcessorType::NeuralAdaptive, "processor2".to_string())
                .with_execution_time(200.0);

        history.add_sample(sample1);
        history.add_sample(sample2);

        let quantum_samples =
            history.get_processor_samples(ProcessorType::QuantumInspired, "processor1");
        assert_eq!(quantum_samples.len(), 1);
        assert_eq!(quantum_samples[0].execution_time_ms, 100.0);

        let neural_samples =
            history.get_processor_samples(ProcessorType::NeuralAdaptive, "processor2");
        assert_eq!(neural_samples.len(), 1);
        assert_eq!(neural_samples[0].execution_time_ms, 200.0);
    }

    #[test]
    fn test_recent_samples() {
        let mut history = PerformanceHistory::new(100);

        for i in 0..10 {
            let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
                .with_execution_time(i as f64);
            history.add_sample(sample);
        }

        let recent = history.get_recent_samples(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].execution_time_ms, 7.0);
        assert_eq!(recent[1].execution_time_ms, 8.0);
        assert_eq!(recent[2].execution_time_ms, 9.0);
    }

    #[test]
    fn test_samples_by_type() {
        let mut history = PerformanceHistory::new(100);

        for i in 0..5 {
            let quantum_sample =
                PerformanceSample::new(ProcessorType::QuantumInspired, "quantum".to_string())
                    .with_execution_time(i as f64);

            let neural_sample =
                PerformanceSample::new(ProcessorType::NeuralAdaptive, "neural".to_string())
                    .with_execution_time(i as f64 + 100.0);

            history.add_sample(quantum_sample);
            history.add_sample(neural_sample);
        }

        let quantum_samples = history.get_samples_by_type(ProcessorType::QuantumInspired);
        let neural_samples = history.get_samples_by_type(ProcessorType::NeuralAdaptive);

        assert_eq!(quantum_samples.len(), 5);
        assert_eq!(neural_samples.len(), 5);
    }

    #[test]
    fn test_baseline_establishment() {
        let mut history = PerformanceHistory::new(100);

        // Add enough samples to establish baseline
        for i in 0..15 {
            let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
                .with_execution_time(100.0 + i as f64);
            history.add_sample(sample);
        }

        let baseline =
            history.get_baseline(ProcessorType::QuantumInspired, "test", "execution_time");
        assert!(baseline.is_some());
        assert!(baseline.unwrap() > 100.0);
    }

    #[test]
    fn test_performance_degradation_detection() {
        let mut history = PerformanceHistory::new(100);

        // Establish baseline with good performance
        for i in 0..15 {
            let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
                .with_execution_time(100.0)
                .with_throughput(1000.0);
            history.add_sample(sample);
        }

        // Add samples with degraded performance
        for i in 0..5 {
            let sample = PerformanceSample::new(
                ProcessorType::QuantumInspired,
                "test".to_string(),
            )
            .with_execution_time(200.0) // Doubled execution time
            .with_throughput(500.0); // Halved throughput
            history.add_sample(sample);
        }

        let degradations = history.check_performance_degradation(
            ProcessorType::QuantumInspired,
            "test",
            0.1, // 10% threshold
        );

        assert!(!degradations.is_empty());
    }

    #[test]
    fn test_processor_summary() {
        let mut history = PerformanceHistory::new(100);

        let sample =
            PerformanceSample::new(ProcessorType::QuantumInspired, "test-processor".to_string())
                .with_execution_time(100.0)
                .with_throughput(500.0);

        history.add_sample(sample);

        let summaries = history.get_processor_summary();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].processor_type, ProcessorType::QuantumInspired);
        assert_eq!(summaries[0].processor_id, "test-processor");
        assert_eq!(summaries[0].sample_count, 1);
    }

    #[test]
    fn test_memory_usage_calculation() {
        let mut history = PerformanceHistory::new(100);

        for i in 0..10 {
            let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string());
            history.add_sample(sample);
        }

        let memory_usage = history.memory_usage();
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_clear_history() {
        let mut history = PerformanceHistory::new(100);

        let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string());
        history.add_sample(sample);

        assert_eq!(history.samples.len(), 1);

        history.clear();
        assert_eq!(history.samples.len(), 0);
        assert_eq!(history.aggregated_metrics.sample_count, 0);
    }
}
