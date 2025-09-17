//! Performance Metrics and Measurement
//!
//! This module contains structures for capturing and tracking performance metrics
//! from various Advanced mode processors.

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Type of Advanced processor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcessorType {
    QuantumInspired,
    NeuralAdaptive,
    QuantumNeuralHybrid,
    MemoryCompression,
}

impl std::fmt::Display for ProcessorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessorType::QuantumInspired => write!(f, "QuantumInspired"),
            ProcessorType::NeuralAdaptive => write!(f, "NeuralAdaptive"),
            ProcessorType::QuantumNeuralHybrid => write!(f, "QuantumNeuralHybrid"),
            ProcessorType::MemoryCompression => write!(f, "MemoryCompression"),
        }
    }
}

impl ProcessorType {
    /// Get all processor types
    pub fn all() -> Vec<ProcessorType> {
        vec![
            ProcessorType::QuantumInspired,
            ProcessorType::NeuralAdaptive,
            ProcessorType::QuantumNeuralHybrid,
            ProcessorType::MemoryCompression,
        ]
    }

    /// Check if processor type supports specific metrics
    pub fn supports_quantum_metrics(&self) -> bool {
        matches!(
            self,
            ProcessorType::QuantumInspired | ProcessorType::QuantumNeuralHybrid
        )
    }

    pub fn supports_neural_metrics(&self) -> bool {
        matches!(
            self,
            ProcessorType::NeuralAdaptive | ProcessorType::QuantumNeuralHybrid
        )
    }

    pub fn supports_compression_metrics(&self) -> bool {
        matches!(self, ProcessorType::MemoryCompression)
    }
}

/// Individual performance sample
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    pub timestamp: u64,
    pub processor_type: ProcessorType,
    pub processor_id: String,
    pub execution_time_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_ratio: f64,
    pub error_rate: f64,
    pub cpu_utilization: f64,
    pub gpu_utilization: f64,
    pub quantum_coherence: Option<f64>,
    pub neural_confidence: Option<f64>,
    pub compression_ratio: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

impl PerformanceSample {
    /// Create a new performance sample
    pub fn new(processor_type: ProcessorType, processor_id: String) -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            processor_type,
            processor_id,
            execution_time_ms: 0.0,
            throughput_ops_per_sec: 0.0,
            memory_usage_mb: 0.0,
            cache_hit_ratio: 0.0,
            error_rate: 0.0,
            cpu_utilization: 0.0,
            gpu_utilization: 0.0,
            quantum_coherence: None,
            neural_confidence: None,
            compression_ratio: None,
            custom_metrics: HashMap::new(),
        }
    }

    /// Create sample with execution time
    pub fn with_execution_time(mut self, execution_time_ms: f64) -> Self {
        self.execution_time_ms = execution_time_ms;
        self
    }

    /// Create sample with throughput
    pub fn with_throughput(mut self, throughput_ops_per_sec: f64) -> Self {
        self.throughput_ops_per_sec = throughput_ops_per_sec;
        self
    }

    /// Create sample with memory usage
    pub fn with_memory_usage(mut self, memory_usage_mb: f64) -> Self {
        self.memory_usage_mb = memory_usage_mb;
        self
    }

    /// Create sample with cache hit ratio
    pub fn with_cache_hit_ratio(mut self, cache_hit_ratio: f64) -> Self {
        self.cache_hit_ratio = cache_hit_ratio.clamp(0.0, 1.0);
        self
    }

    /// Create sample with error rate
    pub fn with_error_rate(mut self, error_rate: f64) -> Self {
        self.error_rate = error_rate.clamp(0.0, 1.0);
        self
    }

    /// Create sample with CPU utilization
    pub fn with_cpu_utilization(mut self, cpu_utilization: f64) -> Self {
        self.cpu_utilization = cpu_utilization.clamp(0.0, 1.0);
        self
    }

    /// Create sample with GPU utilization
    pub fn with_gpu_utilization(mut self, gpu_utilization: f64) -> Self {
        self.gpu_utilization = gpu_utilization.clamp(0.0, 1.0);
        self
    }

    /// Set quantum coherence (for quantum processors)
    pub fn with_quantum_coherence(mut self, coherence: f64) -> Self {
        if self.processor_type.supports_quantum_metrics() {
            self.quantum_coherence = Some(coherence.clamp(0.0, 1.0));
        }
        self
    }

    /// Set neural confidence (for neural processors)
    pub fn with_neural_confidence(mut self, confidence: f64) -> Self {
        if self.processor_type.supports_neural_metrics() {
            self.neural_confidence = Some(confidence.clamp(0.0, 1.0));
        }
        self
    }

    /// Set compression ratio (for memory compression)
    pub fn with_compression_ratio(mut self, ratio: f64) -> Self {
        if self.processor_type.supports_compression_metrics() {
            self.compression_ratio = Some(ratio.max(0.0));
        }
        self
    }

    /// Add custom metric
    pub fn with_custom_metric(mut self, name: String, value: f64) -> Self {
        self.custom_metrics.insert(name, value);
        self
    }

    /// Calculate efficiency score (composite metric)
    pub fn efficiency_score(&self) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        // Throughput factor (higher is better)
        if self.throughput_ops_per_sec > 0.0 {
            score += (self.throughput_ops_per_sec / 1000.0).min(1.0);
            factors += 1;
        }

        // Execution time factor (lower is better)
        if self.execution_time_ms > 0.0 {
            score += (1.0 / (1.0 + self.execution_time_ms / 1000.0)).min(1.0);
            factors += 1;
        }

        // Cache hit ratio factor
        score += self.cache_hit_ratio;
        factors += 1;

        // Error rate factor (lower is better)
        score += 1.0 - self.error_rate;
        factors += 1;

        // Resource utilization factor (balanced is better)
        let cpu_factor = if self.cpu_utilization > 0.9 {
            0.5
        } else {
            self.cpu_utilization
        };
        let gpu_factor = if self.gpu_utilization > 0.9 {
            0.5
        } else {
            self.gpu_utilization
        };
        score += (cpu_factor + gpu_factor) / 2.0;
        factors += 1;

        // Processor-specific factors
        if let Some(coherence) = self.quantum_coherence {
            score += coherence;
            factors += 1;
        }

        if let Some(confidence) = self.neural_confidence {
            score += confidence;
            factors += 1;
        }

        if let Some(ratio) = self.compression_ratio {
            score += (ratio / 10.0).min(1.0); // Normalize compression ratio
            factors += 1;
        }

        if factors > 0 {
            score / factors as f64
        } else {
            0.0
        }
    }

    /// Check if sample indicates performance issues
    pub fn has_performance_issues(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.execution_time_ms > 1000.0 {
            issues.push("High execution time".to_string());
        }

        if self.throughput_ops_per_sec < 10.0 {
            issues.push("Low throughput".to_string());
        }

        if self.cache_hit_ratio < 0.5 {
            issues.push("Low cache hit ratio".to_string());
        }

        if self.error_rate > 0.1 {
            issues.push("High error rate".to_string());
        }

        if self.cpu_utilization > 0.95 {
            issues.push("CPU overutilization".to_string());
        }

        if self.memory_usage_mb > 1000.0 {
            issues.push("High memory usage".to_string());
        }

        if let Some(coherence) = self.quantum_coherence {
            if coherence < 0.3 {
                issues.push("Low quantum coherence".to_string());
            }
        }

        if let Some(confidence) = self.neural_confidence {
            if confidence < 0.5 {
                issues.push("Low neural confidence".to_string());
            }
        }

        issues
    }

    /// Get metric value by name
    pub fn get_metric(&self, name: &str) -> Option<f64> {
        match name {
            "execution_time_ms" => Some(self.execution_time_ms),
            "throughput_ops_per_sec" => Some(self.throughput_ops_per_sec),
            "memory_usage_mb" => Some(self.memory_usage_mb),
            "cache_hit_ratio" => Some(self.cache_hit_ratio),
            "error_rate" => Some(self.error_rate),
            "cpu_utilization" => Some(self.cpu_utilization),
            "gpu_utilization" => Some(self.gpu_utilization),
            "quantum_coherence" => self.quantum_coherence,
            "neural_confidence" => self.neural_confidence,
            "compression_ratio" => self.compression_ratio,
            "efficiency_score" => Some(self.efficiency_score()),
            _ => self.custom_metrics.get(name).copied(),
        }
    }

    /// Get all available metric names
    pub fn metric_names(&self) -> Vec<String> {
        let mut names = vec![
            "execution_time_ms".to_string(),
            "throughput_ops_per_sec".to_string(),
            "memory_usage_mb".to_string(),
            "cache_hit_ratio".to_string(),
            "error_rate".to_string(),
            "cpu_utilization".to_string(),
            "gpu_utilization".to_string(),
            "efficiency_score".to_string(),
        ];

        if self.quantum_coherence.is_some() {
            names.push("quantum_coherence".to_string());
        }

        if self.neural_confidence.is_some() {
            names.push("neural_confidence".to_string());
        }

        if self.compression_ratio.is_some() {
            names.push("compression_ratio".to_string());
        }

        names.extend(self.custom_metrics.keys().cloned());
        names
    }
}

/// Aggregated performance metrics
#[derive(Debug, Default, Clone)]
pub struct AggregatedMetrics {
    pub avg_execution_time: f64,
    pub avg_throughput: f64,
    pub avg_memory_usage: f64,
    pub avg_cache_hit_ratio: f64,
    pub avg_error_rate: f64,
    pub avg_cpu_utilization: f64,
    pub avg_gpu_utilization: f64,
    pub peak_throughput: f64,
    pub min_execution_time: f64,
    pub max_execution_time: f64,
    pub total_operations: usize,
    pub efficiency_score: f64,
    pub sample_count: usize,
}

impl AggregatedMetrics {
    /// Create new aggregated metrics
    pub fn new() -> Self {
        Self {
            min_execution_time: f64::INFINITY,
            max_execution_time: 0.0,
            ..Default::default()
        }
    }

    /// Update aggregated metrics with a new sample
    pub fn update_with_sample(&mut self, sample: &PerformanceSample) {
        let n = self.sample_count as f64;
        let new_n = n + 1.0;

        // Update running averages
        self.avg_execution_time = (self.avg_execution_time * n + sample.execution_time_ms) / new_n;
        self.avg_throughput = (self.avg_throughput * n + sample.throughput_ops_per_sec) / new_n;
        self.avg_memory_usage = (self.avg_memory_usage * n + sample.memory_usage_mb) / new_n;
        self.avg_cache_hit_ratio = (self.avg_cache_hit_ratio * n + sample.cache_hit_ratio) / new_n;
        self.avg_error_rate = (self.avg_error_rate * n + sample.error_rate) / new_n;
        self.avg_cpu_utilization = (self.avg_cpu_utilization * n + sample.cpu_utilization) / new_n;
        self.avg_gpu_utilization = (self.avg_gpu_utilization * n + sample.gpu_utilization) / new_n;
        self.efficiency_score = (self.efficiency_score * n + sample.efficiency_score()) / new_n;

        // Update extremes
        self.peak_throughput = self.peak_throughput.max(sample.throughput_ops_per_sec);
        self.min_execution_time = self.min_execution_time.min(sample.execution_time_ms);
        self.max_execution_time = self.max_execution_time.max(sample.execution_time_ms);

        self.total_operations += 1;
        self.sample_count += 1;
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Get performance variance for execution time
    pub fn execution_time_variance(&self, samples: &[PerformanceSample]) -> f64 {
        if samples.len() <= 1 {
            return 0.0;
        }

        let mean = self.avg_execution_time;
        let variance = samples
            .iter()
            .map(|s| (s.execution_time_ms - mean).powi(2))
            .sum::<f64>()
            / (samples.len() - 1) as f64;

        variance
    }

    /// Get performance stability score (lower variance = higher stability)
    pub fn stability_score(&self, samples: &[PerformanceSample]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }

        let variance = self.execution_time_variance(samples);
        let coefficient_of_variation = if self.avg_execution_time > 0.0 {
            variance.sqrt() / self.avg_execution_time
        } else {
            0.0
        };

        // Convert to stability score (0-1, higher is more stable)
        (1.0 / (1.0 + coefficient_of_variation)).clamp(0.0, 1.0)
    }
}

/// System metrics tracking
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub gpu_usage: f64,
    pub network_io: f64,
    pub disk_io: f64,
    pub temperature: f64,
    pub power_consumption: f64,
    pub system_load: f64,
    pub available_memory_mb: f64,
    pub cpu_frequency_mhz: f64,
    pub timestamp: u64,
}

impl SystemMetrics {
    /// Create new system metrics
    pub fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            gpu_usage: 0.0,
            network_io: 0.0,
            disk_io: 0.0,
            temperature: 0.0,
            power_consumption: 0.0,
            system_load: 0.0,
            available_memory_mb: 0.0,
            cpu_frequency_mhz: 0.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }

    /// Update timestamp
    pub fn update_timestamp(&mut self) {
        self.timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Check if system is under high load
    pub fn is_high_load(&self) -> bool {
        self.cpu_usage > 0.8 || self.memory_usage > 0.9 || self.system_load > 2.0
    }

    /// Get system health score (0-1, higher is better)
    pub fn health_score(&self) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;

        // CPU usage (optimal around 60-80%)
        let cpu_score = if self.cpu_usage > 0.9 {
            0.2
        } else if self.cpu_usage > 0.8 {
            0.8
        } else if self.cpu_usage > 0.6 {
            1.0
        } else {
            self.cpu_usage / 0.6
        };
        score += cpu_score;
        factors += 1;

        // Memory usage (lower is better)
        score += 1.0 - self.memory_usage;
        factors += 1;

        // Temperature (assuming normal is < 70°C)
        if self.temperature > 0.0 {
            let temp_score = if self.temperature > 80.0 {
                0.0
            } else if self.temperature > 70.0 {
                (80.0 - self.temperature) / 10.0
            } else {
                1.0
            };
            score += temp_score;
            factors += 1;
        }

        // System load (lower is better)
        let load_score = (1.0 / (1.0 + self.system_load)).clamp(0.0, 1.0);
        score += load_score;
        factors += 1;

        if factors > 0 {
            score / factors as f64
        } else {
            0.0
        }
    }
}

impl Default for SystemMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution timing helper for measuring performance
#[derive(Debug)]
pub struct ExecutionTimer {
    start_time: Instant,
    label: Option<String>,
}

impl ExecutionTimer {
    /// Create a new execution timer
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            label: None,
        }
    }

    /// Create a labeled execution timer
    pub fn with_label(label: String) -> Self {
        Self {
            start_time: Instant::now(),
            label: Some(label),
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_millis() as f64
    }

    /// Get elapsed time in microseconds
    pub fn elapsed_us(&self) -> f64 {
        self.start_time.elapsed().as_micros() as f64
    }

    /// Get elapsed time in nanoseconds
    pub fn elapsed_ns(&self) -> u64 {
        self.start_time.elapsed().as_nanos() as u64
    }

    /// Restart the timer
    pub fn restart(&mut self) {
        self.start_time = Instant::now();
    }

    /// Get label if set
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Set label
    pub fn set_label(&mut self, label: String) {
        self.label = Some(label);
    }

    /// Create a performance sample from this timer
    pub fn to_sample(
        &self,
        processor_type: ProcessorType,
        processor_id: String,
    ) -> PerformanceSample {
        PerformanceSample::new(processor_type, processor_id).with_execution_time(self.elapsed_ms())
    }
}

impl Default for ExecutionTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_type_display() {
        assert_eq!(
            ProcessorType::QuantumInspired.to_string(),
            "QuantumInspired"
        );
        assert_eq!(ProcessorType::NeuralAdaptive.to_string(), "NeuralAdaptive");
        assert_eq!(
            ProcessorType::QuantumNeuralHybrid.to_string(),
            "QuantumNeuralHybrid"
        );
        assert_eq!(
            ProcessorType::MemoryCompression.to_string(),
            "MemoryCompression"
        );
    }

    #[test]
    fn test_processor_type_capabilities() {
        assert!(ProcessorType::QuantumInspired.supports_quantum_metrics());
        assert!(!ProcessorType::QuantumInspired.supports_neural_metrics());
        assert!(!ProcessorType::QuantumInspired.supports_compression_metrics());

        assert!(!ProcessorType::NeuralAdaptive.supports_quantum_metrics());
        assert!(ProcessorType::NeuralAdaptive.supports_neural_metrics());
        assert!(!ProcessorType::NeuralAdaptive.supports_compression_metrics());

        assert!(ProcessorType::QuantumNeuralHybrid.supports_quantum_metrics());
        assert!(ProcessorType::QuantumNeuralHybrid.supports_neural_metrics());
        assert!(!ProcessorType::QuantumNeuralHybrid.supports_compression_metrics());

        assert!(!ProcessorType::MemoryCompression.supports_quantum_metrics());
        assert!(!ProcessorType::MemoryCompression.supports_neural_metrics());
        assert!(ProcessorType::MemoryCompression.supports_compression_metrics());
    }

    #[test]
    fn test_performance_sample_creation() {
        let sample =
            PerformanceSample::new(ProcessorType::QuantumInspired, "test-processor".to_string());

        assert_eq!(sample.processor_type, ProcessorType::QuantumInspired);
        assert_eq!(sample.processor_id, "test-processor");
        assert_eq!(sample.execution_time_ms, 0.0);
        assert!(sample.timestamp > 0);
    }

    #[test]
    fn test_performance_sample_builder() {
        let sample = PerformanceSample::new(ProcessorType::NeuralAdaptive, "test".to_string())
            .with_execution_time(100.0)
            .with_throughput(500.0)
            .with_cache_hit_ratio(0.8)
            .with_neural_confidence(0.9);

        assert_eq!(sample.execution_time_ms, 100.0);
        assert_eq!(sample.throughput_ops_per_sec, 500.0);
        assert_eq!(sample.cache_hit_ratio, 0.8);
        assert_eq!(sample.neural_confidence, Some(0.9));
    }

    #[test]
    fn test_efficiency_score_calculation() {
        let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
            .with_execution_time(100.0)
            .with_throughput(1000.0)
            .with_cache_hit_ratio(0.9)
            .with_error_rate(0.1)
            .with_cpu_utilization(0.7)
            .with_gpu_utilization(0.6);

        let score = sample.efficiency_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_performance_issues_detection() {
        let problematic_sample = PerformanceSample::new(
            ProcessorType::QuantumInspired,
            "test".to_string(),
        )
        .with_execution_time(2000.0) // High execution time
        .with_throughput(5.0) // Low throughput
        .with_cache_hit_ratio(0.3) // Low cache hit ratio
        .with_error_rate(0.2); // High error rate

        let issues = problematic_sample.has_performance_issues();
        assert!(!issues.is_empty());
        assert!(issues.contains(&"High execution time".to_string()));
        assert!(issues.contains(&"Low throughput".to_string()));
        assert!(issues.contains(&"Low cache hit ratio".to_string()));
        assert!(issues.contains(&"High error rate".to_string()));
    }

    #[test]
    fn test_aggregated_metrics_update() {
        let mut metrics = AggregatedMetrics::new();

        let sample1 = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
            .with_execution_time(100.0)
            .with_throughput(500.0);

        let sample2 = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
            .with_execution_time(200.0)
            .with_throughput(400.0);

        metrics.update_with_sample(&sample1);
        metrics.update_with_sample(&sample2);

        assert_eq!(metrics.sample_count, 2);
        assert_eq!(metrics.avg_execution_time, 150.0);
        assert_eq!(metrics.avg_throughput, 450.0);
        assert_eq!(metrics.peak_throughput, 500.0);
        assert_eq!(metrics.min_execution_time, 100.0);
        assert_eq!(metrics.max_execution_time, 200.0);
    }

    #[test]
    fn test_system_metrics() {
        let mut metrics = SystemMetrics::new();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert!(metrics.timestamp > 0);

        metrics.cpu_usage = 0.5;
        metrics.memory_usage = 0.3;
        metrics.temperature = 65.0;
        metrics.system_load = 1.0;

        assert!(!metrics.is_high_load());

        let health = metrics.health_score();
        assert!(health > 0.0 && health <= 1.0);
    }

    #[test]
    fn test_execution_timer() {
        let timer = ExecutionTimer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));

        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10.0);

        let sample = timer.to_sample(ProcessorType::QuantumInspired, "test".to_string());
        assert!(sample.execution_time_ms >= 10.0);
    }

    #[test]
    fn test_execution_timer_with_label() {
        let mut timer = ExecutionTimer::with_label("test-operation".to_string());
        assert_eq!(timer.label(), Some("test-operation"));

        timer.set_label("new-operation".to_string());
        assert_eq!(timer.label(), Some("new-operation"));
    }

    #[test]
    fn test_metric_access() {
        let sample = PerformanceSample::new(ProcessorType::QuantumInspired, "test".to_string())
            .with_execution_time(100.0)
            .with_custom_metric("custom_value".to_string(), 42.0);

        assert_eq!(sample.get_metric("execution_time_ms"), Some(100.0));
        assert_eq!(sample.get_metric("custom_value"), Some(42.0));
        assert_eq!(sample.get_metric("nonexistent"), None);

        let names = sample.metric_names();
        assert!(names.contains(&"execution_time_ms".to_string()));
        assert!(names.contains(&"custom_value".to_string()));
    }
}
