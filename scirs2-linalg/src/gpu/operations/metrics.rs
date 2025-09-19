//! Multi-dimensional performance metrics for GPU operations

use std::collections::HashMap;

/// Multi-dimensional performance metrics for comprehensive evaluation
#[derive(Debug)]
pub struct MultiDimensionalMetrics {
    /// Execution time metrics
    execution_times: HashMap<String, TimeMetrics>,
    /// Memory usage metrics
    memory_metrics: HashMap<String, MemoryMetrics>,
    /// Energy consumption metrics
    energy_metrics: HashMap<String, EnergyMetrics>,
    /// Throughput metrics
    throughput_metrics: HashMap<String, ThroughputMetrics>,
}

#[derive(Debug, Clone)]
pub struct TimeMetrics {
    pub cpu_time: RunningStats,
    pub gpu_time: RunningStats,
    pub transfer_time: RunningStats,
}

#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub peak_usage: RunningStats,
    pub transfer_volume: RunningStats,
    pub bandwidth_utilization: RunningStats,
}

#[derive(Debug, Clone)]
pub struct EnergyMetrics {
    pub cpu_energy: RunningStats,
    pub gpu_energy: RunningStats,
    pub total_energy: RunningStats,
}

#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    pub operations_per_second: RunningStats,
    pub flops: RunningStats,
    pub memory_bandwidth: RunningStats,
}

/// Running statistics for incremental computation
#[derive(Debug, Clone)]
pub struct RunningStats {
    pub count: usize,
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
}

impl RunningStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            variance: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.variance += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    pub fn std_dev(&self) -> f64 {
        if self.count > 1 {
            (self.variance / (self.count - 1) as f64).sqrt()
        } else {
            0.0
        }
    }
}

impl MultiDimensionalMetrics {
    pub fn new() -> Self {
        Self {
            execution_times: HashMap::new(),
            memory_metrics: HashMap::new(),
            energy_metrics: HashMap::new(),
            throughput_metrics: HashMap::new(),
        }
    }

    /// Record execution time metrics for an operation
    pub fn record_execution_time(
        &mut self,
        operation: &str,
        cpu_time: f64,
        gpu_time: f64,
        transfer_time: f64,
    ) {
        let metrics = self
            .execution_times
            .entry(operation.to_string())
            .or_insert_with(|| TimeMetrics {
                cpu_time: RunningStats::new(),
                gpu_time: RunningStats::new(),
                transfer_time: RunningStats::new(),
            });

        metrics.cpu_time.update(cpu_time);
        metrics.gpu_time.update(gpu_time);
        metrics.transfer_time.update(transfer_time);
    }

    /// Record memory metrics for an operation
    pub fn record_memory_metrics(
        &mut self,
        operation: &str,
        peak_usage: f64,
        transfer_volume: f64,
        bandwidth_utilization: f64,
    ) {
        let metrics = self
            .memory_metrics
            .entry(operation.to_string())
            .or_insert_with(|| MemoryMetrics {
                peak_usage: RunningStats::new(),
                transfer_volume: RunningStats::new(),
                bandwidth_utilization: RunningStats::new(),
            });

        metrics.peak_usage.update(peak_usage);
        metrics.transfer_volume.update(transfer_volume);
        metrics.bandwidth_utilization.update(bandwidth_utilization);
    }

    /// Record energy metrics for an operation
    pub fn record_energy_metrics(&mut self, operation: &str, cpu_energy: f64, gpu_energy: f64) {
        let metrics = self
            .energy_metrics
            .entry(operation.to_string())
            .or_insert_with(|| EnergyMetrics {
                cpu_energy: RunningStats::new(),
                gpu_energy: RunningStats::new(),
                total_energy: RunningStats::new(),
            });

        metrics.cpu_energy.update(cpu_energy);
        metrics.gpu_energy.update(gpu_energy);
        metrics.total_energy.update(cpu_energy + gpu_energy);
    }

    /// Record throughput metrics for an operation
    pub fn record_throughput_metrics(
        &mut self,
        operation: &str,
        ops_per_sec: f64,
        flops: f64,
        memory_bandwidth: f64,
    ) {
        let metrics = self
            .throughput_metrics
            .entry(operation.to_string())
            .or_insert_with(|| ThroughputMetrics {
                operations_per_second: RunningStats::new(),
                flops: RunningStats::new(),
                memory_bandwidth: RunningStats::new(),
            });

        metrics.operations_per_second.update(ops_per_sec);
        metrics.flops.update(flops);
        metrics.memory_bandwidth.update(memory_bandwidth);
    }

    /// Get time metrics for an operation
    pub fn get_time_metrics(&self, operation: &str) -> Option<&TimeMetrics> {
        self.execution_times.get(operation)
    }

    /// Get memory metrics for an operation
    pub fn get_memory_metrics(&self, operation: &str) -> Option<&MemoryMetrics> {
        self.memory_metrics.get(operation)
    }

    /// Get energy metrics for an operation
    pub fn get_energy_metrics(&self, operation: &str) -> Option<&EnergyMetrics> {
        self.energy_metrics.get(operation)
    }

    /// Get throughput metrics for an operation
    pub fn get_throughput_metrics(&self, operation: &str) -> Option<&ThroughputMetrics> {
        self.throughput_metrics.get(operation)
    }

    /// Get all operations that have recorded metrics
    pub fn get_operations(&self) -> Vec<&str> {
        let mut ops = std::collections::HashSet::new();
        ops.extend(self.execution_times.keys().map(|s| s.as_str()));
        ops.extend(self.memory_metrics.keys().map(|s| s.as_str()));
        ops.extend(self.energy_metrics.keys().map(|s| s.as_str()));
        ops.extend(self.throughput_metrics.keys().map(|s| s.as_str()));
        ops.into_iter().collect()
    }
}
