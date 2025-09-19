//! GPU performance profiling and benchmarking

use std::collections::HashMap;

/// Performance profiler for GPU operations
pub struct GpuPerformanceProfiler {
    measurements: std::collections::HashMap<String, Vec<f64>>,
}

impl GpuPerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            measurements: std::collections::HashMap::new(),
        }
    }

    /// Record a performance measurement
    pub fn record(&mut self, operation: &str, time_seconds: f64) {
        self.measurements
            .entry(operation.to_string())
            .or_default()
            .push(time_seconds);
    }

    /// Get average time for an operation
    pub fn average_time(&self, operation: &str) -> Option<f64> {
        self.measurements
            .get(operation)
            .map(|times| times.iter().sum::<f64>() / times.len() as f64)
    }

    /// Get best time for an operation
    pub fn best_time(&self, operation: &str) -> Option<f64> {
        self.measurements
            .get(operation)
            .and_then(|times| times.iter().min_by(|a, b| a.partial_cmp(b).unwrap()))
            .copied()
    }

    /// Get all recorded operations
    pub fn operations(&self) -> Vec<&str> {
        self.measurements.keys().map(|s| s.as_str()).collect()
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
    }
}

impl Default for GpuPerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}
