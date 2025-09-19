//! Statistics tracking for distributed operations

use std::collections::HashMap;

/// Global statistics for distributed operations
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Total number of operations performed
    pub operations_count: usize,

    /// Total data transferred (bytes)
    pub bytes_transferred: usize,

    /// Communication time (milliseconds)
    pub comm_time_ms: u64,

    /// Computation time (milliseconds)
    pub compute_time_ms: u64,

    /// Number of communication events
    pub comm_events: usize,

    /// Load balancing efficiency (0.0 - 1.0)
    pub load_balance_efficiency: f64,

    /// Memory usage per node
    pub memory_usage_per_node: HashMap<usize, usize>,
}

impl DistributedStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a communication event
    pub fn record_communication(&mut self, bytes: usize, time_ms: u64) {
        self.bytes_transferred += bytes;
        self.comm_time_ms += time_ms;
        self.comm_events += 1;
    }

    /// Record computation time
    pub fn record_computation(&mut self, time_ms: u64) {
        self.compute_time_ms += time_ms;
        self.operations_count += 1;
    }

    /// Update memory usage for a node
    pub fn update_memory_usage(&mut self, node_rank: usize, bytes: usize) {
        self.memory_usage_per_node.insert(node_rank, bytes);
    }

    /// Calculate communication to computation ratio
    pub fn comm_compute_ratio(&self) -> f64 {
        if self.compute_time_ms == 0 {
            return 0.0;
        }
        self.comm_time_ms as f64 / self.compute_time_ms as f64
    }

    /// Calculate bandwidth utilization (bytes/ms)
    pub fn bandwidth_utilization(&self) -> f64 {
        if self.comm_time_ms == 0 {
            return 0.0;
        }
        self.bytes_transferred as f64 / self.comm_time_ms as f64
    }
}