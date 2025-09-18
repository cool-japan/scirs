//! Monitoring and statistics collection
//!
//! This module handles cluster statistics, performance monitoring, and telemetry
//! for the distributed computing framework.

use super::types::default_instant;
use std::time::{Duration, Instant};

/// Cluster statistics
#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    /// Total nodes
    pub total_nodes: usize,
    /// Active nodes
    pub active_nodes: usize,
    /// Total tasks processed
    pub total_tasks_processed: u64,
    /// Average task completion time
    pub avg_task_completion_time: Duration,
    /// Cluster throughput
    pub cluster_throughput: f64,
    /// Resource utilization
    pub resource_utilization: ClusterResourceUtilization,
    /// Fault tolerance metrics
    pub fault_tolerance_metrics: FaultToleranceMetrics,
    /// Tasks submitted
    pub tasks_submitted: u64,
    /// Average submission time
    pub avg_submission_time: Duration,
    /// Last update timestamp
    pub last_update: Instant,
}

/// Cluster resource utilization
#[derive(Debug, Clone)]
pub struct ClusterResourceUtilization {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Storage utilization
    pub storage_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
}

/// Fault tolerance metrics
#[derive(Debug, Clone)]
pub struct FaultToleranceMetrics {
    /// Mean time between failures
    pub mtbf: Duration,
    /// Mean time to recovery
    pub mttr: Duration,
    /// Availability percentage
    pub availability: f64,
    /// Successful recoveries
    pub successful_recoveries: u64,
}

// Implementations
impl Default for ClusterStatistics {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            active_nodes: 0,
            total_tasks_processed: 0,
            avg_task_completion_time: Duration::default(),
            cluster_throughput: 0.0,
            resource_utilization: ClusterResourceUtilization {
                cpu_utilization: 0.0,
                memory_utilization: 0.0,
                storage_utilization: 0.0,
                network_utilization: 0.0,
            },
            fault_tolerance_metrics: FaultToleranceMetrics {
                mtbf: Duration::from_secs(168 * 60 * 60), // 1 week
                mttr: Duration::from_secs(15 * 60),
                availability: 0.999,
                successful_recoveries: 0,
            },
            tasks_submitted: 0,
            avg_submission_time: Duration::default(),
            last_update: default_instant(),
        }
    }
}
