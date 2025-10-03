//! Fault tolerance and worker health monitoring for distributed clustering
//!
//! This module provides comprehensive fault tolerance mechanisms including
//! worker failure detection, recovery strategies, and health monitoring.

use scirs2_core::ndarray::Array2;
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::message_passing::RecoveryStrategy;
use crate::error::{ClusteringError, Result};

/// Worker health monitoring and fault tolerance coordinator
#[derive(Debug)]
pub struct FaultToleranceCoordinator<F: Float> {
    pub worker_health: HashMap<usize, WorkerHealthInfo>,
    pub failed_workers: Vec<usize>,
    pub fault_config: FaultToleranceConfig,
    pub checkpoints: Vec<ClusteringCheckpoint<F>>,
    pub replication_map: HashMap<usize, Vec<usize>>, // workerid -> replica_workers
}

/// Worker health information
#[derive(Debug, Clone)]
pub struct WorkerHealthInfo {
    pub workerid: usize,
    pub status: WorkerStatus,
    pub last_heartbeat: u64,
    pub consecutive_failures: u32,
    pub total_failures: u32,
    pub response_times: Vec<u64>,
    pub cpu_usage_history: Vec<f64>,
    pub memory_usage_history: Vec<f64>,
    pub throughput_history: Vec<f64>,
}

/// Worker status enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkerStatus {
    Healthy,
    Degraded,
    Failed,
    Recovering,
    Unknown,
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    pub enabled: bool,
    pub max_failures: usize,
    pub worker_timeout_ms: u64,
    pub heartbeat_interval_ms: u64,
    pub recovery_strategy: RecoveryStrategy,
    pub enable_checkpointing: bool,
    pub checkpoint_interval: usize,
    pub enable_replication: bool,
    pub replication_factor: usize,
    pub auto_replace_workers: bool,
    pub health_check_interval_ms: u64,
    pub degraded_threshold: f64,
    pub failed_threshold: f64,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_failures: 2,
            worker_timeout_ms: 30000,
            heartbeat_interval_ms: 5000,
            recovery_strategy: RecoveryStrategy::Redistribute,
            enable_checkpointing: true,
            checkpoint_interval: 10,
            enable_replication: false,
            replication_factor: 2,
            auto_replace_workers: false,
            health_check_interval_ms: 10000,
            degraded_threshold: 0.7,
            failed_threshold: 0.3,
        }
    }
}

/// Clustering checkpoint for recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringCheckpoint<F: Float> {
    pub iteration: usize,
    pub centroids: Option<Array2<F>>,
    pub global_inertia: f64,
    pub convergence_history: Vec<ConvergenceMetrics>,
    pub worker_assignments: HashMap<usize, Vec<usize>>,
    pub timestamp: u64,
}

/// Convergence metrics for checkpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    pub iteration: usize,
    pub inertia: f64,
    pub centroid_movement: f64,
    pub worker_efficiency: f64,
    pub timestamp: u64,
}

/// Data partition for redistribution
#[derive(Debug, Clone)]
pub struct DataPartition<F: Float> {
    pub partition_id: usize,
    pub data: Array2<F>,
    pub labels: Option<Vec<usize>>,
    pub workerid: usize,
    pub weight: f64,
}

impl<F: Float> DataPartition<F> {
    pub fn new(partition_id: usize, data: Array2<F>, workerid: usize) -> Self {
        let weight = 1.0; // Default weight
        Self {
            partition_id,
            data,
            labels: None,
            workerid,
            weight,
        }
    }
}

impl<F: Float + Debug> FaultToleranceCoordinator<F> {
    /// Create new fault tolerance coordinator
    pub fn new(config: FaultToleranceConfig) -> Self {
        Self {
            worker_health: HashMap::new(),
            failed_workers: Vec::new(),
            fault_config: config,
            checkpoints: Vec::new(),
            replication_map: HashMap::new(),
        }
    }

    /// Register a new worker for health monitoring
    pub fn register_worker(&mut self, workerid: usize) {
        let health_info = WorkerHealthInfo {
            workerid,
            status: WorkerStatus::Healthy,
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            consecutive_failures: 0,
            total_failures: 0,
            response_times: Vec::new(),
            cpu_usage_history: Vec::new(),
            memory_usage_history: Vec::new(),
            throughput_history: Vec::new(),
        };

        self.worker_health.insert(workerid, health_info);
    }

    /// Update worker heartbeat
    pub fn update_heartbeat(
        &mut self,
        workerid: usize,
        cpu_usage: f64,
        memory_usage: f64,
        response_time_ms: u64,
    ) -> Result<()> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        if let Some(health) = self.worker_health.get_mut(&workerid) {
            health.last_heartbeat = current_time;
            health.cpu_usage_history.push(cpu_usage);
            health.memory_usage_history.push(memory_usage);
            health.response_times.push(response_time_ms);

            // Keep history manageable
            if health.cpu_usage_history.len() > 100 {
                health.cpu_usage_history.remove(0);
            }
            if health.memory_usage_history.len() > 100 {
                health.memory_usage_history.remove(0);
            }
            if health.response_times.len() > 100 {
                health.response_times.remove(0);
            }

            // Update worker status based on metrics
            self.update_worker_status(workerid)?;
        } else {
            return Err(ClusteringError::InvalidInput(format!(
                "Worker {} not registered",
                workerid
            )));
        }

        Ok(())
    }

    /// Check all workers for failures and update status
    pub fn check_worker_health(&mut self) -> Vec<usize> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut newly_failed_workers = Vec::new();

        for (&workerid, health) in &mut self.worker_health {
            let time_since_heartbeat = current_time.saturating_sub(health.last_heartbeat);

            if time_since_heartbeat > self.fault_config.worker_timeout_ms
                && health.status != WorkerStatus::Failed
            {
                health.status = WorkerStatus::Failed;
                health.consecutive_failures += 1;
                health.total_failures += 1;
                newly_failed_workers.push(workerid);

                if !self.failed_workers.contains(&workerid) {
                    self.failed_workers.push(workerid);
                }
            }
        }

        newly_failed_workers
    }

    /// Update worker status based on performance metrics
    fn update_worker_status(&mut self, workerid: usize) -> Result<()> {
        let performance_score = if let Some(health) = self.worker_health.get(&workerid) {
            self.calculate_performance_score(health)
        } else {
            return Ok(());
        };

        if let Some(health) = self.worker_health.get_mut(&workerid) {
            let new_status = if performance_score >= self.fault_config.degraded_threshold {
                WorkerStatus::Healthy
            } else if performance_score >= self.fault_config.failed_threshold {
                WorkerStatus::Degraded
            } else {
                WorkerStatus::Failed
            };

            // Update status and handle transitions
            if health.status != new_status {
                match (health.status, new_status) {
                    (WorkerStatus::Healthy, WorkerStatus::Failed)
                    | (WorkerStatus::Degraded, WorkerStatus::Failed) => {
                        health.consecutive_failures += 1;
                        health.total_failures += 1;
                        if !self.failed_workers.contains(&workerid) {
                            self.failed_workers.push(workerid);
                        }
                    }
                    (WorkerStatus::Failed, WorkerStatus::Healthy)
                    | (WorkerStatus::Failed, WorkerStatus::Degraded) => {
                        health.consecutive_failures = 0;
                        self.failed_workers.retain(|&id| id != workerid);
                    }
                    _ => {}
                }
                health.status = new_status;
            }
        }

        Ok(())
    }

    /// Calculate performance score for a worker
    fn calculate_performance_score(&self, health: &WorkerHealthInfo) -> f64 {
        let mut score = 1.0;

        // CPU usage component (lower is better)
        if !health.cpu_usage_history.is_empty() {
            let avg_cpu = health.cpu_usage_history.iter().sum::<f64>()
                / health.cpu_usage_history.len() as f64;
            score *= (1.0 - (avg_cpu - 0.7).max(0.0) * 2.0); // Penalty for >70% CPU
        }

        // Memory usage component (lower is better)
        if !health.memory_usage_history.is_empty() {
            let avg_memory = health.memory_usage_history.iter().sum::<f64>()
                / health.memory_usage_history.len() as f64;
            score *= (1.0 - (avg_memory - 0.8).max(0.0) * 3.0); // Penalty for >80% memory
        }

        // Response time component (lower is better)
        if !health.response_times.is_empty() {
            let avg_response = health.response_times.iter().sum::<u64>() as f64
                / health.response_times.len() as f64;
            let normalized_response = (avg_response / 1000.0).min(2.0); // Normalize to 0-2 seconds
            score *= (1.0 - normalized_response * 0.3); // Penalty for slow response
        }

        // Failure history component
        let failure_penalty = (health.consecutive_failures as f64 * 0.1).min(0.5);
        score *= 1.0 - failure_penalty;

        score.max(0.0).min(1.0)
    }

    /// Check if load balancing is needed based on worker efficiency
    pub fn should_rebalance(&self) -> bool {
        if self.worker_health.len() < 2 {
            return false;
        }

        let efficiency_scores: Vec<f64> = self
            .worker_health
            .values()
            .filter(|h| h.status != WorkerStatus::Failed)
            .map(|h| self.calculate_performance_score(h))
            .collect();

        if efficiency_scores.is_empty() {
            return false;
        }

        let best_efficiency = efficiency_scores.iter().fold(0.0, |a, &b| a.max(b));
        let worst_efficiency = efficiency_scores.iter().fold(1.0, |a, &b| a.min(b));

        // Rebalance if efficiency gap is > 30%
        (best_efficiency - worst_efficiency) > 0.3
    }

    /// Handle worker failure with configured recovery strategy
    pub fn handle_worker_failure(
        &mut self,
        failed_workerid: usize,
        partitions: &mut Vec<DataPartition<F>>,
    ) -> Result<()> {
        if !self.fault_config.enabled {
            return Ok(());
        }

        if self.failed_workers.len() > self.fault_config.max_failures {
            return Err(ClusteringError::InvalidInput(format!(
                "Too many worker failures: {} > {}",
                self.failed_workers.len(),
                self.fault_config.max_failures
            )));
        }

        match self.fault_config.recovery_strategy {
            RecoveryStrategy::Redistribute => {
                self.redistribute_failed_worker_data(failed_workerid, partitions)?;
            }
            RecoveryStrategy::Replace => {
                self.replace_failed_worker(failed_workerid)?;
            }
            RecoveryStrategy::Checkpoint => {
                self.restore_from_checkpoint()?;
            }
            RecoveryStrategy::Restart => {
                return Err(ClusteringError::InvalidInput(
                    "Restart strategy requires external coordination".to_string(),
                ));
            }
            RecoveryStrategy::Degrade => {
                // Continue with fewer workers - no action needed
            }
        }

        Ok(())
    }

    /// Redistribute data from failed worker to healthy workers
    fn redistribute_failed_worker_data(
        &mut self,
        failed_workerid: usize,
        partitions: &mut Vec<DataPartition<F>>,
    ) -> Result<()> {
        let healthy_workers: Vec<usize> = self
            .worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(&id, _)| id)
            .collect();

        if healthy_workers.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No healthy workers available for redistribution".to_string(),
            ));
        }

        // Find partitions assigned to failed worker
        let failed_partitions: Vec<usize> = partitions
            .iter()
            .enumerate()
            .filter(|(_, p)| p.workerid == failed_workerid)
            .map(|(i, _)| i)
            .collect();

        // Redistribute to healthy workers using round-robin
        for (idx, &partition_idx) in failed_partitions.iter().enumerate() {
            let new_worker = healthy_workers[idx % healthy_workers.len()];
            partitions[partition_idx].workerid = new_worker;
        }

        Ok(())
    }

    /// Replace failed worker with a new worker
    fn replace_failed_worker(&mut self, failed_workerid: usize) -> Result<()> {
        if !self.fault_config.auto_replace_workers {
            return Err(ClusteringError::InvalidInput(
                "Worker replacement is disabled".to_string(),
            ));
        }

        // In a real implementation, this would spawn a new worker process
        // For now, we'll simulate by creating a new worker ID
        let new_workerid = self.worker_health.keys().max().unwrap_or(&0) + 1;
        self.register_worker(new_workerid);

        // Mark the new worker as healthy
        if let Some(health) = self.worker_health.get_mut(&new_workerid) {
            health.mark_success();
        }

        Ok(())
    }

    /// Restore from the latest checkpoint
    fn restore_from_checkpoint(&mut self) -> Result<()> {
        if !self.fault_config.enable_checkpointing {
            return Err(ClusteringError::InvalidInput(
                "Checkpointing is disabled".to_string(),
            ));
        }

        if let Some(_latest_checkpoint) = self.checkpoints.last() {
            // In a real implementation, this would restore the clustering state
            // For now, we'll just clear the failed workers list
            self.failed_workers.clear();

            // Reset worker health
            for health in self.worker_health.values_mut() {
                if health.status == WorkerStatus::Failed {
                    health.mark_success();
                }
            }
        }

        Ok(())
    }

    /// Create a checkpoint of the current clustering state
    pub fn create_checkpoint(
        &mut self,
        iteration: usize,
        centroids: Option<&Array2<F>>,
        global_inertia: f64,
        convergence_history: &[ConvergenceMetrics],
        worker_assignments: &HashMap<usize, Vec<usize>>,
    ) {
        if !self.fault_config.enable_checkpointing {
            return;
        }

        if !iteration.is_multiple_of(self.fault_config.checkpoint_interval) {
            return;
        }

        let checkpoint = ClusteringCheckpoint {
            iteration,
            centroids: centroids.cloned(),
            global_inertia,
            convergence_history: convergence_history.to_vec(),
            worker_assignments: worker_assignments.clone(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.checkpoints.push(checkpoint);

        // Keep only recent checkpoints to save memory
        if self.checkpoints.len() > 10 {
            self.checkpoints.remove(0);
        }
    }

    /// Setup data replication for fault tolerance
    pub fn setup_replication(&mut self, partitions: &mut Vec<DataPartition<F>>) -> Result<()> {
        if !self.fault_config.enabled || !self.fault_config.enable_replication {
            return Ok(());
        }

        let replication_factor = self.fault_config.replication_factor;
        let healthy_workers: Vec<usize> = self
            .worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(&id, _)| id)
            .collect();

        if healthy_workers.len() < replication_factor {
            return Err(ClusteringError::InvalidInput(format!(
                "Not enough healthy workers for replication factor {}",
                replication_factor
            )));
        }

        // Setup replicas for each partition
        for partition in partitions.iter_mut() {
            let primary_worker = partition.workerid;
            let mut replica_workers = Vec::new();

            // Select replica workers (excluding primary)
            let available_workers: Vec<usize> = healthy_workers
                .iter()
                .filter(|&&id| id != primary_worker)
                .copied()
                .collect();

            // Select replica workers round-robin
            for i in 0..(replication_factor - 1).min(available_workers.len()) {
                let replica_worker = available_workers[i % available_workers.len()];
                replica_workers.push(replica_worker);
            }

            self.replication_map.insert(primary_worker, replica_workers);
        }

        Ok(())
    }

    /// Get healthy workers list
    pub fn get_healthy_workers(&self) -> Vec<usize> {
        self.worker_health
            .iter()
            .filter(|(_, health)| health.is_healthy(self.fault_config.worker_timeout_ms))
            .map(|(&id, _)| id)
            .collect()
    }

    /// Get failed workers list
    pub fn get_failed_workers(&self) -> &[usize] {
        &self.failed_workers
    }

    /// Get worker health status
    pub fn get_worker_status(&self, workerid: usize) -> Option<WorkerStatus> {
        self.worker_health.get(&workerid).map(|h| h.status)
    }

    /// Get comprehensive health report
    pub fn get_health_report(&self) -> HealthReport {
        let total_workers = self.worker_health.len();
        let healthy_workers = self.get_healthy_workers().len();
        let failed_workers = self.failed_workers.len();
        let degraded_workers = self
            .worker_health
            .values()
            .filter(|h| h.status == WorkerStatus::Degraded)
            .count();

        let avg_performance = if !self.worker_health.is_empty() {
            self.worker_health
                .values()
                .map(|h| self.calculate_performance_score(h))
                .sum::<f64>()
                / self.worker_health.len() as f64
        } else {
            0.0
        };

        HealthReport {
            total_workers,
            healthy_workers,
            degraded_workers,
            failed_workers,
            avg_performance_score: avg_performance,
            replication_enabled: self.fault_config.enable_replication,
            checkpointing_enabled: self.fault_config.enable_checkpointing,
            last_checkpoint: self.checkpoints.last().map(|c| c.timestamp),
        }
    }
}

impl WorkerHealthInfo {
    /// Check if worker is healthy based on heartbeat timeout
    pub fn is_healthy(&self, timeoutms: u64) -> bool {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let time_since_heartbeat = current_time.saturating_sub(self.last_heartbeat);
        self.status != WorkerStatus::Failed && time_since_heartbeat <= timeoutms
    }

    /// Mark worker as successful (reset failure counters)
    pub fn mark_success(&mut self) {
        self.status = WorkerStatus::Healthy;
        self.consecutive_failures = 0;
        self.last_heartbeat = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
    }

    /// Get average response time
    pub fn avg_response_time(&self) -> f64 {
        if self.response_times.is_empty() {
            0.0
        } else {
            self.response_times.iter().sum::<u64>() as f64 / self.response_times.len() as f64
        }
    }

    /// Get average CPU usage
    pub fn avg_cpu_usage(&self) -> f64 {
        if self.cpu_usage_history.is_empty() {
            0.0
        } else {
            self.cpu_usage_history.iter().sum::<f64>() / self.cpu_usage_history.len() as f64
        }
    }

    /// Get average memory usage
    pub fn avg_memory_usage(&self) -> f64 {
        if self.memory_usage_history.is_empty() {
            0.0
        } else {
            self.memory_usage_history.iter().sum::<f64>() / self.memory_usage_history.len() as f64
        }
    }
}

/// Comprehensive health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub total_workers: usize,
    pub healthy_workers: usize,
    pub degraded_workers: usize,
    pub failed_workers: usize,
    pub avg_performance_score: f64,
    pub replication_enabled: bool,
    pub checkpointing_enabled: bool,
    pub last_checkpoint: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fault_tolerance_coordinator_creation() {
        let config = FaultToleranceConfig::default();
        let coordinator = FaultToleranceCoordinator::<f64>::new(config);

        assert!(coordinator.worker_health.is_empty());
        assert!(coordinator.failed_workers.is_empty());
        assert!(coordinator.fault_config.enabled);
    }

    #[test]
    fn test_worker_registration() {
        let config = FaultToleranceConfig::default();
        let mut coordinator = FaultToleranceCoordinator::<f64>::new(config);

        coordinator.register_worker(1);
        assert!(coordinator.worker_health.contains_key(&1));
        assert_eq!(
            coordinator.get_worker_status(1),
            Some(WorkerStatus::Healthy)
        );
    }

    #[test]
    fn test_worker_health_info() {
        let mut health = WorkerHealthInfo {
            workerid: 1,
            status: WorkerStatus::Healthy,
            last_heartbeat: 0,
            consecutive_failures: 0,
            total_failures: 0,
            response_times: vec![100, 200, 150],
            cpu_usage_history: vec![0.5, 0.6, 0.4],
            memory_usage_history: vec![0.3, 0.4, 0.2],
            throughput_history: Vec::new(),
        };

        assert_eq!(health.avg_response_time(), 150.0);
        assert_eq!(health.avg_cpu_usage(), 0.5);
        assert_eq!(health.avg_memory_usage(), 0.3);

        health.mark_success();
        assert_eq!(health.status, WorkerStatus::Healthy);
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn test_health_report() {
        let config = FaultToleranceConfig::default();
        let mut coordinator = FaultToleranceCoordinator::<f64>::new(config);

        coordinator.register_worker(1);
        coordinator.register_worker(2);

        let report = coordinator.get_health_report();
        assert_eq!(report.total_workers, 2);
        assert_eq!(report.healthy_workers, 2);
        assert_eq!(report.failed_workers, 0);
    }
}
