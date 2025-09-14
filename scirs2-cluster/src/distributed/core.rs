//! Core distributed K-means clustering implementation
//!
//! This module provides the main distributed K-means algorithm with
//! support for multiple workers, fault tolerance, and load balancing.

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num_traits::{Float, FromPrimitive, Zero};
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{ClusteringError, Result};
use crate::vq::euclidean_distance;

use super::fault_tolerance::{DataPartition, FaultToleranceCoordinator};
use super::load_balancing::LoadBalancingCoordinator;
use super::message_passing::{ClusteringMessage, MessagePassingCoordinator, MessagePriority};
use super::monitoring::PerformanceMonitor;
use super::partitioning::{DataPartitioner, PartitioningConfig};

/// Main distributed K-means clustering algorithm
#[derive(Debug)]
pub struct DistributedKMeans<F: Float> {
    /// Number of clusters
    pub k: usize,
    /// Configuration parameters
    pub config: DistributedKMeansConfig,
    /// Current centroids
    pub centroids: Option<Array2<F>>,
    /// Worker assignments and data partitions
    pub partitions: Vec<DataPartition<F>>,
    /// Fault tolerance coordinator
    pub fault_coordinator: FaultToleranceCoordinator<F>,
    /// Load balancing coordinator
    pub load_balancer: LoadBalancingCoordinator,
    /// Performance monitor
    pub performance_monitor: PerformanceMonitor,
    /// Message passing coordinator
    pub message_coordinator: Option<MessagePassingCoordinator<F>>,
    /// Data partitioner
    pub partitioner: DataPartitioner<F>,
    /// Current iteration
    pub current_iteration: usize,
    /// Convergence history
    pub convergence_history: Vec<ConvergenceInfo>,
    /// Global inertia
    pub global_inertia: f64,
}

/// Configuration for distributed K-means
#[derive(Debug, Clone)]
pub struct DistributedKMeansConfig {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub n_workers: usize,
    pub init_method: InitializationMethod,
    pub enable_fault_tolerance: bool,
    pub enable_load_balancing: bool,
    pub enable_monitoring: bool,
    pub convergence_check_interval: usize,
    pub checkpoint_interval: usize,
    pub verbose: bool,
    pub random_seed: Option<u64>,
}

impl Default for DistributedKMeansConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-4,
            n_workers: 4,
            init_method: InitializationMethod::KMeansPlusPlus,
            enable_fault_tolerance: true,
            enable_load_balancing: true,
            enable_monitoring: true,
            convergence_check_interval: 5,
            checkpoint_interval: 10,
            verbose: false,
            random_seed: None,
        }
    }
}

/// Centroid initialization methods
#[derive(Debug, Clone)]
pub enum InitializationMethod {
    /// Random initialization
    Random,
    /// K-means++ initialization
    KMeansPlusPlus,
    /// Forgy initialization
    Forgy,
    /// Custom centroids provided by user
    Custom(Array2<f64>),
}

/// Convergence information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceInfo {
    pub iteration: usize,
    pub inertia: f64,
    pub centroid_movement: f64,
    pub converged: bool,
    pub timestamp: SystemTime,
    pub computation_time_ms: u64,
}

/// Clustering result
#[derive(Debug, Clone)]
pub struct ClusteringResult<F: Float> {
    /// Final cluster centroids
    pub centroids: Array2<F>,
    /// Cluster labels for all data points
    pub labels: Array1<usize>,
    /// Final inertia (within-cluster sum of squares)
    pub inertia: f64,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Convergence information
    pub convergence_info: ConvergenceInfo,
    /// Performance statistics
    pub performance_stats: PerformanceStatistics,
}

/// Performance statistics for clustering
#[derive(Debug, Clone)]
pub struct PerformanceStatistics {
    pub total_time_ms: u64,
    pub communication_time_ms: u64,
    pub computation_time_ms: u64,
    pub synchronization_time_ms: u64,
    pub worker_efficiency: f64,
    pub load_balance_score: f64,
    pub fault_tolerance_events: usize,
}

/// Worker computation result
#[derive(Debug, Clone)]
pub struct WorkerResult<F: Float> {
    pub worker_id: usize,
    pub local_centroids: Array2<F>,
    pub local_labels: Array1<usize>,
    pub local_inertia: f64,
    pub point_counts: Array1<usize>,
    pub computation_time_ms: u64,
}

impl<F: Float + FromPrimitive + Debug + Send + Sync + 'static> DistributedKMeans<F> {
    /// Create new distributed K-means instance
    pub fn new(k: usize, config: DistributedKMeansConfig) -> Result<Self> {
        if k == 0 {
            return Err(ClusteringError::InvalidInput(
                "Number of clusters must be greater than 0".to_string(),
            ));
        }

        if config.n_workers == 0 {
            return Err(ClusteringError::InvalidInput(
                "Number of workers must be greater than 0".to_string(),
            ));
        }

        let partitioner_config = PartitioningConfig {
            n_workers: config.n_workers,
            ..Default::default()
        };

        let fault_tolerance_config = super::fault_tolerance::FaultToleranceConfig {
            enabled: config.enable_fault_tolerance,
            ..Default::default()
        };

        let load_balancing_config = super::load_balancing::LoadBalancingConfig {
            enable_dynamic_balancing: config.enable_load_balancing,
            ..Default::default()
        };

        let monitoring_config = super::monitoring::MonitoringConfig {
            enable_detailed_monitoring: config.enable_monitoring,
            ..Default::default()
        };

        Ok(Self {
            k,
            config,
            centroids: None,
            partitions: Vec::new(),
            fault_coordinator: FaultToleranceCoordinator::new(fault_tolerance_config),
            load_balancer: LoadBalancingCoordinator::new(load_balancing_config),
            performance_monitor: PerformanceMonitor::new(monitoring_config),
            message_coordinator: None,
            partitioner: DataPartitioner::new(partitioner_config),
            current_iteration: 0,
            convergence_history: Vec::new(),
            global_inertia: f64::INFINITY,
        })
    }

    /// Fit the distributed K-means model to data
    pub fn fit(&mut self, data: ArrayView2<F>) -> Result<ClusteringResult<F>> {
        let start_time = Instant::now();
        let mut stats = PerformanceStatistics {
            total_time_ms: 0,
            communication_time_ms: 0,
            computation_time_ms: 0,
            synchronization_time_ms: 0,
            worker_efficiency: 0.0,
            load_balance_score: 0.0,
            fault_tolerance_events: 0,
        };

        // Validate input data
        self.validate_input(data)?;

        // Initialize workers and message passing
        self.initialize_workers()?;

        // Partition data across workers
        let partition_start = Instant::now();
        self.partitions = self.partitioner.partition_data(data)?;
        stats.communication_time_ms += partition_start.elapsed().as_millis() as u64;

        if self.config.verbose {
            println!("Data partitioned across {} workers", self.config.n_workers);
        }

        // Initialize centroids
        let init_start = Instant::now();
        self.centroids = Some(self.initialize_centroids(data)?);
        stats.computation_time_ms += init_start.elapsed().as_millis() as u64;

        // Main clustering loop
        let mut converged = false;
        self.current_iteration = 0;

        while self.current_iteration < self.config.max_iterations && !converged {
            let iteration_start = Instant::now();

            // Perform one iteration of distributed K-means
            converged = self.perform_iteration(&mut stats)?;

            // Update convergence history
            let iteration_time = iteration_start.elapsed().as_millis() as u64;
            self.update_convergence_history(iteration_time)?;

            // Check for rebalancing if needed
            if self.config.enable_load_balancing && self.current_iteration % 10 == 0 {
                self.check_and_rebalance(data, &mut stats)?;
            }

            // Create checkpoint if configured
            if self.config.enable_fault_tolerance
                && self.current_iteration % self.config.checkpoint_interval == 0
            {
                self.create_checkpoint()?;
            }

            self.current_iteration += 1;

            if self.config.verbose && self.current_iteration % 10 == 0 {
                println!(
                    "Iteration {}: inertia = {:.6}",
                    self.current_iteration, self.global_inertia
                );
            }
        }

        // Finalize results
        stats.total_time_ms = start_time.elapsed().as_millis() as u64;
        stats.worker_efficiency = self.calculate_worker_efficiency();
        stats.load_balance_score = self.calculate_load_balance_score();

        let final_labels = self.collect_final_labels()?;
        let final_convergence =
            self.convergence_history
                .last()
                .cloned()
                .unwrap_or_else(|| ConvergenceInfo {
                    iteration: self.current_iteration,
                    inertia: self.global_inertia,
                    centroid_movement: 0.0,
                    converged,
                    timestamp: SystemTime::now(),
                    computation_time_ms: 0,
                });

        Ok(ClusteringResult {
            centroids: self.centroids.as_ref().unwrap().clone(),
            labels: final_labels,
            inertia: self.global_inertia,
            n_iterations: self.current_iteration,
            convergence_info: final_convergence,
            performance_stats: stats,
        })
    }

    /// Validate input data
    fn validate_input(&self, data: ArrayView2<F>) -> Result<()> {
        if data.nrows() == 0 {
            return Err(ClusteringError::InvalidInput(
                "Input data is empty".to_string(),
            ));
        }

        if data.ncols() == 0 {
            return Err(ClusteringError::InvalidInput(
                "Input data has no features".to_string(),
            ));
        }

        if data.nrows() < self.k {
            return Err(ClusteringError::InvalidInput(format!(
                "Number of samples ({}) must be at least k ({})",
                data.nrows(),
                self.k
            )));
        }

        if data.nrows() < self.config.n_workers {
            return Err(ClusteringError::InvalidInput(format!(
                "Number of samples ({}) must be at least number of workers ({})",
                data.nrows(),
                self.config.n_workers
            )));
        }

        Ok(())
    }

    /// Initialize workers and communication infrastructure
    fn initialize_workers(&mut self) -> Result<()> {
        // Register workers with fault tolerance coordinator
        for worker_id in 0..self.config.n_workers {
            self.fault_coordinator.register_worker(worker_id);
            self.performance_monitor.register_worker(worker_id);
        }

        // Initialize message passing coordinator if needed
        if self.config.n_workers > 1 {
            let message_config = super::message_passing::MessagePassingConfig::default();
            self.message_coordinator = Some(MessagePassingCoordinator::new(0, message_config));
        }

        Ok(())
    }

    /// Initialize cluster centroids
    fn initialize_centroids(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        match &self.config.init_method {
            InitializationMethod::Random => self.random_initialization(data),
            InitializationMethod::KMeansPlusPlus => self.kmeans_plus_plus_initialization(data),
            InitializationMethod::Forgy => self.forgy_initialization(data),
            InitializationMethod::Custom(centroids) => {
                if centroids.nrows() != self.k || centroids.ncols() != data.ncols() {
                    return Err(ClusteringError::InvalidInput(
                        "Custom centroids dimensions don't match".to_string(),
                    ));
                }
                let converted_centroids =
                    Array2::from_shape_fn((self.k, data.ncols()), |(i, j)| {
                        F::from(centroids[[i, j]]).unwrap_or_else(F::zero)
                    });
                Ok(converted_centroids)
            }
        }
    }

    /// Random centroid initialization
    fn random_initialization(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        use rand::seq::SliceRandom;

        let mut rng = rand::rng();
        let data_indices: Vec<usize> = (0..data.nrows()).collect();
        let selected_indices: Vec<_> = data_indices
            .as_slice()
            .choose_multiple(&mut rng, self.k)
            .cloned()
            .collect();

        let mut centroids = Array2::zeros((self.k, data.ncols()));
        for (i, &data_idx) in selected_indices.iter().enumerate() {
            centroids.row_mut(i).assign(&data.row(data_idx));
        }

        Ok(centroids)
    }

    /// K-means++ centroid initialization
    fn kmeans_plus_plus_initialization(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        use rand::Rng;

        let mut rng = rand::rng();
        let mut centroids = Array2::zeros((self.k, data.ncols()));

        // Choose first centroid randomly
        let first_idx = rng.random_range(0..data.nrows());
        centroids.row_mut(0).assign(&data.row(first_idx));

        // Choose remaining centroids using K-means++ method
        for k in 1..self.k {
            let mut distances = Array1::zeros(data.nrows());

            // Calculate distance to nearest centroid for each point
            for (i, point) in data.rows().into_iter().enumerate() {
                let mut min_dist = F::infinity();
                for centroid in centroids.rows().into_iter().take(k) {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                    }
                }
                distances[i] = min_dist.to_f64().unwrap_or(f64::INFINITY);
            }

            // Choose next centroid with probability proportional to squared distance
            let total_dist: f64 = distances.iter().map(|&d| d * d).sum();
            if total_dist <= 0.0 {
                // Fallback to random selection
                let random_idx = rng.random_range(0..data.nrows());
                centroids.row_mut(k).assign(&data.row(random_idx));
            } else {
                let mut cumulative = 0.0;
                let threshold = rng.random::<f64>() * total_dist;

                let mut selected_idx = 0;
                for (i, &dist) in distances.iter().enumerate() {
                    cumulative += dist * dist;
                    if cumulative >= threshold {
                        selected_idx = i;
                        break;
                    }
                }
                centroids.row_mut(k).assign(&data.row(selected_idx));
            }
        }

        Ok(centroids)
    }

    /// Forgy centroid initialization
    fn forgy_initialization(&self, data: ArrayView2<F>) -> Result<Array2<F>> {
        // Forgy method is equivalent to random initialization
        self.random_initialization(data)
    }

    /// Perform one iteration of distributed K-means
    fn perform_iteration(&mut self, stats: &mut PerformanceStatistics) -> Result<bool> {
        let iteration_start = Instant::now();

        // Broadcast current centroids to all workers
        if self.config.n_workers > 1 {
            let broadcast_start = Instant::now();
            self.broadcast_centroids()?;
            stats.communication_time_ms += broadcast_start.elapsed().as_millis() as u64;
        }

        // Compute local assignments and centroids on each worker
        let compute_start = Instant::now();
        let worker_results = self.compute_worker_assignments()?;
        stats.computation_time_ms += compute_start.elapsed().as_millis() as u64;

        // Synchronize and aggregate results
        let sync_start = Instant::now();
        let (new_centroids, new_inertia) = self.aggregate_worker_results(&worker_results)?;
        stats.synchronization_time_ms += sync_start.elapsed().as_millis() as u64;

        // Check for convergence
        let converged = self.check_convergence(&new_centroids, new_inertia)?;

        // Update centroids and inertia
        self.centroids = Some(new_centroids);
        self.global_inertia = new_inertia;

        Ok(converged)
    }

    /// Broadcast current centroids to all workers
    fn broadcast_centroids(&mut self) -> Result<()> {
        if let (Some(ref centroids), Some(ref mut coordinator)) =
            (&self.centroids, &mut self.message_coordinator)
        {
            let message = ClusteringMessage::UpdateCentroids {
                round: self.current_iteration,
                centroids: centroids.clone(),
            };

            coordinator.broadcast_message(message, MessagePriority::Normal)?;
        }

        Ok(())
    }

    /// Compute assignments and local centroids on each worker
    fn compute_worker_assignments(&mut self) -> Result<Vec<WorkerResult<F>>> {
        let mut results = Vec::new();

        if let Some(ref centroids) = self.centroids {
            for partition in &self.partitions {
                let worker_start = Instant::now();

                // Assign points to nearest centroids
                let mut labels = Array1::zeros(partition.data.nrows());
                let mut local_inertia = F::zero();

                for (i, point) in partition.data.rows().into_iter().enumerate() {
                    let mut min_dist = F::infinity();
                    let mut best_cluster = 0;

                    for (j, centroid) in centroids.rows().into_iter().enumerate() {
                        let dist = euclidean_distance(point, centroid);
                        if dist < min_dist {
                            min_dist = dist;
                            best_cluster = j;
                        }
                    }

                    labels[i] = best_cluster;
                    local_inertia = local_inertia + min_dist * min_dist;
                }

                // Compute local centroids
                let mut local_centroids = Array2::zeros((self.k, partition.data.ncols()));
                let mut point_counts = Array1::zeros(self.k);

                for (i, point) in partition.data.rows().into_iter().enumerate() {
                    let cluster = labels[i];
                    point_counts[cluster] += 1;

                    for (j, &value) in point.iter().enumerate() {
                        local_centroids[[cluster, j]] = local_centroids[[cluster, j]] + value;
                    }
                }

                // Normalize to get means
                for k in 0..self.k {
                    if point_counts[k] > 0 {
                        let count = F::from(point_counts[k]).unwrap();
                        for j in 0..partition.data.ncols() {
                            local_centroids[[k, j]] = local_centroids[[k, j]] / count;
                        }
                    }
                }

                let computation_time = worker_start.elapsed().as_millis() as u64;

                results.push(WorkerResult {
                    worker_id: partition.workerid,
                    local_centroids,
                    local_labels: labels,
                    local_inertia: local_inertia.to_f64().unwrap_or(f64::INFINITY),
                    point_counts,
                    computation_time_ms: computation_time,
                });

                // Update worker performance metrics
                let throughput = partition.data.nrows() as f64 / (computation_time as f64 / 1000.0);
                let efficiency = 1.0 / (1.0 + computation_time as f64 / 10000.0); // Simplified efficiency
                self.performance_monitor.update_worker_metrics(
                    partition.workerid,
                    0.5, // CPU usage (placeholder)
                    0.4, // Memory usage (placeholder)
                    throughput,
                    computation_time as f64,
                )?;
            }
        }

        Ok(results)
    }

    /// Aggregate results from all workers
    fn aggregate_worker_results(
        &self,
        worker_results: &[WorkerResult<F>],
    ) -> Result<(Array2<F>, f64)> {
        if worker_results.is_empty() {
            return Err(ClusteringError::InvalidInput(
                "No worker results to aggregate".to_string(),
            ));
        }

        let n_features = worker_results[0].local_centroids.ncols();
        let mut global_centroids = Array2::zeros((self.k, n_features));
        let mut global_counts: Array1<usize> = Array1::zeros(self.k);
        let mut global_inertia = 0.0;

        // Aggregate weighted centroids and counts
        for result in worker_results {
            global_inertia += result.local_inertia;

            for k in 0..self.k {
                let count = F::from(result.point_counts[k]).unwrap();
                global_counts[k] += result.point_counts[k];

                for j in 0..n_features {
                    global_centroids[[k, j]] =
                        global_centroids[[k, j]] + result.local_centroids[[k, j]] * count;
                }
            }
        }

        // Normalize to get global means
        for k in 0..self.k {
            if global_counts[k] > 0 {
                let count = F::from(global_counts[k]).unwrap();
                for j in 0..n_features {
                    global_centroids[[k, j]] = global_centroids[[k, j]] / count;
                }
            }
        }

        Ok((global_centroids, global_inertia))
    }

    /// Check for convergence
    fn check_convergence(&self, new_centroids: &Array2<F>, newinertia: f64) -> Result<bool> {
        if let Some(ref old_centroids) = self.centroids {
            // Calculate centroid movement
            let mut max_movement = F::zero();
            for (old_row, new_row) in old_centroids.rows().into_iter().zip(new_centroids.rows()) {
                let movement = euclidean_distance(old_row, new_row);
                if movement > max_movement {
                    max_movement = movement;
                }
            }

            // Check convergence criteria
            let movement_converged =
                max_movement.to_f64().unwrap_or(f64::INFINITY) < self.config.tolerance;
            let inertia_change = (self.global_inertia - newinertia).abs();
            let inertia_converged =
                inertia_change < self.config.tolerance * self.global_inertia.abs();

            Ok(movement_converged || inertia_converged)
        } else {
            Ok(false)
        }
    }

    /// Update convergence history
    fn update_convergence_history(&mut self, iteration_timems: u64) -> Result<()> {
        let centroid_movement = if let Some(ref centroids) = self.centroids {
            if self.convergence_history.is_empty() {
                0.0
            } else {
                // Calculate movement from previous iteration (simplified)
                self.config.tolerance * 2.0 // Placeholder
            }
        } else {
            0.0
        };

        let converged = self.current_iteration >= self.config.max_iterations
            || centroid_movement < self.config.tolerance;

        let convergence_info = ConvergenceInfo {
            iteration: self.current_iteration,
            inertia: self.global_inertia,
            centroid_movement,
            converged,
            timestamp: SystemTime::now(),
            computation_time_ms: iteration_timems,
        };

        self.convergence_history.push(convergence_info);

        Ok(())
    }

    /// Check for load imbalance and rebalance if needed
    fn check_and_rebalance(
        &mut self,
        data: ArrayView2<F>,
        stats: &mut PerformanceStatistics,
    ) -> Result<()> {
        if !self.config.enable_load_balancing {
            return Ok(());
        }

        // Check if rebalancing is needed
        if self.fault_coordinator.should_rebalance() {
            let rebalance_start = Instant::now();

            // Re-partition data
            self.partitions = self.partitioner.partition_data(data)?;

            stats.communication_time_ms += rebalance_start.elapsed().as_millis() as u64;
            stats.fault_tolerance_events += 1;

            if self.config.verbose {
                println!(
                    "Load rebalancing performed at iteration {}",
                    self.current_iteration
                );
            }
        }

        Ok(())
    }

    /// Create checkpoint for fault tolerance
    fn create_checkpoint(&mut self) -> Result<()> {
        if !self.config.enable_fault_tolerance {
            return Ok(());
        }

        let worker_assignments = self
            .partitions
            .iter()
            .map(|p| (p.workerid, vec![p.partition_id]))
            .collect();

        self.fault_coordinator.create_checkpoint(
            self.current_iteration,
            self.centroids.as_ref(),
            self.global_inertia,
            &[], // Convergence history (simplified)
            &worker_assignments,
        );

        Ok(())
    }

    /// Calculate worker efficiency
    fn calculate_worker_efficiency(&self) -> f64 {
        let worker_metrics = self.performance_monitor.get_worker_metrics();
        if worker_metrics.is_empty() {
            return 0.0;
        }

        let avg_health_score = worker_metrics.values().map(|m| m.health_score).sum::<f64>()
            / worker_metrics.len() as f64;

        avg_health_score
    }

    /// Calculate load balance score
    fn calculate_load_balance_score(&self) -> f64 {
        if self.partitions.is_empty() {
            return 1.0;
        }

        let partition_sizes: Vec<usize> = self.partitions.iter().map(|p| p.data.nrows()).collect();
        let avg_size = partition_sizes.iter().sum::<usize>() as f64 / partition_sizes.len() as f64;

        if avg_size == 0.0 {
            return 1.0;
        }

        let variance = partition_sizes
            .iter()
            .map(|&size| (size as f64 - avg_size).powi(2))
            .sum::<f64>()
            / partition_sizes.len() as f64;

        let coefficient_of_variation = variance.sqrt() / avg_size;
        1.0 / (1.0 + coefficient_of_variation)
    }

    /// Collect final labels from all partitions
    fn collect_final_labels(&self) -> Result<Array1<usize>> {
        let total_points: usize = self.partitions.iter().map(|p| p.data.nrows()).sum();
        let mut labels = Array1::zeros(total_points);
        let mut offset = 0;

        // This is a simplified version - in practice, we'd need to track
        // original data point indices through the partitioning process
        for partition in &self.partitions {
            if let Some(ref partition_labels) = partition.labels {
                let end_offset = offset + partition_labels.len();
                labels
                    .slice_mut(s![offset..end_offset])
                    .assign(&Array1::from_vec(partition_labels.clone()).view());
                offset = end_offset;
            }
        }

        Ok(labels)
    }

    /// Predict cluster assignments for new data
    pub fn predict(&self, data: ArrayView2<F>) -> Result<Array1<usize>> {
        if let Some(ref centroids) = self.centroids {
            let mut labels = Array1::zeros(data.nrows());

            for (i, point) in data.rows().into_iter().enumerate() {
                let mut min_dist = F::infinity();
                let mut best_cluster = 0;

                for (j, centroid) in centroids.rows().into_iter().enumerate() {
                    let dist = euclidean_distance(point, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }

                labels[i] = best_cluster;
            }

            Ok(labels)
        } else {
            Err(ClusteringError::InvalidInput(
                "Model has not been fitted yet".to_string(),
            ))
        }
    }

    /// Get current centroids
    pub fn centroids(&self) -> Option<&Array2<F>> {
        self.centroids.as_ref()
    }

    /// Get convergence history
    pub fn convergence_history(&self) -> &[ConvergenceInfo] {
        &self.convergence_history
    }

    /// Get current inertia
    pub fn inertia(&self) -> f64 {
        self.global_inertia
    }

    /// Get number of iterations performed
    pub fn n_iterations(&self) -> usize {
        self.current_iteration
    }

    /// Get performance monitor
    pub fn performance_monitor(&self) -> &PerformanceMonitor {
        &self.performance_monitor
    }

    /// Get fault tolerance coordinator
    pub fn fault_coordinator(&self) -> &FaultToleranceCoordinator<F> {
        &self.fault_coordinator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    #[test]
    fn test_distributed_kmeans_creation() {
        let config = DistributedKMeansConfig::default();
        let kmeans = DistributedKMeans::<f64>::new(3, config);

        assert!(kmeans.is_ok());
        let kmeans = kmeans.unwrap();
        assert_eq!(kmeans.k, 3);
        assert!(kmeans.centroids.is_none());
    }

    #[test]
    fn test_input_validation() {
        let config = DistributedKMeansConfig::default();
        let kmeans = DistributedKMeans::<f64>::new(3, config).unwrap();

        // Empty data
        let empty_data = Array2::<f64>::zeros((0, 2));
        assert!(kmeans.validate_input(empty_data.view()).is_err());

        // Too few samples
        let small_data = Array2::<f64>::zeros((2, 2));
        assert!(kmeans.validate_input(small_data.view()).is_err());

        // Valid data
        let valid_data = Array2::<f64>::zeros((10, 2));
        assert!(kmeans.validate_input(valid_data.view()).is_ok());
    }

    #[test]
    fn test_random_initialization() {
        let config = DistributedKMeansConfig::default();
        let kmeans = DistributedKMeans::<f64>::new(3, config).unwrap();

        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();

        let centroids = kmeans.random_initialization(data.view()).unwrap();
        assert_eq!(centroids.shape(), &[3, 2]);
    }

    #[test]
    fn test_kmeans_plus_plus_initialization() {
        let config = DistributedKMeansConfig::default();
        let kmeans = DistributedKMeans::<f64>::new(2, config).unwrap();

        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0, 5.0, 5.0, 6.0, 6.0,
            ],
        )
        .unwrap();

        let centroids = kmeans.kmeans_plus_plus_initialization(data.view()).unwrap();
        assert_eq!(centroids.shape(), &[2, 2]);

        // Centroids should be different (with high probability)
        let dist = euclidean_distance(centroids.row(0), centroids.row(1));
        assert!(dist > 0.0);
    }

    #[test]
    fn test_predict() {
        let config = DistributedKMeansConfig::default();
        let mut kmeans = DistributedKMeans::<f64>::new(2, config).unwrap();

        // Set known centroids
        let centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 10.0, 10.0]).unwrap();
        kmeans.centroids = Some(centroids);

        // Test prediction
        let test_data =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 9.0, 9.0, -1.0, -1.0, 11.0, 11.0])
                .unwrap();

        let labels = kmeans.predict(test_data.view()).unwrap();
        assert_eq!(labels.len(), 4);

        // Points should be assigned to nearest centroids
        assert_eq!(labels[0], 0); // (1,1) closer to (0,0)
        assert_eq!(labels[1], 1); // (9,9) closer to (10,10)
        assert_eq!(labels[2], 0); // (-1,-1) closer to (0,0)
        assert_eq!(labels[3], 1); // (11,11) closer to (10,10)
    }

    #[test]
    fn test_convergence_check() {
        let config = DistributedKMeansConfig {
            tolerance: 0.1,
            ..Default::default()
        };
        let kmeans = DistributedKMeans::<f64>::new(2, config).unwrap();

        let old_centroids = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        let new_centroids_converged = Array2::from_shape_vec(
            (2, 2),
            vec![0.05, 0.05, 1.05, 1.05], // Small movement
        )
        .unwrap();

        let new_centroids_not_converged = Array2::from_shape_vec(
            (2, 2),
            vec![0.5, 0.5, 1.5, 1.5], // Large movement
        )
        .unwrap();

        // Set up kmeans with old centroids
        let mut kmeans_converged = kmeans;
        kmeans_converged.centroids = Some(old_centroids.clone());
        kmeans_converged.global_inertia = 100.0;

        // Test convergence
        assert!(kmeans_converged
            .check_convergence(&new_centroids_converged, 99.0)
            .unwrap());

        let mut kmeans_not_converged = DistributedKMeans::<f64>::new(
            2,
            DistributedKMeansConfig {
                tolerance: 0.1,
                ..Default::default()
            },
        )
        .unwrap();
        kmeans_not_converged.centroids = Some(old_centroids);
        kmeans_not_converged.global_inertia = 100.0;

        assert!(!kmeans_not_converged
            .check_convergence(&new_centroids_not_converged, 50.0)
            .unwrap());
    }

    #[test]
    fn test_load_balance_score() {
        let config = DistributedKMeansConfig::default();
        let mut kmeans = DistributedKMeans::<f64>::new(2, config).unwrap();

        // Balanced partitions
        let partition1 = DataPartition::new(0, Array2::zeros((100, 2)), 0);
        let partition2 = DataPartition::new(1, Array2::zeros((100, 2)), 1);
        kmeans.partitions = vec![partition1, partition2];

        let balanced_score = kmeans.calculate_load_balance_score();
        assert!(balanced_score > 0.9);

        // Imbalanced partitions
        let partition1 = DataPartition::new(0, Array2::zeros((10, 2)), 0);
        let partition2 = DataPartition::new(1, Array2::zeros((190, 2)), 1);
        kmeans.partitions = vec![partition1, partition2];

        let imbalanced_score = kmeans.calculate_load_balance_score();
        assert!(imbalanced_score < balanced_score);
    }
}
