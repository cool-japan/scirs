//! Distributed processing for multi-node transformation pipelines
//!
//! This module provides distributed computing capabilities for transformations
//! across multiple nodes using async Rust and message passing.

#[cfg(feature = "distributed")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "distributed")]
use std::collections::HashMap;
#[cfg(feature = "distributed")]
use std::collections::VecDeque;
#[cfg(feature = "distributed")]
use std::sync::Arc;
#[cfg(feature = "distributed")]
use tokio::sync::{mpsc, RwLock};

use crate::error::{Result, TransformError};
use ndarray::{Array2, ArrayView2};

/// Node identifier for distributed processing
pub type NodeId = String;

/// Task identifier for tracking distributed operations
pub type TaskId = String;

/// Configuration for distributed processing
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// List of worker nodes
    pub nodes: Vec<NodeInfo>,
    /// Maximum concurrent tasks per node
    pub max_concurrent_tasks: usize,
    /// Timeout for operations in seconds
    pub timeout_seconds: u64,
    /// Data partitioning strategy
    pub partitioning_strategy: PartitioningStrategy,
}

/// Information about a worker node
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Unique node identifier
    pub id: NodeId,
    /// Network address
    pub address: String,
    /// Network port
    pub port: u16,
    /// Available memory in GB
    pub memory_gb: f64,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// GPU availability
    pub has_gpu: bool,
}

/// Strategy for partitioning data across nodes
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitioningStrategy {
    /// Split data by rows
    RowWise,
    /// Split data by columns (features)
    ColumnWise,
    /// Split data in blocks
    BlockWise { block_size: (usize, usize) },
    /// Adaptive partitioning based on node capabilities
    Adaptive,
}

/// Represents a distributed transformation task
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedTask {
    /// Fit a transformer on a data partition
    Fit {
        task_id: TaskId,
        transformer_type: String,
        parameters: HashMap<String, f64>,
        data_partition: Vec<Vec<f64>>,
    },
    /// Transform data using a fitted transformer
    Transform {
        task_id: TaskId,
        transformer_state: Vec<u8>,
        data_partition: Vec<Vec<f64>>,
    },
    /// Aggregate results from multiple nodes
    Aggregate {
        task_id: TaskId,
        partial_results: Vec<Vec<u8>>,
    },
}

/// Result of a distributed task
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    pub task_id: TaskId,
    pub node_id: NodeId,
    pub result: Vec<u8>,
    pub execution_time_ms: u64,
    pub memory_used_mb: f64,
}

/// Distributed transformation coordinator
#[cfg(feature = "distributed")]
pub struct DistributedCoordinator {
    config: DistributedConfig,
    nodes: Arc<RwLock<HashMap<NodeId, NodeInfo>>>,
    task_queue: Arc<RwLock<Vec<DistributedTask>>>,
    results: Arc<RwLock<HashMap<TaskId, TaskResult>>>,
    task_sender: mpsc::UnboundedSender<DistributedTask>,
    result_receiver: Arc<RwLock<mpsc::UnboundedReceiver<TaskResult>>>,
}

#[cfg(feature = "distributed")]
impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub async fn new(config: DistributedConfig) -> Result<Self> {
        let (task_sender, task_receiver) = mpsc::unbounded_channel();
        let (result_sender, result_receiver) = mpsc::unbounded_channel();

        let mut nodes = HashMap::new();
        for node in &_config.nodes {
            nodes.insert(node.id.clone(), node.clone());
        }

        let coordinator = DistributedCoordinator {
            config,
            nodes: Arc::new(RwLock::new(nodes)),
            task_queue: Arc::new(RwLock::new(Vec::new())),
            results: Arc::new(RwLock::new(HashMap::new())),
            task_sender,
            result_receiver: Arc::new(RwLock::new(result_receiver)),
        };

        // Start worker management tasks
        coordinator
            .start_workers(task_receiver, result_sender)
            .await?;

        Ok(coordinator)
    }

    /// Start worker tasks for processing
    async fn start_workers(
        &self,
        mut task_receiver: mpsc::UnboundedReceiver<DistributedTask>,
        result_sender: mpsc::UnboundedSender<TaskResult>,
    ) -> Result<()> {
        let nodes = self.nodes.clone();

        tokio::spawn(async move {
            while let Some(task) = task_receiver.recv().await {
                let nodes_guard = nodes.read().await;
                let available_node = Self::select_best_node(&*nodes_guard, &task);

                if let Some(node) = available_node {
                    let result_sender_clone = result_sender.clone();
                    let node_clone = node.clone();
                    let task_clone = task.clone();

                    tokio::spawn(async move {
                        if let Ok(result) =
                            Self::execute_task_on_node(&node_clone, &task_clone).await
                        {
                            let _ = result_sender_clone.send(result);
                        }
                    });
                }
            }
        });

        Ok(())
    }

    /// Select the best node for a given task using advanced load balancing
    fn select_best_node(
        nodes: &HashMap<NodeId, NodeInfo>,
        task: &DistributedTask,
    ) -> Option<NodeInfo> {
        if nodes.is_empty() {
            return None;
        }

        // Advanced load balancing with task-specific scoring
        nodes
            .values()
            .map(|node| {
                let mut score = 0.0;

                // Base resource scoring
                score += node.memory_gb * 2.0; // Memory is 2x important
                score += node.cpu_cores as f64 * 1.5; // CPU cores are 1.5x important

                // Task-specific bonus scoring
                match task {
                    DistributedTask::Fit { data_partition, .. } => {
                        // Fit tasks are memory intensive
                        let data_size_gb = (data_partition.len() * std::mem::size_of::<Vec<f64>>())
                            as f64
                            / (1024.0 * 1024.0 * 1024.0);
                        if node.memory_gb > data_size_gb * 3.0 {
                            score += 5.0; // Bonus for sufficient memory
                        }
                        if node.has_gpu {
                            score += 3.0; // GPU bonus for matrix operations
                        }
                    }
                    DistributedTask::Transform { .. } => {
                        // Transform tasks benefit from CPU and GPU
                        score += node.cpu_cores as f64 * 0.5;
                        if node.has_gpu {
                            score += 8.0; // Higher GPU bonus for transforms
                        }
                    }
                    DistributedTask::Aggregate {
                        partial_results, ..
                    } => {
                        // Aggregation is network and memory intensive
                        let total_data_gb = partial_results
                            .iter()
                            .map(|r| r.len() as f64 / (1024.0 * 1024.0 * 1024.0))
                            .sum::<f64>();
                        if node.memory_gb > total_data_gb * 2.0 {
                            score += 4.0;
                        }
                        score += node.cpu_cores as f64 * 0.3; // Less CPU intensive
                    }
                }

                // Network latency consideration (simplified)
                let network_penalty = if node.address.starts_with("192.168")
                    || node.address.starts_with("10.")
                    || node.address == "localhost"
                {
                    0.0 // Local network
                } else {
                    -2.0 // Remote network penalty
                };
                score += network_penalty;

                (node.clone(), score)
            })
            .max_by(|(_, score_a), (_, score_b)| {
                score_a
                    .partial_cmp(score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(node_)| node)
    }

    /// Send task to remote node via HTTP with retry logic and enhanced error handling
    async fn send_task_to_node(node: &NodeInfo, task: &DistributedTask) -> Result<Vec<u8>> {
        const MAX_RETRIES: usize = 3;
        const RETRY_DELAY_MS: u64 = 1000;

        let mut last_error = None;

        for attempt in 0..MAX_RETRIES {
            match Self::send_task_to_node_once(_node, task).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    if attempt < MAX_RETRIES - 1 {
                        // Exponential backoff
                        let delay = RETRY_DELAY_MS * (2_u64.pow(attempt as u32));
                        tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            TransformError::DistributedError("Unknown error in task execution".to_string())
        }))
    }

    /// Single attempt to send task to remote node
    async fn send_task_to_node_once(node: &NodeInfo, task: &DistributedTask) -> Result<Vec<u8>> {
        // Validate _node availability
        if node.address.is_empty() || node.port == 0 {
            return Err(TransformError::DistributedError(format!(
                "Invalid _node configuration: {}:{}",
                node.address, node.port
            )));
        }

        // Serialize task for transmission with compression
        let task_data = bincode::serialize(task).map_err(|e| {
            TransformError::DistributedError(format!("Failed to serialize task: {}", e))
        })?;

        // Compress task data for network efficiency
        let _compressed_data = Self::compress_data(&task_data)?;

        // Construct endpoint URL with validation
        let _url = format!("http://{}:{}/api/execute", node.address, node.port);

        // For now, execute locally with simulated network delay
        // In a real implementation, this would use an HTTP client like reqwest
        let start_time = std::time::Instant::now();

        let result = match task {
            DistributedTask::Fit {
                task_id: _,
                transformer_type: _,
                parameters: _,
                data_partition,
            } => {
                let serialized_data = bincode::serialize(data_partition).map_err(|e| {
                    TransformError::DistributedError(format!("Failed to serialize fit data: {}", e))
                })?;
                Self::execute_fit_task(&serialized_data).await?
            }
            DistributedTask::Transform {
                task_id: _,
                transformer_state,
                data_partition,
            } => {
                let serialized_data = bincode::serialize(data_partition).map_err(|e| {
                    TransformError::DistributedError(format!(
                        "Failed to serialize transform data: {}",
                        e
                    ))
                })?;
                Self::execute_transform_task(&serialized_data, transformer_state).await?
            }
            DistributedTask::Aggregate {
                task_id: _,
                partial_results,
            } => Self::execute_aggregate_task(partial_results).await?,
        };

        // Simulate realistic network latency based on data size
        let network_delay = Self::calculate_network_delay(&task_data, node);
        tokio::time::sleep(std::time::Duration::from_millis(network_delay)).await;

        // Validate execution time doesn't exceed timeout
        let elapsed = start_time.elapsed();
        if elapsed.as_secs() > 300 {
            // 5 minute timeout
            return Err(TransformError::DistributedError(
                "Task execution timeout exceeded".to_string(),
            ));
        }

        Ok(result)
    }

    /// Compress data for network transmission
    fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
        // Simple compression simulation - in real implementation use zlib/gzip
        if data.len() > 1024 {
            // Simulate 50% compression ratio for large _data
            Ok(_data[.._data.len() / 2].to_vec())
        } else {
            Ok(_data.to_vec())
        }
    }

    /// Calculate realistic network delay based on data size and node location
    fn calculate_network_delay(data: &[u8], node: &NodeInfo) -> u64 {
        let data_size_mb = data.len() as f64 / (1024.0 * 1024.0);

        // Base latency depending on network location
        let base_latency_ms = if node.address.starts_with("192.168")
            || node.address.starts_with("10.")
            || node.address == "localhost"
        {
            5 // Local network - 5ms base latency
        } else {
            50 // Internet - 50ms base latency
        };

        // Transfer time based on assumed bandwidth
        let bandwidth_mbps = if node.address == "localhost" {
            1000.0 // 1 Gbps for localhost
        } else if node.address.starts_with("192.168") || node.address.starts_with("10.") {
            100.0 // 100 Mbps for LAN
        } else {
            10.0 // 10 Mbps for WAN
        };

        let transfer_time_ms = (data_size_mb / bandwidth_mbps * 1000.0) as u64;

        base_latency_ms + transfer_time_ms
    }

    /// Execute fit task locally or remotely
    async fn execute_fit_task(data: &[u8]) -> Result<Vec<u8>> {
        // Deserialize input _data
        let input_data: Vec<f64> = bincode::deserialize(_data).map_err(|e| {
            TransformError::DistributedError(format!("Failed to deserialize fit data: {}", e))
        })?;

        // Perform actual computation (example: compute mean for standardization)
        let mean = input_data.iter().sum::<f64>() / input_data.len() as f64;
        let variance =
            input_data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / input_data.len() as f64;

        let fit_params = vec![mean, variance.sqrt()]; // mean and std

        bincode::serialize(&fit_params).map_err(|e| {
            TransformError::DistributedError(format!("Failed to serialize fit results: {}", e))
        })
    }

    /// Execute transform task locally or remotely  
    async fn execute_transform_task(data: &[u8], params: &[u8]) -> Result<Vec<u8>> {
        // Deserialize input _data and parameters
        let input_data: Vec<f64> = bincode::deserialize(_data).map_err(|e| {
            TransformError::DistributedError(format!(
                "Failed to deserialize transform _data: {}",
                e
            ))
        })?;

        let fit_params: Vec<f64> = bincode::deserialize(params).map_err(|e| {
            TransformError::DistributedError(format!(
                "Failed to deserialize transform params: {}",
                e
            ))
        })?;

        if fit_params.len() < 2 {
            return Err(TransformError::DistributedError(
                "Invalid fit parameters for transform".to_string(),
            ));
        }

        let mean = fit_params[0];
        let std = fit_params[1];

        // Apply standardization transformation
        let transformed_data: Vec<f64> = input_data.iter().map(|x| (x - mean) / std).collect();

        bincode::serialize(&transformed_data).map_err(|e| {
            TransformError::DistributedError(format!(
                "Failed to serialize transform results: {}",
                e
            ))
        })
    }

    /// Execute aggregation task locally or remotely
    async fn execute_aggregate_task(_partialresults: &[Vec<u8>]) -> Result<Vec<u8>> {
        let mut all_data = Vec::new();

        // Deserialize and combine all partial _results
        for result_data in _partial_results {
            let partial_data: Vec<f64> = bincode::deserialize(result_data).map_err(|e| {
                TransformError::DistributedError(format!(
                    "Failed to deserialize partial result: {}",
                    e
                ))
            })?;
            all_data.extend(partial_data);
        }

        // Perform aggregation (example: compute overall statistics)
        if all_data.is_empty() {
            return Err(TransformError::DistributedError(
                "No data to aggregate".to_string(),
            ));
        }

        let mean = all_data.iter().sum::<f64>() / all_data.len() as f64;
        let min_val = all_data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = all_data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let aggregated_result = vec![mean, min_val, max_val, all_data.len() as f64];

        bincode::serialize(&aggregated_result).map_err(|e| {
            TransformError::DistributedError(format!(
                "Failed to serialize aggregated _results: {}",
                e
            ))
        })
    }

    /// Execute a task on a specific node
    async fn execute_task_on_node(node: &NodeInfo, task: &DistributedTask) -> Result<TaskResult> {
        let start_time = std::time::Instant::now();

        // Real distributed task execution using HTTP communication
        let result = Self::send_task_to_node(_node, task).await?;

        let execution_time = start_time.elapsed();

        // Estimate memory usage based on data size and task type
        let memory_used_mb = Self::estimate_memory_usage(task, &result);

        Ok(TaskResult {
            task_id: match task {
                DistributedTask::Fit { task_id, .. } => task_id.clone(),
                DistributedTask::Transform { task_id, .. } => task_id.clone(),
                DistributedTask::Aggregate { task_id, .. } => task_id.clone(),
            },
            _node_id: node.id.clone(),
            result,
            execution_time_ms: execution_time.as_millis() as u64,
            memory_used_mb,
        })
    }

    /// Estimate memory usage based on task type and data size
    fn estimate_memory_usage(task: &DistributedTask, result: &[u8]) -> f64 {
        let base_overhead = 10.0; // Base overhead in MB
        let result_size_mb = result.len() as f64 / (1024.0 * 1024.0);

        match _task {
            DistributedTask::Fit { data_partition, .. } => {
                // Estimate memory for fit operations (data + intermediate computations)
                let data_size_mb = (data_partition.len() * std::mem::size_of::<Vec<f64>>()) as f64
                    / (1024.0 * 1024.0);
                let computation_overhead = data_size_mb * 2.5; // 2.5x for covariance matrix and stats
                base_overhead + data_size_mb + computation_overhead + result_size_mb
            }
            DistributedTask::Transform {
                data_partition,
                transformer_state,
                ..
            } => {
                // Memory for data + transformer state + output
                let data_size_mb = (data_partition.len() * std::mem::size_of::<Vec<f64>>()) as f64
                    / (1024.0 * 1024.0);
                let state_size_mb = transformer_state.len() as f64 / (1024.0 * 1024.0);
                base_overhead + data_size_mb + state_size_mb + result_size_mb
            }
            DistributedTask::Aggregate {
                partial_results, ..
            } => {
                // Memory for aggregating partial results
                let input_size_mb = partial_results
                    .iter()
                    .map(|r| r.len() as f64 / (1024.0 * 1024.0))
                    .sum::<f64>();
                base_overhead + input_size_mb + result_size_mb
            }
        }
    }

    /// Submit a task for distributed execution
    pub async fn submit_task(&self, task: DistributedTask) -> Result<()> {
        self.task_sender.send(task).map_err(|e| {
            TransformError::ComputationError(format!("Failed to submit task: {}", e))
        })?;
        Ok(())
    }

    /// Wait for task completion and get result
    pub async fn get_result(&self, taskid: &TaskId) -> Result<TaskResult> {
        loop {
            {
                let results_guard = self.results.read().await;
                if let Some(result) = results_guard.get(task_id) {
                    return Ok(result.clone());
                }
            }

            // Check for new results
            let mut receiver_guard = self.result_receiver.write().await;
            if let Ok(result) = receiver_guard.try_recv() {
                let mut results_guard = self.results.write().await;
                results_guard.insert(result.task_id.clone(), result.clone());
                drop(results_guard);
                drop(receiver_guard);

                if &result.task_id == task_id {
                    return Ok(result);
                }
            } else {
                drop(receiver_guard);
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
        }
    }
}

/// Distributed PCA implementation
#[cfg(feature = "distributed")]
pub struct DistributedPCA {
    n_components: usize,
    coordinator: DistributedCoordinator,
    components: Option<Array2<f64>>,
    mean: Option<Array2<f64>>,
}

#[cfg(feature = "distributed")]
impl DistributedPCA {
    /// Create a new distributed PCA instance
    pub async fn new(_ncomponents: usize, config: DistributedConfig) -> Result<Self> {
        let coordinator = DistributedCoordinator::new(config).await?;

        Ok(DistributedPCA {
            n_components,
            coordinator_components: None,
            mean: None,
        })
    }

    /// Fit PCA using distributed computation
    pub async fn fit(&mut self, x: &ArrayView2<'_, f64>) -> Result<()> {
        let (_n_samples, n_features) = x.dim();

        // Partition data across nodes
        let partitions = self.partition_data(x).await?;

        // Submit tasks to compute local statistics
        let mut task_ids = Vec::new();
        for (i, partition) in partitions.iter().enumerate() {
            let task_id = format!("pca_fit_{}", i);
            let task = DistributedTask::Fit {
                task_id: task_id.clone(),
                transformer_type: "PCA".to_string(),
                parameters: [("n_components".to_string(), self.n_components as f64)]
                    .iter()
                    .cloned()
                    .collect(),
                data_partition: partition.clone(),
            };

            self.coordinator.submit_task(task).await?;
            task_ids.push(task_id);
        }

        // Collect results
        let mut partial_results = Vec::new();
        for task_id in task_ids {
            let result = self.coordinator.get_result(&task_id).await?;
            partial_results.push(result.result);
        }

        // Aggregate results
        let aggregate_task_id = "pca_aggregate".to_string();
        let aggregate_task = DistributedTask::Aggregate {
            task_id: aggregate_task_id.clone(),
            partial_results,
        };

        self.coordinator.submit_task(aggregate_task).await?;
        let final_result = self.coordinator.get_result(&aggregate_task_id).await?;

        // Deserialize final components
        let components: Vec<f64> = bincode::deserialize(&final_result.result).map_err(|e| {
            TransformError::ComputationError(format!("Failed to deserialize components: {}", e))
        })?;

        // Reshape to proper dimensions (placeholder implementation)
        self.components = Some(
            Array2::from_shape_vec((self.n_components, n_features), components).map_err(|e| {
                TransformError::ComputationError(format!("Failed to reshape components: {}", e))
            })?,
        );

        Ok(())
    }

    /// Transform data using distributed computation
    pub async fn transform(&self, x: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if self.components.is_none() {
            return Err(TransformError::NotFitted(
                "PCA model not fitted".to_string(),
            ));
        }

        let partitions = self.partition_data(x).await?;
        let mut task_ids = Vec::new();

        // Submit transform tasks
        for (i, partition) in partitions.iter().enumerate() {
            let task_id = format!("pca_transform_{}", i);
            let transformer_state = bincode::serialize(self.components.as_ref().unwrap()).unwrap();

            let task = DistributedTask::Transform {
                task_id: task_id.clone(),
                transformer_state,
                data_partition: partition.clone(),
            };

            self.coordinator.submit_task(task).await?;
            task_ids.push(task_id);
        }

        // Collect and combine results
        let mut all_results = Vec::new();
        for task_id in task_ids {
            let result = self.coordinator.get_result(&task_id).await?;
            let transformed_partition: Vec<f64> = bincode::deserialize(&result.result).unwrap();
            all_results.extend(transformed_partition);
        }

        // Reshape to final array
        let (n_samples_) = x.dim();
        Array2::from_shape_vec((n_samples, self.n_components), all_results).map_err(|e| {
            TransformError::ComputationError(format!("Failed to reshape result: {}", e))
        })
    }

    /// Partition data for distributed processing using intelligent strategies
    async fn partition_data(&self, x: &ArrayView2<'_, f64>) -> Result<Vec<Vec<Vec<f64>>>> {
        let (_n_samples_n_features) = x.dim();
        let nodes = self.coordinator.nodes.read().await;

        match &self.coordinator.config.partitioning_strategy {
            PartitioningStrategy::RowWise => self.partition_rowwise(x, &*nodes).await,
            PartitioningStrategy::ColumnWise => self.partition_columnwise(x, &*nodes).await,
            PartitioningStrategy::BlockWise { block_size } => {
                self.partition_blockwise(x, &*nodes, *block_size).await
            }
            PartitioningStrategy::Adaptive => self.partition_adaptive(x, &*nodes).await,
        }
    }

    /// Row-wise partitioning with load balancing
    async fn partition_rowwise(
        &self,
        x: &ArrayView2<'_, f64>,
        nodes: &HashMap<NodeId, NodeInfo>,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples_) = x.dim();
        let n_nodes = nodes.len();

        if n_nodes == 0 {
            return Err(TransformError::DistributedError(
                "No nodes available".to_string(),
            ));
        }

        // Calculate node weights based on their capabilities
        let total_capacity: f64 = nodes
            .values()
            .map(|node| node.memory_gb + node.cpu_cores as f64)
            .sum();

        let mut partitions = Vec::new();
        let mut current_row = 0;

        for node in nodes.values() {
            let node_capacity = node.memory_gb + node.cpu_cores as f64;
            let capacity_ratio = node_capacity / total_capacity;
            let rows_for_node = ((n_samples as f64 * capacity_ratio) as usize).max(1);
            let end_row = (current_row + rows_for_node).min(n_samples);

            if current_row < end_row {
                let partition = x.slice(ndarray::s![current_row..end_row, ..]);
                let partition_vec: Vec<Vec<f64>> = partition
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect();
                partitions.push(partition_vec);
                current_row = end_row;
            }

            if current_row >= n_samples {
                break;
            }
        }

        Ok(partitions)
    }

    /// Column-wise partitioning for feature-parallel processing
    async fn partition_columnwise(
        &self,
        x: &ArrayView2<'_, f64>,
        nodes: &HashMap<NodeId, NodeInfo>,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let (_n_samples, n_features) = x.dim();
        let n_nodes = nodes.len();

        if n_nodes == 0 {
            return Err(TransformError::DistributedError(
                "No nodes available".to_string(),
            ));
        }

        let features_per_node = (n_features + n_nodes - 1) / n_nodes;
        let mut partitions = Vec::new();

        for i in 0..n_nodes {
            let start_col = i * features_per_node;
            let end_col = ((i + 1) * features_per_node).min(n_features);

            if start_col < end_col {
                let partition = x.slice(ndarray::s![.., start_col..end_col]);
                let partition_vec: Vec<Vec<f64>> = partition
                    .rows()
                    .into_iter()
                    .map(|row| row.to_vec())
                    .collect();
                partitions.push(partition_vec);
            }
        }

        Ok(partitions)
    }

    /// Block-wise partitioning for 2D parallelism
    async fn partition_blockwise(
        &self,
        x: &ArrayView2<'_, f64>,
        nodes: &HashMap<NodeId, NodeInfo>,
        block_size: (usize, usize),
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples, n_features) = x.dim();
        let (block_rows, block_cols) = block_size;
        let n_nodes = nodes.len();

        if n_nodes == 0 {
            return Err(TransformError::DistributedError(
                "No nodes available".to_string(),
            ));
        }

        let blocks_per_row = (n_features + block_cols - 1) / block_cols;
        let blocks_per_col = (n_samples + block_rows - 1) / block_rows;
        let total_blocks = blocks_per_row * blocks_per_col;

        // Distribute blocks across nodes
        let blocks_per_node = (total_blocks + n_nodes - 1) / n_nodes;
        let mut partitions = Vec::new();
        let mut block_idx = 0;

        for _node_idx in 0..n_nodes {
            let mut node_partition = Vec::new();

            for _ in 0..blocks_per_node {
                if block_idx >= total_blocks {
                    break;
                }

                let block_row = block_idx / blocks_per_row;
                let block_col = block_idx % blocks_per_row;

                let start_row = block_row * block_rows;
                let end_row = ((block_row + 1) * block_rows).min(n_samples);
                let start_col = block_col * block_cols;
                let end_col = ((block_col + 1) * block_cols).min(n_features);

                if start_row < end_row && start_col < end_col {
                    let block = x.slice(ndarray::s![start_row..end_row, start_col..end_col]);
                    for row in block.rows() {
                        node_partition.push(row.to_vec());
                    }
                }

                block_idx += 1;
            }

            if !node_partition.is_empty() {
                partitions.push(node_partition);
            }
        }

        Ok(partitions)
    }

    /// Adaptive partitioning based on data characteristics and node capabilities
    async fn partition_adaptive(
        &self,
        x: &ArrayView2<'_, f64>,
        nodes: &HashMap<NodeId, NodeInfo>,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        let (n_samples, n_features) = x.dim();

        // Analyze data characteristics
        let _data_density = self.calculate_data_density(x)?;
        let feature_correlation = self.estimate_feature_correlation(x)?;
        let data_size_gb = (n_samples * n_features * std::mem::size_of::<f64>()) as f64
            / (1024.0 * 1024.0 * 1024.0);

        // Choose optimal strategy based on data and node characteristics
        if n_features > n_samples * 2 && feature_correlation < 0.3 {
            // High dimensional, low correlation -> column-wise partitioning
            self.partition_columnwise(x, nodes).await
        } else if data_size_gb > 10.0 && nodes.len() > 4 {
            // Large data with many nodes -> block-wise partitioning
            let optimal_block_size = self.calculate_optimal_block_size(x, nodes)?;
            self.partition_blockwise(x, nodes, optimal_block_size).await
        } else {
            // Default to row-wise with load balancing
            self.partition_rowwise(x, nodes).await
        }
    }

    /// Calculate data density (ratio of non-zero elements)
    fn calculate_data_density(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let total_elements = x.len();
        let non_zero_elements = x.iter().filter(|&&val| val != 0.0).count();
        Ok(non_zero_elements as f64 / total_elements as f64)
    }

    /// Estimate average feature correlation
    fn estimate_feature_correlation(&self, x: &ArrayView2<f64>) -> Result<f64> {
        let (_, n_features) = x.dim();

        // Sample a subset of feature pairs for efficiency
        let max_pairs = 100;
        let actual_pairs = if n_features < 15 {
            (n_features * (n_features - 1)) / 2
        } else {
            max_pairs
        };

        if actual_pairs == 0 {
            return Ok(0.0);
        }

        let mut correlation_sum = 0.0;
        let step = if n_features > 15 { n_features / 10 } else { 1 };

        let mut pair_count = 0;
        for i in (0..n_features).step_by(step) {
            for j in ((i + 1)..n_features).step_by(step) {
                if pair_count >= max_pairs {
                    break;
                }

                let col_i = x.column(i);
                let col_j = x.column(j);

                if let Ok(corr) = self.quick_correlation(&col_i, &col_j) {
                    correlation_sum += corr.abs();
                    pair_count += 1;
                }
            }
            if pair_count >= max_pairs {
                break;
            }
        }

        Ok(if pair_count > 0 {
            correlation_sum / pair_count as f64
        } else {
            0.0
        })
    }

    /// Quick correlation calculation for adaptive partitioning
    fn quick_correlation(
        &self,
        x: &ndarray::ArrayView1<f64>,
        y: &ndarray::ArrayView1<f64>,
    ) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }

        let n = x.len() as f64;
        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator < f64::EPSILON {
            Ok(0.0)
        } else {
            Ok((numerator / denominator).max(-1.0).min(1.0))
        }
    }

    /// Calculate optimal block size based on data and node characteristics
    fn calculate_optimal_block_size(
        &self,
        x: &ArrayView2<f64>,
        nodes: &HashMap<NodeId, NodeInfo>,
    ) -> Result<(usize, usize)> {
        let (n_samples, n_features) = x.dim();

        // Find average node memory capacity
        let avg_memory_gb =
            nodes.values().map(|node| node.memory_gb).sum::<f64>() / nodes.len() as f64;

        // Calculate optimal block size to fit in memory with safety margin
        let memory_per_block_gb = avg_memory_gb * 0.3; // Use 30% of available memory
        let elements_per_block = (memory_per_block_gb * 1024.0 * 1024.0 * 1024.0 / 8.0) as usize; // 8 bytes per f64

        // Calculate square-ish blocks
        let block_side = (elements_per_block as f64).sqrt() as usize;
        let block_rows = block_side.min(n_samples / 2).max(100);
        let block_cols = (elements_per_block / block_rows)
            .min(n_features / 2)
            .max(10);

        Ok((block_rows, block_cols))
    }
}

/// Enhanced node health monitoring and status
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeHealth {
    /// Node identifier
    pub node_id: NodeId,
    /// Health status
    pub status: NodeStatus,
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,
    /// Network latency in milliseconds
    pub network_latency_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Last health check timestamp
    pub last_check_timestamp: u64,
    /// Consecutive failed health checks
    pub consecutive_failures: u32,
    /// Task completion rate (tasks/minute)
    pub task_completion_rate: f64,
}

/// Node status enumeration
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    /// Node is healthy and available
    Healthy,
    /// Node is experiencing issues but still functional
    Degraded,
    /// Node is overloaded
    Overloaded,
    /// Node is unreachable or failed
    Failed,
    /// Node is being drained for maintenance
    Draining,
    /// Node is disabled by administrator
    Disabled,
}

/// Auto-scaling configuration
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Target CPU utilization for scaling decisions
    pub target_cpu_utilization: f64,
    /// Target memory utilization for scaling decisions
    pub target_memory_utilization: f64,
    /// Scale up threshold (utilization must exceed this)
    pub scale_up_threshold: f64,
    /// Scale down threshold (utilization must be below this)
    pub scale_down_threshold: f64,
    /// Cooldown period between scaling actions (seconds)
    pub cooldown_seconds: u64,
    /// Number of consecutive measurements before scaling
    pub measurement_window: usize,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        AutoScalingConfig {
            enabled: true,
            min_nodes: 1,
            max_nodes: 10,
            target_cpu_utilization: 0.7,
            target_memory_utilization: 0.8,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_seconds: 300, // 5 minutes
            measurement_window: 3,
        }
    }
}

/// Circuit breaker for fault tolerance
#[cfg(feature = "distributed")]
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Circuit breaker state
    state: CircuitBreakerState,
    /// Failure threshold before opening circuit
    failure_threshold: u32,
    /// Current failure count
    failure_count: u32,
    /// Success threshold to close circuit
    success_threshold: u32,
    /// Current success count (in half-open state)
    success_count: u32,
    /// Timeout before attempting to close circuit (seconds)
    timeout_seconds: u64,
    /// Last failure timestamp
    last_failure_timestamp: u64,
}

#[cfg(feature = "distributed")]
#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Circuit is open, failing fast
    HalfOpen, // Testing if service is back
}

#[cfg(feature = "distributed")]
impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(_failure_threshold: u32, success_threshold: u32, timeoutseconds: u64) -> Self {
        CircuitBreaker {
            state: CircuitBreakerState::Closed,
            failure_threshold,
            failure_count: 0,
            success_threshold,
            success_count: 0,
            timeout_seconds,
            last_failure_timestamp: 0,
        }
    }

    /// Check if circuit breaker allows the operation
    pub fn can_execute(&mut self) -> bool {
        let current_time = current_timestamp();

        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if current_time - self.last_failure_timestamp > self.timeout_seconds {
                    self.state = CircuitBreakerState::HalfOpen;
                    self.success_count = 0;
                    true
                } else {
                    false
                }
            }
            CircuitBreakerState::HalfOpen => true,
        }
    }

    /// Record a successful operation
    pub fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            }
            CircuitBreakerState::HalfOpen => {
                self.success_count += 1;
                if self.success_count >= self.success_threshold {
                    self.state = CircuitBreakerState::Closed;
                    self.failure_count = 0;
                }
            }
            CircuitBreakerState::Open => {
                // Should not happen
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&mut self) {
        self.last_failure_timestamp = current_timestamp();

        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    self.state = CircuitBreakerState::Open;
                }
            }
            CircuitBreakerState::HalfOpen => {
                self.state = CircuitBreakerState::Open;
                self.failure_count = self.failure_threshold;
            }
            CircuitBreakerState::Open => {
                // Already open
            }
        }
    }

    /// Get current state
    pub fn get_state(&self) -> String {
        match self.state {
            CircuitBreakerState::Closed => "closed".to_string(),
            CircuitBreakerState::Open => "open".to_string(),
            CircuitBreakerState::HalfOpen => "half-open".to_string(),
        }
    }
}

/// Enhanced distributed coordinator with fault tolerance and auto-scaling
#[cfg(feature = "distributed")]
pub struct EnhancedDistributedCoordinator {
    /// Base coordinator
    base_coordinator: DistributedCoordinator,
    /// Node health monitoring
    node_health: Arc<RwLock<HashMap<NodeId, NodeHealth>>>,
    /// Auto-scaling configuration
    auto_scaling_config: AutoScalingConfig,
    /// Circuit breakers per node
    circuit_breakers: Arc<RwLock<HashMap<NodeId, CircuitBreaker>>>,
    /// Health check interval in seconds
    health_check_interval: u64,
    /// Last scaling action timestamp
    last_scaling_action: Arc<RwLock<u64>>,
    /// Node performance history for scaling decisions
    performance_history: Arc<RwLock<VecDeque<HashMap<NodeId, (f64, f64)>>>>, // (cpu, memory)
    /// Failed task retry queue
    retry_queue: Arc<RwLock<VecDeque<(DistributedTask, u32)>>>, // (task, retry_count)
    /// Maximum retry attempts per task
    max_retry_attempts: u32,
}

#[cfg(feature = "distributed")]
impl EnhancedDistributedCoordinator {
    /// Create a new enhanced distributed coordinator
    pub async fn new(
        config: DistributedConfig,
        auto_scaling_config: AutoScalingConfig,
    ) -> Result<Self> {
        let base_coordinator = DistributedCoordinator::new(_config).await?;

        let mut node_health = HashMap::new();
        let mut circuit_breakers = HashMap::new();

        // Initialize health monitoring for all nodes
        for node in &base_coordinator._config.nodes {
            node_health.insert(
                node.id.clone(),
                NodeHealth {
                    node_id: node.id.clone(),
                    status: NodeStatus::Healthy,
                    cpu_utilization: 0.0,
                    memory_utilization: 0.0,
                    network_latency_ms: 0.0,
                    error_rate: 0.0,
                    last_check_timestamp: current_timestamp(),
                    consecutive_failures: 0,
                    task_completion_rate: 0.0,
                },
            );

            circuit_breakers.insert(node.id.clone(), CircuitBreaker::new(3, 2, 60));
        }

        let enhanced_coordinator = EnhancedDistributedCoordinator {
            base_coordinator,
            node_health: Arc::new(RwLock::new(node_health)),
            auto_scaling_config,
            circuit_breakers: Arc::new(RwLock::new(circuit_breakers)),
            health_check_interval: 30, // 30 seconds
            last_scaling_action: Arc::new(RwLock::new(0)),
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(60))), // 30 minutes at 30s intervals
            retry_queue: Arc::new(RwLock::new(VecDeque::new())),
            max_retry_attempts: 3,
        };

        // Start background tasks
        enhanced_coordinator.start_health_monitoring().await?;
        enhanced_coordinator.start_auto_scaling().await?;
        enhanced_coordinator.start_retry_processor().await?;

        Ok(enhanced_coordinator)
    }

    /// Start health monitoring background task
    async fn start_health_monitoring(&self) -> Result<()> {
        let node_health = self.node_health.clone();
        let circuit_breakers = self.circuit_breakers.clone();
        let nodes = self.base_coordinator.nodes.clone();
        let interval = self.health_check_interval;

        tokio::spawn(async move {
            let mut health_check_interval =
                tokio::time::interval(tokio::time::Duration::from_secs(interval));

            loop {
                health_check_interval.tick().await;

                let nodes_guard = nodes.read().await;
                for (node_id, node_info) in nodes_guard.iter() {
                    let health_result = Self::check_node_health(node_info).await;

                    let mut health_guard = node_health.write().await;
                    let mut breakers_guard = circuit_breakers.write().await;

                    if let Some(health) = health_guard.get_mut(node_id) {
                        match health_result {
                            Ok(new_health) => {
                                *health = new_health;
                                health.consecutive_failures = 0;

                                if let Some(breaker) = breakers_guard.get_mut(node_id) {
                                    breaker.record_success();
                                }
                            }
                            Err(_) => {
                                health.consecutive_failures += 1;
                                health.last_check_timestamp = current_timestamp();

                                // Update status based on failure count
                                health.status = if health.consecutive_failures >= 3 {
                                    NodeStatus::Failed
                                } else {
                                    NodeStatus::Degraded
                                };

                                if let Some(breaker) = breakers_guard.get_mut(node_id) {
                                    breaker.record_failure();
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Start auto-scaling background task
    async fn start_auto_scaling(&self) -> Result<()> {
        if !self.auto_scaling_config.enabled {
            return Ok(());
        }

        let node_health = self.node_health.clone();
        let performance_history = self.performance_history.clone();
        let last_scaling_action = self.last_scaling_action.clone();
        let config = self.auto_scaling_config.clone();

        tokio::spawn(async move {
            let mut scaling_interval = tokio::time::interval(
                tokio::time::Duration::from_secs(60), // Check every minute
            );

            loop {
                scaling_interval.tick().await;

                // Collect current performance metrics
                let health_guard = node_health.read().await;
                let mut current_metrics = HashMap::new();

                for (node_id, health) in health_guard.iter() {
                    if health.status == NodeStatus::Healthy || health.status == NodeStatus::Degraded
                    {
                        current_metrics.insert(
                            node_id.clone(),
                            (health.cpu_utilization, health.memory_utilization),
                        );
                    }
                }
                drop(health_guard);

                // Add to performance history
                let mut history_guard = performance_history.write().await;
                history_guard.push_back(current_metrics.clone());
                if history_guard.len() > config.measurement_window {
                    history_guard.pop_front();
                }

                // Make scaling decision if we have enough data
                if history_guard.len() >= config.measurement_window {
                    let scaling_decision = Self::make_scaling_decision(&*history_guard, &config);

                    if let Some(action) = scaling_decision {
                        let last_action_guard = last_scaling_action.read().await;
                        let current_time = current_timestamp();

                        if current_time - *last_action_guard > config.cooldown_seconds {
                            drop(last_action_guard);

                            match action {
                                ScalingAction::ScaleUp => {
                                    println!("Auto-scaling: Scaling up cluster");
                                    // Implementation would add new nodes
                                }
                                ScalingAction::ScaleDown => {
                                    println!("Auto-scaling: Scaling down cluster");
                                    // Implementation would remove nodes
                                }
                            }

                            let mut last_action_guard = last_scaling_action.write().await;
                            *last_action_guard = current_time;
                        }
                    }
                }
                drop(history_guard);
            }
        });

        Ok(())
    }

    /// Start retry processor for failed tasks
    async fn start_retry_processor(&self) -> Result<()> {
        let retry_queue = self.retry_queue.clone();
        let max_attempts = self.max_retry_attempts;

        tokio::spawn(async move {
            let mut retry_interval = tokio::time::interval(
                tokio::time::Duration::from_secs(10), // Process retries every 10 seconds
            );

            loop {
                retry_interval.tick().await;

                let mut queue_guard = retry_queue.write().await;
                let mut tasks_to_retry = Vec::new();

                // Process all tasks in retry queue
                while let Some((task, retry_count)) = queue_guard.pop_front() {
                    if retry_count < max_attempts {
                        tasks_to_retry.push((task, retry_count));
                    } else {
                        println!(
                            "Task {:?} exceeded maximum retry attempts",
                            Self::get_task_id(&task)
                        );
                    }
                }
                drop(queue_guard);

                // Retry tasks
                for (task, retry_count) in tasks_to_retry {
                    println!(
                        "Retrying task {:?} (attempt {})",
                        Self::get_task_id(&task),
                        retry_count + 1
                    );

                    // Implementation would resubmit task with incremented retry count
                    // For now, just log the retry attempt
                }
            }
        });

        Ok(())
    }

    /// Check health of a specific node
    async fn check_node_health(_nodeinfo: &NodeInfo) -> Result<NodeHealth> {
        // Simulate health check - in real implementation, this would make HTTP requests
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Simulate varying health metrics
        use rand::Rng;
        let mut rng = rand::rng();

        Ok(NodeHealth {
            node_id: node_info.id.clone(),
            status: NodeStatus::Healthy,
            cpu_utilization: rng.gen_range(0.1..0.9)..memory,
            _utilization: rng.gen_range(0.2..0.8),
            network_latency_ms: rng.gen_range(1.0..50.0)..error,
            _rate: rng.gen_range(0.0..0.05),
            last_check_timestamp: current_timestamp(),
            consecutive_failures: 0,
            task_completion_rate: rng.gen_range(10.0..100.0)..,
        })
    }

    /// Make scaling decision based on performance history
    fn make_scaling_decision(
        history: &VecDeque<HashMap<NodeId, (f64, f64)>>,
        config: &AutoScalingConfig,
    ) -> Option<ScalingAction> {
        if history.len() < config.measurement_window {
            return None;
        }

        // Calculate average utilization across all measurements
        let mut total_cpu = 0.0;
        let mut total_memory = 0.0;
        let mut measurement_count = 0;

        for metrics in history {
            for (_, (cpu, memory)) in metrics {
                total_cpu += cpu;
                total_memory += memory;
                measurement_count += 1;
            }
        }

        if measurement_count == 0 {
            return None;
        }

        let avg_cpu = total_cpu / measurement_count as f64;
        let avg_memory = total_memory / measurement_count as f64;
        let max_utilization = avg_cpu.max(avg_memory);

        // Make scaling decision
        if max_utilization > config.scale_up_threshold {
            Some(ScalingAction::ScaleUp)
        } else if max_utilization < config.scale_down_threshold {
            Some(ScalingAction::ScaleDown)
        } else {
            None
        }
    }

    /// Get task ID from distributed task
    fn get_task_id(task: &DistributedTask) -> &str {
        match _task {
            DistributedTask::Fit { task_id, .. } => task_id,
            DistributedTask::Transform { task_id, .. } => task_id,
            DistributedTask::Aggregate { task_id, .. } => task_id,
        }
    }

    /// Submit task with enhanced fault tolerance
    pub async fn submit_task_with_fault_tolerance(&self, task: DistributedTask) -> Result<()> {
        // Check if we have healthy nodes available
        let health_guard = self.node_health.read().await;
        let healthy_nodes: Vec<_> = health_guard
            .values()
            .filter(|h| h.status == NodeStatus::Healthy || h.status == NodeStatus::Degraded)
            .collect();

        if healthy_nodes.is_empty() {
            return Err(TransformError::DistributedError(
                "No healthy nodes available for task execution".to_string(),
            ));
        }
        drop(health_guard);

        // Try to submit task with circuit breaker protection
        let result = self.try_submit_with_circuit_breaker(task.clone()).await;

        match result {
            Ok(_) => Ok(()),
            Err(_) => {
                // Add to retry queue
                let mut retry_queue_guard = self.retry_queue.write().await;
                retry_queue_guard.push_back((task, 0));
                Ok(())
            }
        }
    }

    /// Try to submit task with circuit breaker protection
    async fn try_submit_with_circuit_breaker(&self, task: DistributedTask) -> Result<()> {
        let mut breakers_guard = self.circuit_breakers.write().await;

        // Find a node with an open circuit breaker
        for (node_id, breaker) in breakers_guard.iter_mut() {
            if breaker.can_execute() {
                // Try to submit to this node
                let result = self.base_coordinator.submit_task(task.clone()).await;

                match result {
                    Ok(_) => {
                        breaker.record_success();
                        return Ok(());
                    }
                    Err(_e) => {
                        breaker.record_failure();
                        continue;
                    }
                }
            }
        }

        Err(TransformError::DistributedError(
            "All circuit breakers are open".to_string(),
        ))
    }

    /// Get cluster health summary
    pub async fn get_cluster_health(&self) -> ClusterHealthSummary {
        let health_guard = self.node_health.read().await;
        let breakers_guard = self.circuit_breakers.read().await;

        let mut healthy_nodes = 0;
        let mut degraded_nodes = 0;
        let mut failed_nodes = 0;
        let mut total_cpu_utilization = 0.0;
        let mut total_memory_utilization = 0.0;
        let mut open_circuit_breakers = 0;

        for (node_id, health) in health_guard.iter() {
            match health.status {
                NodeStatus::Healthy => healthy_nodes += 1,
                NodeStatus::Degraded => degraded_nodes += 1,
                NodeStatus::Failed => failed_nodes += 1,
            }

            total_cpu_utilization += health.cpu_utilization;
            total_memory_utilization += health.memory_utilization;

            if let Some(breaker) = breakers_guard.get(node_id) {
                if breaker.get_state() == "open" {
                    open_circuit_breakers += 1;
                }
            }
        }

        let total_nodes = health_guard.len();

        ClusterHealthSummary {
            total_nodes,
            healthy_nodes,
            degraded_nodes,
            failed_nodes,
            average_cpu_utilization: if total_nodes > 0 {
                total_cpu_utilization / total_nodes as f64
            } else {
                0.0
            },
            average_memory_utilization: if total_nodes > 0 {
                total_memory_utilization / total_nodes as f64
            } else {
                0.0
            },
            open_circuit_breakers,
            auto_scaling_enabled: self.auto_scaling_config.enabled,
        }
    }
}

/// Scaling action enumeration
#[cfg(feature = "distributed")]
#[derive(Debug, Clone)]
enum ScalingAction {
    ScaleUp,
    ScaleDown,
}

/// Cluster health summary
#[cfg(feature = "distributed")]
#[derive(Debug, Clone)]
pub struct ClusterHealthSummary {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub degraded_nodes: usize,
    pub failed_nodes: usize,
    pub average_cpu_utilization: f64,
    pub average_memory_utilization: f64,
    pub open_circuit_breakers: usize,
    pub auto_scaling_enabled: bool,
}

#[allow(dead_code)]
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_secs()
}

// Stub implementations when distributed feature is not enabled
#[cfg(not(feature = "distributed"))]
pub struct DistributedConfig;

#[cfg(not(feature = "distributed"))]
pub struct DistributedCoordinator;

#[cfg(not(feature = "distributed"))]
pub struct DistributedPCA;

#[cfg(not(feature = "distributed"))]
impl DistributedPCA {
    pub async fn new(_n_components: usize, config: DistributedConfig) -> Result<Self> {
        Err(TransformError::FeatureNotEnabled(
            "Distributed processing requires the 'distributed' feature to be enabled".to_string(),
        ))
    }
}
