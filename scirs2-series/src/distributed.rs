//! Distributed computing support for time series processing
//!
//! This module provides infrastructure for distributing time series computations
//! across multiple nodes, supporting both synchronous and asynchronous processing.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Float;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, Instant};

use crate::error::{Result, TimeSeriesError};
use statrs::statistics::Statistics;

/// Configuration for distributed computing cluster
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// List of worker node addresses
    pub nodes: Vec<String>,
    /// Maximum number of concurrent tasks per node
    pub max_concurrent_tasks: usize,
    /// Timeout for task execution
    pub task_timeout: Duration,
    /// Chunk size for data splitting
    pub chunk_size: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            nodes: vec!["localhost:8080".to_string()],
            max_concurrent_tasks: 4,
            task_timeout: Duration::from_secs(30),
            chunk_size: 10000,
            load_balancing: LoadBalancingStrategy::RoundRobin,
            fault_tolerance: FaultToleranceConfig::default(),
        }
    }
}

/// Load balancing strategies for task distribution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution across nodes
    RoundRobin,
    /// Distribute based on current node load
    LoadBased,
    /// Random distribution
    Random,
    /// Weighted distribution based on node capabilities
    Weighted,
}

/// Fault tolerance configuration
#[derive(Debug, Clone)]
pub struct FaultToleranceConfig {
    /// Maximum number of retry attempts
    pub max_retries: usize,
    /// Retry delay
    pub retry_delay: Duration,
    /// Whether to enable task replication
    pub enable_replication: bool,
    /// Replication factor
    pub replication_factor: usize,
    /// Node failure detection timeout
    pub failure_detection_timeout: Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            enable_replication: false,
            replication_factor: 2,
            failure_detection_timeout: Duration::from_secs(10),
        }
    }
}

/// Task definition for distributed execution
#[derive(Debug, Clone)]
pub struct DistributedTask<F: Float> {
    /// Unique task identifier
    pub id: String,
    /// Task type
    pub task_type: TaskType,
    /// Input data chunk
    pub input_data: Array1<F>,
    /// Additional parameters
    pub parameters: HashMap<String, f64>,
    /// Priority level
    pub priority: TaskPriority,
    /// Dependencies on other tasks
    pub dependencies: Vec<String>,
}

/// Types of tasks that can be executed in distributed manner
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    /// Time series forecasting
    Forecasting,
    /// Decomposition analysis
    Decomposition,
    /// Feature extraction
    FeatureExtraction,
    /// Anomaly detection
    AnomalyDetection,
    /// Cross-correlation computation
    CrossCorrelation,
    /// Fourier transform
    FourierTransform,
    /// Wavelet transform
    WaveletTransform,
    /// Custom user-defined task
    Custom(String),
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Low priority
    Low = 1,
    /// Normal priority
    Normal = 2,
    /// High priority
    High = 3,
    /// Critical priority
    Critical = 4,
}

/// Result of a distributed task execution
#[derive(Debug, Clone)]
pub struct TaskResult<F: Float> {
    /// Task identifier
    pub taskid: String,
    /// Execution status
    pub status: TaskStatus,
    /// Result data
    pub data: Option<Array1<F>>,
    /// Execution metrics
    pub metrics: TaskMetrics,
    /// Error information if failed
    pub error: Option<String>,
}

/// Task execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    Timeout,
}

/// Metrics for task execution
#[derive(Debug, Clone)]
pub struct TaskMetrics {
    /// Execution time
    pub execution_time: Duration,
    /// Node that executed the task
    pub executed_on: String,
    /// Memory usage during execution
    pub memory_usage: usize,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Network transfer time
    pub network_time: Duration,
}

impl Default for TaskMetrics {
    fn default() -> Self {
        Self {
            execution_time: Duration::ZERO,
            executed_on: String::new(),
            memory_usage: 0,
            cpu_utilization: 0.0,
            network_time: Duration::ZERO,
        }
    }
}

/// Node information in the cluster
#[derive(Debug, Clone)]
pub struct NodeInfo {
    /// Node address
    pub address: String,
    /// Current status
    pub status: NodeStatus,
    /// Available CPU cores
    pub cpu_cores: usize,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Current load (0.0 to 1.0)
    pub current_load: f64,
    /// Number of running tasks
    pub running_tasks: usize,
    /// Node capabilities
    pub capabilities: Vec<String>,
    /// Last heartbeat timestamp
    pub last_heartbeat: Instant,
}

/// Node status in the cluster
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is available for tasks
    Available,
    /// Node is busy with tasks
    Busy,
    /// Node is offline
    Offline,
    /// Node is under maintenance
    Maintenance,
    /// Node has failed
    Failed,
}

/// Distributed time series processor
pub struct DistributedProcessor<
    F: Float
        + Debug
        + Clone
        + scirs2_core::numeric::FromPrimitive
        + scirs2_core::numeric::Zero
        + scirs2_core::ndarray::ScalarOperand,
> {
    /// Cluster configuration
    config: ClusterConfig,
    /// Node registry
    nodes: HashMap<String, NodeInfo>,
    /// Task queue
    task_queue: Vec<DistributedTask<F>>,
    /// Running tasks
    running_tasks: HashMap<String, DistributedTask<F>>,
    /// Completed tasks
    completed_tasks: HashMap<String, TaskResult<F>>,
    /// Load balancer state
    load_balancer_state: LoadBalancerState,
}

/// Internal state for load balancer
#[derive(Debug, Default)]
struct LoadBalancerState {
    /// Round-robin counter
    round_robin_counter: usize,
    /// Node weights for weighted distribution
    #[allow(dead_code)]
    node_weights: HashMap<String, f64>,
    /// Node load history
    #[allow(dead_code)]
    load_history: HashMap<String, Vec<f64>>,
}

impl<
        F: Float
            + Debug
            + Clone
            + scirs2_core::numeric::FromPrimitive
            + scirs2_core::numeric::Zero
            + scirs2_core::ndarray::ScalarOperand,
    > DistributedProcessor<F>
{
    /// Create a new distributed processor
    pub fn new(config: ClusterConfig) -> Self {
        let mut nodes = HashMap::new();

        // Initialize node information
        for address in &config.nodes {
            nodes.insert(
                address.clone(),
                NodeInfo {
                    address: address.clone(),
                    status: NodeStatus::Available,
                    cpu_cores: 4, // Default values - in practice would be detected
                    total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                    available_memory: 6 * 1024 * 1024 * 1024, // 6GB
                    current_load: 0.0,
                    running_tasks: 0,
                    capabilities: vec!["time_series".to_string(), "forecasting".to_string()],
                    last_heartbeat: Instant::now(),
                },
            );
        }

        Self {
            config,
            nodes,
            task_queue: Vec::new(),
            running_tasks: HashMap::new(),
            completed_tasks: HashMap::new(),
            load_balancer_state: LoadBalancerState::default(),
        }
    }

    /// Submit a task for distributed execution
    pub fn submit_task(&mut self, task: DistributedTask<F>) -> Result<()> {
        // Validate task dependencies
        for dep_id in &task.dependencies {
            if !self.completed_tasks.contains_key(dep_id)
                && !self.running_tasks.contains_key(dep_id)
            {
                return Err(TimeSeriesError::InvalidInput(format!(
                    "Dependency task {dep_id} not found"
                )));
            }
        }

        // Add to queue with priority ordering (higher priority first)
        let insert_pos = self
            .task_queue
            .binary_search_by(|t| t.priority.cmp(&task.priority).reverse())
            .unwrap_or_else(|pos| pos);

        self.task_queue.insert(insert_pos, task);
        Ok(())
    }

    /// Process distributed time series forecasting
    pub fn distributed_forecast(
        &mut self,
        data: &Array1<F>,
        horizon: usize,
        method: &str,
    ) -> Result<Array1<F>> {
        // Split data into chunks for parallel processing
        let chunk_size = self
            .config
            .chunk_size
            .min(data.len() / self.config.nodes.len().max(1));
        let chunks: Vec<Array1<F>> = data
            .axis_chunks_iter(Axis(0), chunk_size)
            .map(|chunk| chunk.to_owned())
            .collect();

        // Create forecasting tasks for each chunk
        let mut tasks = Vec::new();
        for (i, chunk) in chunks.iter().enumerate() {
            let mut params = HashMap::new();
            params.insert("horizon".to_string(), horizon as f64);
            params.insert("chunk_index".to_string(), i as f64);

            let task = DistributedTask {
                id: format!("forecast_chunk_{i}"),
                task_type: TaskType::Forecasting,
                input_data: chunk.clone(),
                parameters: params,
                priority: TaskPriority::Normal,
                dependencies: vec![],
            };

            tasks.push(task);
        }

        // Submit tasks
        for task in tasks {
            self.submit_task(task)?;
        }

        // Process tasks (simplified simulation)
        self.process_pending_tasks()?;

        // Aggregate results
        self.aggregate_forecast_results(horizon)
    }

    /// Process distributed feature extraction
    pub fn distributed_feature_extraction(
        &mut self,
        data: &Array1<F>,
        features: &[String],
    ) -> Result<Array2<F>> {
        // Split data into overlapping windows for feature extraction
        let window_size = 1000.min(data.len() / 2);
        let overlap = window_size / 4;
        let step = window_size - overlap;

        let mut tasks = Vec::new();
        let mut i = 0;
        let mut start = 0;

        while start + window_size <= data.len() {
            let end = (start + window_size).min(data.len());
            let window = data.slice(scirs2_core::ndarray::s![start..end]).to_owned();

            let mut params = HashMap::new();
            params.insert("window_index".to_string(), i as f64);
            params.insert("window_size".to_string(), window_size as f64);

            let task = DistributedTask {
                id: format!("features_window_{i}"),
                task_type: TaskType::FeatureExtraction,
                input_data: window,
                parameters: params,
                priority: TaskPriority::Normal,
                dependencies: vec![],
            };

            tasks.push(task);
            start += step;
            i += 1;
        }

        // Submit tasks
        for task in tasks {
            self.submit_task(task)?;
        }

        // Process tasks
        self.process_pending_tasks()?;

        // Aggregate feature results
        self.aggregate_feature_results(features.len())
    }

    /// Select optimal node for task execution
    fn select_node_for_task(&mut self, task: &DistributedTask<F>) -> Result<String> {
        let available_nodes: Vec<&String> = self
            .nodes
            .iter()
            .filter(|(_, info)| {
                info.status == NodeStatus::Available
                    && info.running_tasks < self.config.max_concurrent_tasks
            })
            .map(|(address, _)| address)
            .collect();

        if available_nodes.is_empty() {
            return Err(TimeSeriesError::ComputationError(
                "No available nodes for task execution".to_string(),
            ));
        }

        let selected_node = match self.config.load_balancing {
            LoadBalancingStrategy::RoundRobin => {
                let index = self.load_balancer_state.round_robin_counter % available_nodes.len();
                self.load_balancer_state.round_robin_counter += 1;
                available_nodes[index].clone()
            }
            LoadBalancingStrategy::LoadBased => {
                // Select node with lowest current load
                available_nodes
                    .iter()
                    .min_by(|a, b| {
                        let load_a = self.nodes.get(*a as &str).unwrap().current_load;
                        let load_b = self.nodes.get(*b as &str).unwrap().current_load;
                        load_a
                            .partial_cmp(&load_b)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap()
                    .to_string()
            }
            LoadBalancingStrategy::Random => {
                // Simple hash-based selection for deterministic behavior
                let hash = task.id.len() % available_nodes.len();
                available_nodes[hash].clone()
            }
            LoadBalancingStrategy::Weighted => {
                // Select based on node weights (simplified implementation)
                available_nodes[0].clone() // Fallback to first available
            }
        };

        Ok(selected_node)
    }

    /// Process pending tasks in the queue
    fn process_pending_tasks(&mut self) -> Result<()> {
        while let Some(task) = self.task_queue.pop() {
            // Check if dependencies are satisfied
            let dependencies_satisfied = task.dependencies.iter().all(|dep_id| {
                self.completed_tasks
                    .get(dep_id)
                    .map(|result| result.status == TaskStatus::Completed)
                    .unwrap_or(false)
            });

            if !dependencies_satisfied {
                // Put task back in queue
                self.task_queue.push(task);
                continue;
            }

            // Select node for execution
            let node_address = self.select_node_for_task(&task)?;

            // Simulate task execution
            let result = self.execute_task_on_node(&task, &node_address)?;

            // Store result
            self.completed_tasks.insert(task.id.clone(), result);
            self.running_tasks.remove(&task.id);
        }

        Ok(())
    }

    /// Execute a task on a specific node (simulated)
    fn execute_task_on_node(
        &mut self,
        task: &DistributedTask<F>,
        node_address: &str,
    ) -> Result<TaskResult<F>> {
        let start_time = Instant::now();

        // Mark task as running
        self.running_tasks.insert(task.id.clone(), task.clone());

        // Update node status
        if let Some(node) = self.nodes.get_mut(node_address) {
            node.running_tasks += 1;
            node.current_load = node.running_tasks as f64 / self.config.max_concurrent_tasks as f64;
        }

        // Simulate task execution based on task type
        let result_data = match task.task_type {
            TaskType::Forecasting => self.simulate_forecasting_task(task)?,
            TaskType::FeatureExtraction => self.simulate_feature_extraction_task(task)?,
            TaskType::AnomalyDetection => self.simulate_anomaly_detection_task(task)?,
            TaskType::Decomposition => self.simulate_decomposition_task(task)?,
            _ => {
                // Generic processing
                task.input_data.clone()
            }
        };

        let execution_time = start_time.elapsed();

        // Update node status
        if let Some(node) = self.nodes.get_mut(node_address) {
            node.running_tasks = node.running_tasks.saturating_sub(1);
            node.current_load = node.running_tasks as f64 / self.config.max_concurrent_tasks as f64;
        }

        Ok(TaskResult {
            taskid: task.id.clone(),
            status: TaskStatus::Completed,
            data: Some(result_data),
            metrics: TaskMetrics {
                execution_time,
                executed_on: node_address.to_string(),
                memory_usage: task.input_data.len() * std::mem::size_of::<F>(),
                cpu_utilization: 0.8,                    // Simulated
                network_time: Duration::from_millis(10), // Simulated
            },
            error: None,
        })
    }

    /// Simulate forecasting task execution
    fn simulate_forecasting_task(&self, task: &DistributedTask<F>) -> Result<Array1<F>> {
        let horizon = task
            .parameters
            .get("horizon")
            .map(|&h| h as usize)
            .unwrap_or(10);

        // Simple linear extrapolation for simulation
        let data = &task.input_data;
        if data.len() < 2 {
            return Ok(Array1::zeros(horizon));
        }

        let slope = (data[data.len() - 1] - data[data.len() - 2]) / F::one();
        let mut forecast = Array1::zeros(horizon);

        for i in 0..horizon {
            forecast[i] = data[data.len() - 1] + slope * F::from(i + 1).unwrap();
        }

        Ok(forecast)
    }

    /// Simulate feature extraction task execution
    fn simulate_feature_extraction_task(&self, task: &DistributedTask<F>) -> Result<Array1<F>> {
        let data = &task.input_data;

        // Extract basic statistical features
        let mean = data.mean().unwrap_or(F::zero());
        let variance = data.var(F::zero());
        let min = data.iter().fold(F::infinity(), |acc, &x| acc.min(x));
        let max = data.iter().fold(F::neg_infinity(), |acc, &x| acc.max(x));

        // Simulate more features
        let features = vec![
            mean,
            variance.sqrt(), // Standard deviation
            min,
            max,
            max - min, // Range
        ];

        Ok(Array1::from_vec(features))
    }

    /// Simulate anomaly detection task execution
    fn simulate_anomaly_detection_task(&self, task: &DistributedTask<F>) -> Result<Array1<F>> {
        let data = &task.input_data;
        let mean = data.mean().unwrap_or(F::zero());
        let std_dev = data.var(F::zero()).sqrt();

        // Simple z-score based anomaly detection
        let threshold = F::from(3.0).unwrap();
        let mut anomaly_scores = Array1::zeros(data.len());

        for (i, &value) in data.iter().enumerate() {
            let z_score = (value - mean) / std_dev;
            anomaly_scores[i] = if z_score.abs() > threshold {
                F::one()
            } else {
                F::zero()
            };
        }

        Ok(anomaly_scores)
    }

    /// Simulate decomposition task execution
    fn simulate_decomposition_task(&self, task: &DistributedTask<F>) -> Result<Array1<F>> {
        // Simple trend extraction using moving average
        let data = &task.input_data;
        let window_size = 10.min(data.len() / 2);
        let mut trend = Array1::zeros(data.len());

        for i in 0..data.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(data.len());

            let window_sum = data.slice(scirs2_core::ndarray::s![start..end]).sum();
            let window_len = F::from(end - start).unwrap();
            trend[i] = window_sum / window_len;
        }

        Ok(trend)
    }

    /// Aggregate forecasting results from multiple tasks
    fn aggregate_forecast_results(&self, horizon: usize) -> Result<Array1<F>> {
        let mut all_forecasts = Vec::new();
        let mut chunk_indices = Vec::new();

        // Collect all forecast results
        for (taskid, result) in &self.completed_tasks {
            if taskid.starts_with("forecast_chunk_") && result.status == TaskStatus::Completed {
                if let Some(data) = &result.data {
                    all_forecasts.push(data.clone());

                    // Extract chunk index for proper ordering
                    if let Some(chunk_str) = taskid.strip_prefix("forecast_chunk_") {
                        if let Ok(index) = chunk_str.parse::<usize>() {
                            chunk_indices.push(index);
                        }
                    }
                }
            }
        }

        if all_forecasts.is_empty() {
            return Ok(Array1::zeros(horizon));
        }

        // Sort forecasts by chunk index
        let mut indexed_forecasts: Vec<(usize, Array1<F>)> =
            chunk_indices.into_iter().zip(all_forecasts).collect();
        indexed_forecasts.sort_by_key(|(index_, _)| *index_);

        // Aggregate by averaging (simple ensemble approach)
        let mut final_forecast = Array1::zeros(horizon);
        let mut count = 0;

        for (_, forecast) in indexed_forecasts {
            let actual_horizon = forecast.len().min(horizon);
            for i in 0..actual_horizon {
                final_forecast[i] = final_forecast[i] + forecast[i];
            }
            count += 1;
        }

        if count > 0 {
            final_forecast = final_forecast / F::from(count).unwrap();
        }

        Ok(final_forecast)
    }

    /// Aggregate feature extraction results
    fn aggregate_feature_results(&self, numfeatures: usize) -> Result<Array2<F>> {
        let mut all_features = Vec::new();
        let mut window_indices = Vec::new();

        // Collect all feature results
        for (taskid, result) in &self.completed_tasks {
            if taskid.starts_with("features_window_") && result.status == TaskStatus::Completed {
                if let Some(data) = &result.data {
                    all_features.push(data.clone());

                    // Extract window index
                    if let Some(window_str) = taskid.strip_prefix("features_window_") {
                        if let Ok(index) = window_str.parse::<usize>() {
                            window_indices.push(index);
                        }
                    }
                }
            }
        }

        if all_features.is_empty() {
            return Ok(Array2::zeros((0, numfeatures)));
        }

        // Sort by window index
        let mut indexed_features: Vec<(usize, Array1<F>)> =
            window_indices.into_iter().zip(all_features).collect();
        indexed_features.sort_by_key(|(index_, _)| *index_);

        // Combine into matrix
        let num_windows = indexed_features.len();
        let feature_size = indexed_features[0].1.len().min(numfeatures);
        let mut result = Array2::zeros((num_windows, feature_size));

        for (row, (_, features)) in indexed_features.iter().enumerate() {
            for col in 0..feature_size {
                if col < features.len() {
                    result[[row, col]] = features[col];
                }
            }
        }

        Ok(result)
    }

    /// Get cluster status information
    pub fn get_cluster_status(&self) -> ClusterStatus {
        let total_nodes = self.nodes.len();
        let available_nodes = self
            .nodes
            .values()
            .filter(|node| node.status == NodeStatus::Available)
            .count();

        let total_running_tasks = self.running_tasks.len();
        let total_completed_tasks = self.completed_tasks.len();
        let total_queued_tasks = self.task_queue.len();

        let average_load = if total_nodes > 0 {
            self.nodes
                .values()
                .map(|node| node.current_load)
                .sum::<f64>()
                / total_nodes as f64
        } else {
            0.0
        };

        ClusterStatus {
            total_nodes,
            available_nodes,
            total_running_tasks,
            total_completed_tasks,
            total_queued_tasks,
            average_load,
            nodes: self.nodes.clone(),
        }
    }

    /// Clear completed tasks to free memory
    pub fn clear_completed_tasks(&mut self) {
        self.completed_tasks.clear();
    }

    /// Cancel a running task
    pub fn cancel_task(&mut self, taskid: &str) -> Result<()> {
        if let Some(_task) = self.running_tasks.remove(taskid) {
            // Add cancelled result
            self.completed_tasks.insert(
                taskid.to_string(),
                TaskResult {
                    taskid: taskid.to_string(),
                    status: TaskStatus::Cancelled,
                    data: None,
                    metrics: TaskMetrics::default(),
                    error: Some("Task cancelled by user".to_string()),
                },
            );
            Ok(())
        } else {
            Err(TimeSeriesError::InvalidInput(format!(
                "Task {taskid} not found in running tasks"
            )))
        }
    }
}

/// Cluster status information
#[derive(Debug, Clone)]
pub struct ClusterStatus {
    /// Total number of nodes in cluster
    pub total_nodes: usize,
    /// Number of available nodes
    pub available_nodes: usize,
    /// Number of currently running tasks
    pub total_running_tasks: usize,
    /// Number of completed tasks
    pub total_completed_tasks: usize,
    /// Number of tasks in queue
    pub total_queued_tasks: usize,
    /// Average load across all nodes
    pub average_load: f64,
    /// Detailed node information
    pub nodes: HashMap<String, NodeInfo>,
}

/// Convenience functions for common distributed operations
#[allow(dead_code)]
pub fn distributed_moving_average<
    F: Float
        + Debug
        + Clone
        + scirs2_core::numeric::FromPrimitive
        + scirs2_core::numeric::Zero
        + scirs2_core::ndarray::ScalarOperand,
>(
    processor: &mut DistributedProcessor<F>,
    data: &Array1<F>,
    window_size: usize,
) -> Result<Array1<F>> {
    // Create custom task for moving average computation
    let task = DistributedTask {
        id: "moving_average".to_string(),
        task_type: TaskType::Custom("moving_average".to_string()),
        input_data: data.clone(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("window_size".to_string(), window_size as f64);
            params
        },
        priority: TaskPriority::Normal,
        dependencies: vec![],
    };

    processor.submit_task(task)?;
    processor.process_pending_tasks()?;

    // Get result
    if let Some(result) = processor.completed_tasks.get("moving_average") {
        if let Some(data) = &result.data {
            Ok(data.clone())
        } else {
            Err(TimeSeriesError::ComputationError(
                "Moving average computation failed".to_string(),
            ))
        }
    } else {
        Err(TimeSeriesError::ComputationError(
            "Moving average task not found".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cluster_config_default() {
        let config = ClusterConfig::default();
        assert_eq!(config.nodes.len(), 1);
        assert_eq!(config.max_concurrent_tasks, 4);
        assert_eq!(config.chunk_size, 10000);
    }

    #[test]
    fn test_distributed_processor_creation() {
        let config = ClusterConfig::default();
        let processor: DistributedProcessor<f64> = DistributedProcessor::new(config);

        assert_eq!(processor.nodes.len(), 1);
        assert!(processor.task_queue.is_empty());
        assert!(processor.running_tasks.is_empty());
        assert!(processor.completed_tasks.is_empty());
    }

    #[test]
    fn test_task_submission() {
        let config = ClusterConfig::default();
        let mut processor: DistributedProcessor<f64> = DistributedProcessor::new(config);

        let task = DistributedTask {
            id: "test_task".to_string(),
            task_type: TaskType::Forecasting,
            input_data: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            parameters: HashMap::new(),
            priority: TaskPriority::Normal,
            dependencies: vec![],
        };

        assert!(processor.submit_task(task).is_ok());
        assert_eq!(processor.task_queue.len(), 1);
    }

    #[test]
    fn test_task_priority_ordering() {
        let config = ClusterConfig::default();
        let mut processor: DistributedProcessor<f64> = DistributedProcessor::new(config);

        // Submit tasks with different priorities
        let low_task = DistributedTask {
            id: "low".to_string(),
            task_type: TaskType::Forecasting,
            input_data: Array1::zeros(10),
            parameters: HashMap::new(),
            priority: TaskPriority::Low,
            dependencies: vec![],
        };

        let high_task = DistributedTask {
            id: "high".to_string(),
            task_type: TaskType::Forecasting,
            input_data: Array1::zeros(10),
            parameters: HashMap::new(),
            priority: TaskPriority::High,
            dependencies: vec![],
        };

        processor.submit_task(low_task).unwrap();
        processor.submit_task(high_task).unwrap();

        // High priority task should be first
        assert_eq!(processor.task_queue[0].priority, TaskPriority::High);
        assert_eq!(processor.task_queue[1].priority, TaskPriority::Low);
    }

    #[test]
    fn test_distributed_forecasting() {
        let config = ClusterConfig::default();
        let mut processor: DistributedProcessor<f64> = DistributedProcessor::new(config);

        let data = Array1::from_vec((1..100).map(|x| x as f64).collect());
        let horizon = 10;

        let result = processor.distributed_forecast(&data, horizon, "linear");
        assert!(result.is_ok());

        let forecast = result.unwrap();
        assert_eq!(forecast.len(), horizon);
    }

    #[test]
    fn test_cluster_status() {
        let config = ClusterConfig::default();
        let processor: DistributedProcessor<f64> = DistributedProcessor::new(config);

        let status = processor.get_cluster_status();
        assert_eq!(status.total_nodes, 1);
        assert_eq!(status.available_nodes, 1);
        assert_eq!(status.total_running_tasks, 0);
        assert_eq!(status.total_completed_tasks, 0);
        assert_eq!(status.total_queued_tasks, 0);
    }

    #[test]
    fn test_load_balancing_strategies() {
        // Test that different strategies are properly defined
        assert_ne!(
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LoadBased
        );
        assert_ne!(
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::Weighted
        );
    }

    #[test]
    fn test_task_status_enum() {
        assert_eq!(TaskStatus::Pending, TaskStatus::Pending);
        assert_ne!(TaskStatus::Running, TaskStatus::Completed);
    }

    #[test]
    fn test_fault_tolerance_config() {
        let config = FaultToleranceConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.replication_factor, 2);
        assert!(!config.enable_replication);
    }
}
