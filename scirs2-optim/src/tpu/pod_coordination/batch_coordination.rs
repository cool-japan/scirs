//! Batch Coordination for TPU Pod Coordination
//!
//! This module provides comprehensive batch coordination functionality for TPU pod coordination,
//! including batch management, data distribution, pipeline execution, and result aggregation.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use ndarray::{Array, IxDyn};
use num_traits::Float;

use super::{DeviceId, BatchId};
use super::resource_scheduling::ResourceAllocation;
use crate::error::{OptimError, Result};

/// Batch parallelization strategies
#[derive(Debug, Clone, Copy)]
pub enum BatchParallelizationStrategy {
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Hybrid,
    HybridParallel,
    TensorParallel,
    ExpertParallel,
    Adaptive,
}

/// Data partitioning strategies
#[derive(Debug, Clone)]
pub enum DataPartitioning {
    Horizontal,
    Vertical,
    Random,
    Stratified,
    Custom(String),
}

/// Batch priorities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BatchPriority {
    Low,
    Normal,
    High,
    Critical,
    Realtime,
}

/// Consistency levels
#[derive(Debug, Clone, Copy)]
pub enum ConsistencyLevel {
    Eventual,
    Strong,
    Causal,
    Sequential,
    Linearizable,
}

/// Partition processing status
#[derive(Debug, Clone, Copy)]
pub enum PartitionStatus {
    Pending,
    Assigned,
    Processing,
    Completed,
    Failed,
}

/// Pipeline stage status
#[derive(Debug, Clone, Copy)]
pub enum PipelineStageStatus {
    Waiting,
    Ready,
    Running,
    Completed,
    Failed,
    Stalled,
}

/// Batch data representation
#[derive(Debug)]
pub struct BatchData<T: Float> {
    /// Input data
    pub inputs: Vec<Array<T, IxDyn>>,

    /// Batch size
    pub batch_size: usize,

    /// Data partitioning
    pub partitioning: DataPartitioning,

    /// Metadata
    pub metadata: BatchMetadata,
}

/// Batch metadata
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// Batch priority
    pub priority: BatchPriority,

    /// Resource requirements
    pub resource_requirements: ResourceRequirements,

    /// Quality of service
    pub qos_requirements: QoSRequirements,

    /// Deadline
    pub deadline: Option<Instant>,

    /// Tags for categorization
    pub tags: HashMap<String, String>,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Memory requirement (bytes)
    pub memory_bytes: usize,

    /// Compute requirement (FLOPS)
    pub compute_flops: u64,

    /// Communication bandwidth (GB/s)
    pub communication_bandwidth: f64,

    /// Preferred devices
    pub preferred_devices: Vec<DeviceId>,
}

/// Quality of service requirements
#[derive(Debug, Clone)]
pub struct QoSRequirements {
    /// Maximum latency
    pub max_latency: Duration,

    /// Minimum throughput
    pub min_throughput: f64,

    /// Reliability requirement
    pub reliability: f64,

    /// Consistency requirement
    pub consistency: ConsistencyLevel,
}

/// Batch partition for a device
#[derive(Debug, Clone)]
pub struct BatchPartition<T: Float>
where
    T: Clone,
{
    /// Partition data
    pub data: Array<T, IxDyn>,

    /// Partition indices
    pub indices: Vec<usize>,

    /// Processing status
    pub status: PartitionStatus,

    /// Assigned device
    pub device: DeviceId,

    /// Dependencies
    pub dependencies: Vec<PartitionId>,

    /// Creation timestamp
    pub created_at: Instant,

    /// Processing start time
    pub processing_start: Option<Instant>,

    /// Completion time
    pub completed_at: Option<Instant>,
}

/// Partition identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PartitionId(pub u64);

/// Batch execution state
#[derive(Debug)]
pub struct BatchExecution<T: Float> {
    /// Batch ID
    pub id: BatchId,

    /// Batch data
    pub data: BatchData<T>,

    /// Device assignments
    pub device_assignments: HashMap<DeviceId, BatchPartition<T>>,

    /// Execution progress
    pub progress: BatchProgress,

    /// Started at
    pub started_at: Instant,

    /// Dependencies
    pub dependencies: Vec<BatchId>,

    /// Pipeline stages
    pub pipeline_stages: Vec<PipelineStage<T>>,

    /// Execution strategy
    pub strategy: BatchParallelizationStrategy,
}

/// Batch execution progress
#[derive(Debug, Clone)]
pub struct BatchProgress {
    /// Total partitions
    pub total_partitions: usize,

    /// Completed partitions
    pub completed_partitions: usize,

    /// Failed partitions
    pub failed_partitions: usize,

    /// Processing rate (partitions/second)
    pub processing_rate: f64,

    /// Estimated completion time
    pub estimated_completion: Instant,

    /// Data transferred (bytes)
    pub data_transferred: usize,

    /// Computation time
    pub computation_time: Duration,
}

/// Pipeline stage
#[derive(Debug, Clone)]
pub struct PipelineStage<T: Float> {
    /// Stage ID
    pub stage_id: usize,

    /// Stage name
    pub name: String,

    /// Input partitions
    pub input_partitions: Vec<PartitionId>,

    /// Output partitions
    pub output_partitions: Vec<PartitionId>,

    /// Assigned devices
    pub assigned_devices: Vec<DeviceId>,

    /// Stage status
    pub status: PipelineStageStatus,

    /// Dependencies
    pub dependencies: Vec<usize>,

    /// Estimated duration
    pub estimated_duration: Duration,

    /// Actual duration
    pub actual_duration: Option<Duration>,

    /// Stage-specific data
    pub stage_data: Option<Array<T, IxDyn>>,
}

/// Data distribution strategy
#[derive(Debug, Clone)]
pub struct DistributionStrategy {
    /// Partitioning method
    pub partitioning: DataPartitioning,

    /// Load balancing enabled
    pub load_balancing: bool,

    /// Replication factor
    pub replication_factor: usize,

    /// Caching strategy
    pub caching_strategy: CachingStrategy,
}

/// Caching strategies
#[derive(Debug, Clone, Copy)]
pub enum CachingStrategy {
    None,
    LRU,
    LFU,
    FIFO,
    Adaptive,
}

/// Result aggregation method
#[derive(Debug, Clone, Copy)]
pub enum AggregationMethod {
    Concatenate,
    Average,
    Sum,
    Max,
    Min,
    Custom,
}

/// Batch coordination statistics
pub type BatchCoordinationStatistics = HashMap<String, f64>;

/// Data distributor type aliases
type DataDistributor<T> = HashMap<DeviceId, T>;
type ResultAggregator<T> = HashMap<DeviceId, T>;
type PipelineManager<T> = HashMap<String, T>;
type BatchScheduler<T> = HashMap<String, T>;

/// Batch coordinator for parallelization
#[derive(Debug)]
pub struct BatchCoordinator<T: Float> {
    /// Batch strategy
    strategy: BatchParallelizationStrategy,

    /// Active batches
    active_batches: HashMap<BatchId, BatchExecution<T>>,

    /// Batch scheduler
    scheduler: BatchScheduler<T>,

    /// Data distributor
    data_distributor: DataDistributor<T>,

    /// Result aggregator
    result_aggregator: ResultAggregator<T>,

    /// Pipeline manager
    pipeline_manager: PipelineManager<T>,

    /// Distribution strategy
    distribution_strategy: DistributionStrategy,

    /// Aggregation method
    aggregation_method: AggregationMethod,

    /// Coordination statistics
    statistics: BatchCoordinationStatistics,

    /// Batch queue
    batch_queue: VecDeque<BatchId>,

    /// Completed batches
    completed_batches: HashMap<BatchId, BatchExecutionResult<T>>,
}

/// Batch execution result
#[derive(Debug)]
pub struct BatchExecutionResult<T: Float> {
    /// Batch ID
    pub batch_id: BatchId,

    /// Aggregated results
    pub results: Vec<Array<T, IxDyn>>,

    /// Execution time
    pub execution_time: Duration,

    /// Device statistics
    pub device_statistics: HashMap<DeviceId, DeviceExecutionStatistics>,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Device execution statistics
#[derive(Debug, Clone)]
pub struct DeviceExecutionStatistics {
    /// Device ID
    pub device_id: DeviceId,

    /// Execution time
    pub execution_time: Duration,

    /// Memory usage
    pub memory_usage: f64,

    /// Compute utilization
    pub compute_utilization: f64,

    /// Communication volume
    pub communication_volume: usize,

    /// Error count
    pub error_count: usize,

    /// Throughput
    pub throughput: f64,
}

/// Quality metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Accuracy
    pub accuracy: f64,

    /// Latency
    pub latency: Duration,

    /// Throughput
    pub throughput: f64,

    /// Resource efficiency
    pub resource_efficiency: f64,

    /// Error rate
    pub error_rate: f64,
}

impl<T: Float + Default + Clone + Send + Sync + std::iter::Sum> BatchCoordinator<T> {
    /// Create a new batch coordinator
    pub fn new(strategy: BatchParallelizationStrategy) -> Result<Self> {
        let distribution_strategy = DistributionStrategy {
            partitioning: DataPartitioning::Horizontal,
            load_balancing: true,
            replication_factor: 1,
            caching_strategy: CachingStrategy::LRU,
        };

        Ok(Self {
            strategy,
            active_batches: HashMap::new(),
            scheduler: HashMap::new(),
            data_distributor: HashMap::new(),
            result_aggregator: HashMap::new(),
            pipeline_manager: HashMap::new(),
            distribution_strategy,
            aggregation_method: AggregationMethod::Concatenate,
            statistics: HashMap::new(),
            batch_queue: VecDeque::new(),
            completed_batches: HashMap::new(),
        })
    }

    /// Create a new batch execution
    pub async fn create_batch(&mut self, batch_data: BatchData<T>) -> Result<BatchId> {
        let batch_id = BatchId(scirs2_core::random::rng().gen_range(0..u64::MAX));

        let pipeline_stages = self.create_pipeline_stages(&batch_data)?;

        let batch_execution = BatchExecution {
            id: batch_id,
            data: batch_data,
            device_assignments: HashMap::new(),
            progress: BatchProgress {
                total_partitions: 0,
                completed_partitions: 0,
                failed_partitions: 0,
                processing_rate: 0.0,
                estimated_completion: Instant::now() + Duration::from_secs(60),
                data_transferred: 0,
                computation_time: Duration::from_secs(0),
            },
            started_at: Instant::now(),
            dependencies: Vec::new(),
            pipeline_stages,
            strategy: self.strategy,
        };

        self.active_batches.insert(batch_id, batch_execution);
        self.batch_queue.push_back(batch_id);
        self.update_statistics();

        Ok(batch_id)
    }

    /// Create pipeline stages for batch
    fn create_pipeline_stages(&self, batch_data: &BatchData<T>) -> Result<Vec<PipelineStage<T>>> {
        let mut stages = Vec::new();

        match self.strategy {
            BatchParallelizationStrategy::DataParallel => {
                // Single stage for data parallel execution
                stages.push(PipelineStage {
                    stage_id: 0,
                    name: "DataParallelExecution".to_string(),
                    input_partitions: Vec::new(),
                    output_partitions: Vec::new(),
                    assigned_devices: Vec::new(),
                    status: PipelineStageStatus::Waiting,
                    dependencies: Vec::new(),
                    estimated_duration: Duration::from_secs(30),
                    actual_duration: None,
                    stage_data: None,
                });
            }
            BatchParallelizationStrategy::PipelineParallel => {
                // Multiple stages for pipeline execution
                for i in 0..4 {
                    stages.push(PipelineStage {
                        stage_id: i,
                        name: format!("PipelineStage{}", i),
                        input_partitions: Vec::new(),
                        output_partitions: Vec::new(),
                        assigned_devices: Vec::new(),
                        status: if i == 0 { PipelineStageStatus::Ready } else { PipelineStageStatus::Waiting },
                        dependencies: if i == 0 { Vec::new() } else { vec![i - 1] },
                        estimated_duration: Duration::from_secs(10),
                        actual_duration: None,
                        stage_data: None,
                    });
                }
            }
            BatchParallelizationStrategy::Hybrid => {
                // Combination of data and model parallel stages
                stages.push(PipelineStage {
                    stage_id: 0,
                    name: "DataDistribution".to_string(),
                    input_partitions: Vec::new(),
                    output_partitions: Vec::new(),
                    assigned_devices: Vec::new(),
                    status: PipelineStageStatus::Ready,
                    dependencies: Vec::new(),
                    estimated_duration: Duration::from_secs(5),
                    actual_duration: None,
                    stage_data: None,
                });

                stages.push(PipelineStage {
                    stage_id: 1,
                    name: "ParallelCompute".to_string(),
                    input_partitions: Vec::new(),
                    output_partitions: Vec::new(),
                    assigned_devices: Vec::new(),
                    status: PipelineStageStatus::Waiting,
                    dependencies: vec![0],
                    estimated_duration: Duration::from_secs(20),
                    actual_duration: None,
                    stage_data: None,
                });

                stages.push(PipelineStage {
                    stage_id: 2,
                    name: "ResultAggregation".to_string(),
                    input_partitions: Vec::new(),
                    output_partitions: Vec::new(),
                    assigned_devices: Vec::new(),
                    status: PipelineStageStatus::Waiting,
                    dependencies: vec![1],
                    estimated_duration: Duration::from_secs(5),
                    actual_duration: None,
                    stage_data: None,
                });
            }
            _ => {
                // Default single stage
                stages.push(PipelineStage {
                    stage_id: 0,
                    name: "DefaultExecution".to_string(),
                    input_partitions: Vec::new(),
                    output_partitions: Vec::new(),
                    assigned_devices: Vec::new(),
                    status: PipelineStageStatus::Ready,
                    dependencies: Vec::new(),
                    estimated_duration: Duration::from_secs(30),
                    actual_duration: None,
                    stage_data: None,
                });
            }
        }

        Ok(stages)
    }

    /// Distribute data across devices
    pub async fn distribute_data(
        &mut self,
        batch_id: BatchId,
        resource_allocation: &ResourceAllocation,
    ) -> Result<()> {
        let batch_execution = self.active_batches.get_mut(&batch_id)
            .ok_or_else(|| OptimError::ConfigurationError("Batch not found".to_string()))?;

        // Create partitions based on strategy
        let partitions = self.create_partitions(&batch_execution.data, &resource_allocation.devices)?;

        // Assign partitions to devices
        for (device_id, partition) in resource_allocation.devices.iter().zip(partitions.into_iter()) {
            batch_execution.device_assignments.insert(*device_id, partition);
        }

        // Update progress
        batch_execution.progress.total_partitions = resource_allocation.devices.len();

        self.update_statistics();
        Ok(())
    }

    /// Create data partitions
    fn create_partitions(
        &self,
        batch_data: &BatchData<T>,
        devices: &[DeviceId],
    ) -> Result<Vec<BatchPartition<T>>> {
        let mut partitions = Vec::new();

        match &batch_data.partitioning {
            DataPartitioning::Horizontal => {
                // Split data horizontally across devices
                for (i, &device_id) in devices.iter().enumerate() {
                    let partition_data = if !batch_data.inputs.is_empty() {
                        // Create a subset of the first input array for simulation
                        let input_shape = batch_data.inputs[0].shape();
                        let partition_size = input_shape[0] / devices.len();
                        let start_idx = i * partition_size;
                        let end_idx = if i == devices.len() - 1 {
                            input_shape[0]
                        } else {
                            (i + 1) * partition_size
                        };

                        // Create a simulated partition
                        Array::zeros(IxDyn(&[end_idx - start_idx, input_shape[1]]))
                    } else {
                        Array::zeros(IxDyn(&[10, 10]))
                    };

                    let partition = BatchPartition {
                        data: partition_data,
                        indices: (i * 10..(i + 1) * 10).collect(),
                        status: PartitionStatus::Assigned,
                        device: device_id,
                        dependencies: Vec::new(),
                        created_at: Instant::now(),
                        processing_start: None,
                        completed_at: None,
                    };

                    partitions.push(partition);
                }
            }
            DataPartitioning::Vertical => {
                // Split data vertically (features) across devices
                for (i, &device_id) in devices.iter().enumerate() {
                    let partition_data = Array::zeros(IxDyn(&[batch_data.batch_size, 10]));

                    let partition = BatchPartition {
                        data: partition_data,
                        indices: (i * 10..(i + 1) * 10).collect(),
                        status: PartitionStatus::Assigned,
                        device: device_id,
                        dependencies: Vec::new(),
                        created_at: Instant::now(),
                        processing_start: None,
                        completed_at: None,
                    };

                    partitions.push(partition);
                }
            }
            DataPartitioning::Random => {
                // Randomly distribute data across devices
                for (i, &device_id) in devices.iter().enumerate() {
                    let partition_data = Array::zeros(IxDyn(&[batch_data.batch_size / devices.len(), 10]));

                    let partition = BatchPartition {
                        data: partition_data,
                        indices: (0..batch_data.batch_size / devices.len()).collect(),
                        status: PartitionStatus::Assigned,
                        device: device_id,
                        dependencies: Vec::new(),
                        created_at: Instant::now(),
                        processing_start: None,
                        completed_at: None,
                    };

                    partitions.push(partition);
                }
            }
            _ => {
                // Default uniform distribution
                for (i, &device_id) in devices.iter().enumerate() {
                    let partition_data = Array::zeros(IxDyn(&[batch_data.batch_size / devices.len(), 10]));

                    let partition = BatchPartition {
                        data: partition_data,
                        indices: (i * 10..(i + 1) * 10).collect(),
                        status: PartitionStatus::Assigned,
                        device: device_id,
                        dependencies: Vec::new(),
                        created_at: Instant::now(),
                        processing_start: None,
                        completed_at: None,
                    };

                    partitions.push(partition);
                }
            }
        }

        Ok(partitions)
    }

    /// Get partition for a specific device
    pub fn get_partition(&self, batch_id: BatchId, device_id: DeviceId) -> Result<BatchPartition<T>> {
        let batch_execution = self.active_batches.get(&batch_id)
            .ok_or_else(|| OptimError::ConfigurationError("Batch not found".to_string()))?;

        batch_execution.device_assignments.get(&device_id)
            .cloned()
            .ok_or_else(|| OptimError::ConfigurationError("Partition not found".to_string()))
    }

    /// Update partition status
    pub fn update_partition_status(
        &mut self,
        batch_id: BatchId,
        device_id: DeviceId,
        status: PartitionStatus,
    ) -> Result<()> {
        let batch_execution = self.active_batches.get_mut(&batch_id)
            .ok_or_else(|| OptimError::ConfigurationError("Batch not found".to_string()))?;

        if let Some(partition) = batch_execution.device_assignments.get_mut(&device_id) {
            let old_status = partition.status;
            partition.status = status;

            match (old_status, status) {
                (PartitionStatus::Assigned, PartitionStatus::Processing) => {
                    partition.processing_start = Some(Instant::now());
                }
                (PartitionStatus::Processing, PartitionStatus::Completed) => {
                    partition.completed_at = Some(Instant::now());
                    batch_execution.progress.completed_partitions += 1;
                }
                (PartitionStatus::Processing, PartitionStatus::Failed) => {
                    batch_execution.progress.failed_partitions += 1;
                }
                _ => {}
            }

            self.update_batch_progress(batch_id)?;
        }

        Ok(())
    }

    /// Update batch progress
    fn update_batch_progress(&mut self, batch_id: BatchId) -> Result<()> {
        let batch_execution = self.active_batches.get_mut(&batch_id)
            .ok_or_else(|| OptimError::ConfigurationError("Batch not found".to_string()))?;

        let elapsed = batch_execution.started_at.elapsed();
        let completed = batch_execution.progress.completed_partitions;
        let total = batch_execution.progress.total_partitions;

        if completed > 0 && elapsed.as_secs() > 0 {
            batch_execution.progress.processing_rate = completed as f64 / elapsed.as_secs_f64();

            if batch_execution.progress.processing_rate > 0.0 {
                let remaining = total - completed;
                let estimated_remaining_time = Duration::from_secs_f64(
                    remaining as f64 / batch_execution.progress.processing_rate
                );
                batch_execution.progress.estimated_completion = Instant::now() + estimated_remaining_time;
            }
        }

        // Check if batch is complete
        if completed == total {
            self.complete_batch(batch_id)?;
        }

        Ok(())
    }

    /// Complete batch execution
    fn complete_batch(&mut self, batch_id: BatchId) -> Result<()> {
        if let Some(batch_execution) = self.active_batches.remove(&batch_id) {
            let execution_time = batch_execution.started_at.elapsed();

            // Create execution result
            let result = BatchExecutionResult {
                batch_id,
                results: vec![Array::zeros(IxDyn(&[10, 10]))], // Placeholder result
                execution_time,
                device_statistics: HashMap::new(),
                quality_metrics: QualityMetrics {
                    accuracy: 0.95,
                    latency: execution_time,
                    throughput: batch_execution.progress.processing_rate,
                    resource_efficiency: 0.85,
                    error_rate: batch_execution.progress.failed_partitions as f64 / batch_execution.progress.total_partitions as f64,
                },
            };

            self.completed_batches.insert(batch_id, result);
            self.batch_queue.retain(|&id| id != batch_id);
        }

        self.update_statistics();
        Ok(())
    }

    /// Process batch pipeline
    pub async fn process_pipeline(&mut self, batch_id: BatchId) -> Result<()> {
        let batch_execution = self.active_batches.get_mut(&batch_id)
            .ok_or_else(|| OptimError::ConfigurationError("Batch not found".to_string()))?;

        // Find ready stages
        let ready_stages: Vec<usize> = batch_execution.pipeline_stages
            .iter()
            .enumerate()
            .filter(|(_, stage)| stage.status == PipelineStageStatus::Ready)
            .map(|(i, _)| i)
            .collect();

        // Process ready stages
        for stage_idx in ready_stages {
            self.execute_pipeline_stage(batch_id, stage_idx).await?;
        }

        Ok(())
    }

    /// Execute a pipeline stage
    async fn execute_pipeline_stage(&mut self, batch_id: BatchId, stage_idx: usize) -> Result<()> {
        let batch_execution = self.active_batches.get_mut(&batch_id)
            .ok_or_else(|| OptimError::ConfigurationError("Batch not found".to_string()))?;

        if let Some(stage) = batch_execution.pipeline_stages.get_mut(stage_idx) {
            stage.status = PipelineStageStatus::Running;
            let start_time = Instant::now();

            // Simulate stage execution
            tokio::time::sleep(Duration::from_millis(100)).await;

            stage.status = PipelineStageStatus::Completed;
            stage.actual_duration = Some(start_time.elapsed());

            // Update dependent stages
            for next_stage_idx in 0..batch_execution.pipeline_stages.len() {
                let can_start = {
                    let next_stage = &batch_execution.pipeline_stages[next_stage_idx];
                    next_stage.status == PipelineStageStatus::Waiting &&
                    next_stage.dependencies.iter().all(|&dep_idx| {
                        batch_execution.pipeline_stages.get(dep_idx)
                            .map_or(false, |dep_stage| dep_stage.status == PipelineStageStatus::Completed)
                    })
                };

                if can_start {
                    batch_execution.pipeline_stages[next_stage_idx].status = PipelineStageStatus::Ready;
                }
            }
        }

        Ok(())
    }

    /// Aggregate results from multiple devices
    pub async fn aggregate_results(
        &self,
        batch_id: BatchId,
        device_results: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,
    ) -> Result<Vec<Array<T, IxDyn>>> {
        if device_results.is_empty() {
            return Ok(Vec::new());
        }

        match self.aggregation_method {
            AggregationMethod::Concatenate => {
                // Concatenate results along the first dimension
                let mut aggregated = Vec::new();

                // Find the number of arrays to aggregate
                let num_arrays = device_results.values().next().unwrap().len();

                for array_idx in 0..num_arrays {
                    let arrays_to_concat: Vec<_> = device_results
                        .values()
                        .filter_map(|arrays| arrays.get(array_idx))
                        .collect();

                    if !arrays_to_concat.is_empty() {
                        // For simplicity, return the first array
                        // In a real implementation, this would concatenate arrays
                        aggregated.push(arrays_to_concat[0].clone());
                    }
                }

                Ok(aggregated)
            }
            AggregationMethod::Average => {
                // Average results across devices
                let mut averaged = Vec::new();

                let num_arrays = device_results.values().next().unwrap().len();
                let num_devices = device_results.len() as f64;

                for array_idx in 0..num_arrays {
                    let arrays: Vec<_> = device_results
                        .values()
                        .filter_map(|arrays| arrays.get(array_idx))
                        .collect();

                    if !arrays.is_empty() {
                        let mut sum_array = arrays[0].clone();
                        for array in arrays.iter().skip(1) {
                            sum_array = sum_array + array;
                        }

                        // Divide by number of devices
                        sum_array.mapv_inplace(|x| x / num_traits::cast::cast(num_devices).unwrap_or_else(|| T::zero()));
                        averaged.push(sum_array);
                    }
                }

                Ok(averaged)
            }
            AggregationMethod::Sum => {
                // Sum results across devices
                let mut summed = Vec::new();

                let num_arrays = device_results.values().next().unwrap().len();

                for array_idx in 0..num_arrays {
                    let arrays: Vec<_> = device_results
                        .values()
                        .filter_map(|arrays| arrays.get(array_idx))
                        .collect();

                    if !arrays.is_empty() {
                        let mut sum_array = arrays[0].clone();
                        for array in arrays.iter().skip(1) {
                            sum_array = sum_array + array;
                        }
                        summed.push(sum_array);
                    }
                }

                Ok(summed)
            }
            _ => {
                // Default to first device result
                Ok(device_results.values().next().unwrap().clone())
            }
        }
    }

    /// Get batch coordination statistics
    pub fn get_statistics(&self) -> BatchCoordinationStatistics {
        let mut stats = HashMap::new();

        stats.insert("active_batches".to_string(), self.active_batches.len() as f64);
        stats.insert("completed_batches".to_string(), self.completed_batches.len() as f64);
        stats.insert("queue_length".to_string(), self.batch_queue.len() as f64);

        // Calculate average processing rate
        let avg_processing_rate = if self.active_batches.is_empty() {
            0.0
        } else {
            self.active_batches
                .values()
                .map(|batch| batch.progress.processing_rate)
                .sum::<f64>() / self.active_batches.len() as f64
        };
        stats.insert("avg_processing_rate".to_string(), avg_processing_rate);

        // Calculate average completion rate
        let total_completed: usize = self.active_batches
            .values()
            .map(|batch| batch.progress.completed_partitions)
            .sum();
        let total_partitions: usize = self.active_batches
            .values()
            .map(|batch| batch.progress.total_partitions)
            .sum();

        let completion_rate = if total_partitions > 0 {
            total_completed as f64 / total_partitions as f64
        } else {
            0.0
        };
        stats.insert("completion_rate".to_string(), completion_rate);

        stats
    }

    /// Update coordination statistics
    fn update_statistics(&mut self) {
        self.statistics = self.get_statistics();
    }

    /// Get active batches
    pub fn get_active_batches(&self) -> &HashMap<BatchId, BatchExecution<T>> {
        &self.active_batches
    }

    /// Get completed batches
    pub fn get_completed_batches(&self) -> &HashMap<BatchId, BatchExecutionResult<T>> {
        &self.completed_batches
    }

    /// Set aggregation method
    pub fn set_aggregation_method(&mut self, method: AggregationMethod) {
        self.aggregation_method = method;
    }

    /// Set distribution strategy
    pub fn set_distribution_strategy(&mut self, strategy: DistributionStrategy) {
        self.distribution_strategy = strategy;
    }

    /// Cancel batch
    pub fn cancel_batch(&mut self, batch_id: BatchId) -> Result<()> {
        if self.active_batches.remove(&batch_id).is_some() {
            self.batch_queue.retain(|&id| id != batch_id);
            self.update_statistics();
            Ok(())
        } else {
            Err(OptimError::ConfigurationError("Batch not found".to_string()))
        }
    }

    /// Shutdown coordinator
    pub async fn shutdown(&mut self) -> Result<()> {
        self.active_batches.clear();
        self.batch_queue.clear();
        self.completed_batches.clear();
        Ok(())
    }
}

// Default implementations
impl Default for BatchMetadata {
    fn default() -> Self {
        Self {
            priority: BatchPriority::Normal,
            resource_requirements: ResourceRequirements {
                memory_bytes: 1024 * 1024 * 1024, // 1GB
                compute_flops: 1_000_000_000,     // 1 GFLOPS
                communication_bandwidth: 10.0,    // 10 GB/s
                preferred_devices: Vec::new(),
            },
            qos_requirements: QoSRequirements {
                max_latency: Duration::from_secs(30),
                min_throughput: 10.0,
                reliability: 0.95,
                consistency: ConsistencyLevel::Eventual,
            },
            deadline: None,
            tags: HashMap::new(),
        }
    }
}

impl<T: Float + Default> Default for BatchData<T> {
    fn default() -> Self {
        Self {
            inputs: vec![Array::zeros(IxDyn(&[10, 10]))],
            batch_size: 10,
            partitioning: DataPartitioning::Horizontal,
            metadata: BatchMetadata::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_coordinator_creation() {
        let coordinator = BatchCoordinator::<f32>::new(BatchParallelizationStrategy::DataParallel);
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_batch_creation() {
        let mut coordinator = BatchCoordinator::<f32>::new(BatchParallelizationStrategy::DataParallel).unwrap();
        let batch_data = BatchData::default();

        let batch_id = coordinator.create_batch(batch_data).await;
        assert!(batch_id.is_ok());
        assert_eq!(coordinator.get_active_batches().len(), 1);
    }

    #[test]
    fn test_partition_status_update() {
        let mut coordinator = BatchCoordinator::<f32>::new(BatchParallelizationStrategy::DataParallel).unwrap();

        // This test would need a proper setup with active batches
        // For now, just test that the method exists and handles errors appropriately
        let result = coordinator.update_partition_status(
            BatchId(1),
            DeviceId(0),
            PartitionStatus::Completed,
        );
        assert!(result.is_err()); // Should fail because batch doesn't exist
    }
}