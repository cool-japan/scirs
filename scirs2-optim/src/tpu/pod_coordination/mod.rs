//! TPU Pod Coordination Module
//!
//! This module provides comprehensive coordination mechanisms for TPU pods,
//! enabling efficient batch parallelization and distributed optimization
//! across multiple TPU devices and nodes.
//!
//! The module has been refactored into a modular architecture for better maintainability:
//!
//! - **coordination**: Core coordination logic, configuration, and strategies
//! - **topology**: Topology management, device layout, and communication topology
//! - **communication**: Communication management, buffers, and optimization
//! - **synchronization**: Synchronization barriers, events, and coordination
//! - **load_balancing**: Load balancing, device monitoring, and migration management
//! - **fault_tolerance**: Failure detection, recovery strategies, and checkpointing
//! - **performance**: Performance analysis, metrics collection, and optimization
//! - **resource_scheduling**: Resource allocation, scheduling, and management
//! - **batch_coordination**: Batch management, data distribution, and pipeline execution
//! - **gradient_aggregation**: Gradient aggregation, compression, and communication optimization
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_optim::tpu::pod_coordination::{
//!     TPUPodCoordinator, PodCoordinationConfig, PodTopology,
//!     BatchParallelizationStrategy, CommunicationPattern
//! };
//!
//! // Create coordination configuration
//! let config = PodCoordinationConfig {
//!     topology: PodTopology::Pod4x4,
//!     num_devices: 16,
//!     batch_strategy: BatchParallelizationStrategy::DataParallel,
//!     communication_pattern: CommunicationPattern::AllReduce,
//!     // ... other configuration
//!     ..Default::default()
//! };
//!
//! // Initialize TPU pod coordinator
//! let coordinator = TPUPodCoordinator::new(config)?;
//!
//! // Coordinate batch execution across the pod
//! let batch_data = BatchData::default();
//! let optimization_step = OptimizationStep::new(|partition| {
//!     // Your optimization logic here
//!     Ok(vec![])
//! });
//!
//! let result = coordinator.coordinate_batch_execution(batch_data, optimization_step).await?;
//! ```
//!
//! ## Architecture Overview
//!
//! The TPU pod coordination system is designed with a hierarchical architecture:
//!
//! 1. **Pod Coordinator**: Central orchestration component
//! 2. **Topology Manager**: Manages device layout and communication topology
//! 3. **Communication Manager**: Handles inter-device communication
//! 4. **Synchronization Manager**: Coordinates timing and barriers
//! 5. **Load Balancer**: Distributes workload efficiently
//! 6. **Fault Tolerance Manager**: Handles failures and recovery
//! 7. **Performance Analyzer**: Monitors and optimizes performance
//! 8. **Resource Scheduler**: Allocates and manages resources
//! 9. **Batch Coordinator**: Manages batch execution and pipelines
//! 10. **Gradient Aggregator**: Aggregates gradients across devices

// Core coordination functionality
pub mod coordination;

// Topology management
pub mod topology;

// Communication management
pub mod communication;

// Synchronization mechanisms
pub mod synchronization;

// Load balancing
pub mod load_balancing;

// Fault tolerance
pub mod fault_tolerance;

// Performance analysis
pub mod performance;

// Resource scheduling
pub mod resource_scheduling;

// Batch coordination
pub mod batch_coordination;

// Gradient aggregation
pub mod gradient_aggregation;

// Re-export core types and functionality
pub use coordination::*;
pub use topology::*;
pub use communication::*;
pub use synchronization::*;
pub use load_balancing::*;
pub use fault_tolerance::*;
pub use performance::*;
pub use resource_scheduling::*;
pub use batch_coordination::*;
pub use gradient_aggregation::*;

// Additional common imports for convenience
use std::collections::HashMap;
use std::time::{Duration, Instant};
use ndarray::{Array, IxDyn};
use num_traits::Float;

use crate::error::{OptimError, Result};

/// Device identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(pub usize);

/// Batch identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchId(pub u64);

/// Pod topology configurations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PodTopology {
    Single,
    Pod2x2,
    Pod4x4,
    Pod8x8,
    Pod16x16,
    Pod32x32,
}

// Re-export key enums and structs that are commonly used together
pub use coordination::{
    TPUPodCoordinator,
    PodCoordinationConfig,
    CoordinationStrategy,
    CommunicationPattern,
    SynchronizationMode,
    BatchParallelizationStrategy,
    GradientAggregationMethod,
    LoadBalancingStrategy,
    MemoryManagementStrategy,
};

pub use topology::{
    TopologyManager,
    DeviceLayout,
    CommunicationTopology,
    CommunicationLink,
    LinkType,
    CommunicationStep,
    CommunicationStepType,
    TopologyProperties,
    RoutingTable,
    BandwidthMatrix,
    LatencyMatrix,
};

pub use communication::{
    CommunicationManager,
    CommunicationId,
    ActiveCommunication,
    CommunicationBuffer,
    BufferStatus,
    CompressionInfo,
    CompressionAlgorithm,
    CommunicationProgress,
};

pub use synchronization::{
    SynchronizationManager,
    BarrierId,
    BarrierState,
    BarrierType,
    SyncEvent,
    SyncEventType,
    SyncEventData,
    DeviceStatus,
};

pub use load_balancing::{
    PodLoadBalancer,
    LoadBalancer,
    DeviceLoad,
    LoadSnapshot,
    RebalancingPolicy,
    RebalancingTrigger,
    RebalancingAction,
    LoadBalancingAlgorithm,
    DeviceAvailability,
};

pub use fault_tolerance::{
    FaultToleranceManager,
    FailureDetector,
    FailureType,
    RecoveryStrategy,
    FailureDetectionAlgorithm,
    FailureInfo,
    FailureStatus,
    RecoveryAction,
    RecoveryPriority,
    CheckpointInfo,
    CheckpointType,
    RedundancyConfig,
    RedundancyStrategy,
    ConsistencyLevel,
    DetectionConfig,
};

pub use performance::{
    PodPerformanceAnalyzer,
    PodPerformanceMetrics,
    DevicePerformanceMetrics,
    PerformanceBenchmark,
    PerformanceAlert,
    AlertType,
    AlertSeverity,
    OptimizationRecommendation,
    RecommendationType,
    EffortLevel,
    RecommendationPriority,
    PerformanceTrend,
    TrendDirection,
};

pub use resource_scheduling::{
    ResourceScheduler,
    ResourceRequirements,
    ResourcePriority,
    ResourceAllocation,
    SchedulingRequest,
    RequestStatus,
    SchedulingStrategy,
    ResourcePoolConfig,
    ReservationPolicy,
    DeviceReservation,
    ReservationType,
    QueueMetrics,
    AllocationMetrics,
};

pub use batch_coordination::{
    BatchCoordinator,
    BatchData,
    BatchMetadata,
    BatchPartition,
    PartitionId,
    BatchExecution,
    BatchProgress,
    PipelineStage,
    PipelineStageStatus,
    DistributionStrategy,
    CachingStrategy,
    AggregationMethod,
    BatchExecutionResult,
    DeviceExecutionStatistics,
    QualityMetrics,
    DataPartitioning,
    BatchPriority,
    PartitionStatus,
};

pub use gradient_aggregation::{
    GradientAggregator,
    GradientBuffer,
    GradientBufferStatus,
    AggregationState,
    AggregationStatistics,
    AggregationConfig,
    CompressionSettings,
    QuantizationSettings,
    QuantizationMethod,
    SparsificationMethod,
    CommunicationOptimization,
    FederatedParams,
    LocalSGDParams,
    SCAFFOLDParams,
    CommunicationStats,
    BufferMetadata,
    CompressionParameters,
};

/// Optimization step interface for batch execution
pub struct OptimizationStep<T: Float> {
    /// Step function
    pub stepfn: std::sync::Arc<dyn Fn(BatchPartition<T>) -> Result<Vec<Array<T, IxDyn>>> + Send + Sync>,
}

impl<T: Float> Clone for OptimizationStep<T> {
    fn clone(&self) -> Self {
        Self {
            stepfn: self.stepfn.clone(),
        }
    }
}

impl<T: Float + Default + Clone + Send + Sync + std::iter::Sum + ndarray::ScalarOperand> OptimizationStep<T> {
    /// Execute the optimization step
    pub async fn execute(&self, partition: BatchPartition<T>) -> Result<Vec<Array<T, IxDyn>>> {
        (self.stepfn)(partition)
    }

    /// Create a new optimization step
    pub fn new<F>(stepfn: F) -> Self
    where
        F: Fn(BatchPartition<T>) -> Result<Vec<Array<T, IxDyn>>> + Send + Sync + 'static,
    {
        Self {
            stepfn: std::sync::Arc::new(stepfn),
        }
    }
}

/// Distributed execution result
#[derive(Debug)]
pub struct DistributedExecutionResult<T: Float> {
    /// Gradients from each device
    pub gradients: HashMap<DeviceId, Vec<Array<T, IxDyn>>>,

    /// Statistics from each device
    pub statistics: HashMap<DeviceId, DeviceExecutionStatistics>,
}

/// Device execution result
#[derive(Debug)]
pub struct DeviceExecutionResult<T: Float> {
    /// Device ID
    pub deviceid: DeviceId,

    /// Computed gradients
    pub gradients: Vec<Array<T, IxDyn>>,

    /// Execution statistics
    pub statistics: DeviceExecutionStatistics,
}

/// Reduce operations for all-reduce
#[derive(Debug, Clone, Copy)]
pub enum ReduceOperation {
    Sum,
    Average,
    Max,
    Min,
    Product,
    LogicalAnd,
    LogicalOr,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

/// Pod performance statistics aggregated from all components
#[derive(Debug, Clone)]
pub struct PodPerformanceStatistics {
    /// Topology statistics
    pub topology_stats: topology::TopologyStatistics,

    /// Communication statistics
    pub communication_stats: communication::CommunicationStatistics,

    /// Synchronization statistics
    pub synchronization_stats: synchronization::SynchronizationStatistics,

    /// Load balance statistics
    pub load_balance_stats: load_balancing::LoadBalanceStatistics,

    /// Fault tolerance statistics
    pub fault_tolerance_stats: fault_tolerance::FaultToleranceStatistics,

    /// Batch coordination statistics
    pub batch_coordination_stats: batch_coordination::BatchCoordinationStatistics,

    /// Gradient aggregation statistics
    pub gradient_aggregation_stats: gradient_aggregation::GradientAggregationStatistics,
}

/// Comprehensive coordination builder for easy setup
pub struct PodCoordinationBuilder {
    config: PodCoordinationConfig,
}

impl PodCoordinationBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            config: PodCoordinationConfig::default(),
        }
    }

    /// Set pod topology
    pub fn topology(mut self, topology: PodTopology) -> Self {
        self.config.topology = topology;
        self
    }

    /// Set number of devices
    pub fn num_devices(mut self, num_devices: usize) -> Self {
        self.config.num_devices = num_devices;
        self
    }

    /// Set coordination strategy
    pub fn coordination_strategy(mut self, strategy: CoordinationStrategy) -> Self {
        self.config.coordination_strategy = strategy;
        self
    }

    /// Set communication pattern
    pub fn communication_pattern(mut self, pattern: CommunicationPattern) -> Self {
        self.config.communication_pattern = pattern;
        self
    }

    /// Set synchronization mode
    pub fn synchronization_mode(mut self, mode: SynchronizationMode) -> Self {
        self.config.synchronization_mode = mode;
        self
    }

    /// Set batch parallelization strategy
    pub fn batch_strategy(mut self, strategy: BatchParallelizationStrategy) -> Self {
        self.config.batch_strategy = strategy;
        self
    }

    /// Set gradient aggregation method
    pub fn gradient_aggregation(mut self, method: GradientAggregationMethod) -> Self {
        self.config.gradient_aggregation = method;
        self
    }

    /// Enable fault tolerance
    pub fn enable_fault_tolerance(mut self, enable: bool) -> Self {
        self.config.enable_fault_tolerance = enable;
        self
    }

    /// Set heartbeat interval
    pub fn heartbeat_interval(mut self, interval_ms: u64) -> Self {
        self.config.heartbeat_interval_ms = interval_ms;
        self
    }

    /// Set operation timeout
    pub fn operation_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.operation_timeout_ms = timeout_ms;
        self
    }

    /// Enable performance monitoring
    pub fn enable_performance_monitoring(mut self, enable: bool) -> Self {
        self.config.enable_performance_monitoring = enable;
        self
    }

    /// Set load balancing strategy
    pub fn load_balancing_strategy(mut self, strategy: LoadBalancingStrategy) -> Self {
        self.config.load_balancing_strategy = strategy;
        self
    }

    /// Set memory management strategy
    pub fn memory_management(mut self, strategy: MemoryManagementStrategy) -> Self {
        self.config.memory_management = strategy;
        self
    }

    /// Enable adaptive optimization
    pub fn adaptive_optimization(mut self, enable: bool) -> Self {
        self.config.adaptive_optimization = enable;
        self
    }

    /// Build the TPU pod coordinator
    pub fn build<T: Float + Default + Clone + Send + Sync + ndarray::ScalarOperand + std::iter::Sum>(
        self,
    ) -> Result<TPUPodCoordinator<T>> {
        TPUPodCoordinator::new(self.config)
    }

    /// Get the configuration
    pub fn get_config(&self) -> &PodCoordinationConfig {
        &self.config
    }
}

impl Default for PodCoordinationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for common operations
pub mod utils {
    use super::*;

    /// Create a simple data parallel configuration
    pub fn create_data_parallel_config(num_devices: usize) -> PodCoordinationConfig {
        PodCoordinationConfig {
            topology: match num_devices {
                1 => PodTopology::Single,
                2..=4 => PodTopology::Pod2x2,
                5..=16 => PodTopology::Pod4x4,
                17..=64 => PodTopology::Pod8x8,
                65..=256 => PodTopology::Pod16x16,
                _ => PodTopology::Pod32x32,
            },
            num_devices,
            coordination_strategy: CoordinationStrategy::Hierarchical,
            communication_pattern: CommunicationPattern::AllReduce,
            synchronization_mode: SynchronizationMode::Synchronous,
            batch_strategy: BatchParallelizationStrategy::DataParallel,
            gradient_aggregation: GradientAggregationMethod::Average,
            enable_fault_tolerance: true,
            heartbeat_interval_ms: 1000,
            operation_timeout_ms: 30000,
            enable_performance_monitoring: true,
            load_balancing_strategy: LoadBalancingStrategy::Dynamic,
            memory_management: MemoryManagementStrategy::DynamicPartitioning,
            adaptive_optimization: true,
        }
    }

    /// Create a pipeline parallel configuration
    pub fn create_pipeline_parallel_config(num_devices: usize) -> PodCoordinationConfig {
        PodCoordinationConfig {
            batch_strategy: BatchParallelizationStrategy::PipelineParallel,
            communication_pattern: CommunicationPattern::Ring,
            synchronization_mode: SynchronizationMode::BulkSynchronous,
            ..create_data_parallel_config(num_devices)
        }
    }

    /// Create a model parallel configuration
    pub fn create_model_parallel_config(num_devices: usize) -> PodCoordinationConfig {
        PodCoordinationConfig {
            batch_strategy: BatchParallelizationStrategy::ModelParallel,
            communication_pattern: CommunicationPattern::AllGather,
            coordination_strategy: CoordinationStrategy::Mesh,
            ..create_data_parallel_config(num_devices)
        }
    }

    /// Create a hybrid parallel configuration
    pub fn create_hybrid_parallel_config(num_devices: usize) -> PodCoordinationConfig {
        PodCoordinationConfig {
            batch_strategy: BatchParallelizationStrategy::Hybrid,
            communication_pattern: CommunicationPattern::AllToAll,
            coordination_strategy: CoordinationStrategy::Adaptive,
            synchronization_mode: SynchronizationMode::Adaptive,
            gradient_aggregation: GradientAggregationMethod::WeightedAverage,
            ..create_data_parallel_config(num_devices)
        }
    }

    /// Validate configuration compatibility
    pub fn validate_config(config: &PodCoordinationConfig) -> Result<()> {
        // Check device count matches topology
        let expected_devices = match config.topology {
            PodTopology::Single => 1,
            PodTopology::Pod2x2 => 4,
            PodTopology::Pod4x4 => 16,
            PodTopology::Pod8x8 => 64,
            PodTopology::Pod16x16 => 256,
            PodTopology::Pod32x32 => 1024,
        };

        if config.num_devices != expected_devices {
            return Err(OptimError::ConfigurationError(format!(
                "Device count {} does not match topology {:?} (expected {})",
                config.num_devices, config.topology, expected_devices
            )));
        }

        // Check strategy compatibility
        match (config.batch_strategy, config.communication_pattern) {
            (BatchParallelizationStrategy::DataParallel, CommunicationPattern::AllReduce) => (),
            (BatchParallelizationStrategy::ModelParallel, CommunicationPattern::AllGather) => (),
            (BatchParallelizationStrategy::PipelineParallel, CommunicationPattern::Ring) => (),
            _ => {
                // Allow other combinations but warn
                println!(
                    "Warning: Batch strategy {:?} with communication pattern {:?} may not be optimal",
                    config.batch_strategy, config.communication_pattern
                );
            }
        }

        // Check timeout values
        if config.operation_timeout_ms < config.heartbeat_interval_ms {
            return Err(OptimError::ConfigurationError(
                "Operation timeout must be greater than heartbeat interval".to_string(),
            ));
        }

        Ok(())
    }

    /// Calculate optimal device count for given workload
    pub fn calculate_optimal_device_count(
        workload_size: usize,
        memory_per_device: usize,
        target_utilization: f64,
    ) -> usize {
        let memory_needed = (workload_size as f64 / target_utilization) as usize;
        let device_count = (memory_needed + memory_per_device - 1) / memory_per_device;

        // Round to nearest valid topology size
        match device_count {
            1 => 1,
            2..=4 => 4,
            5..=16 => 16,
            17..=64 => 64,
            65..=256 => 256,
            _ => 1024,
        }
    }

    /// Estimate performance for configuration
    pub fn estimate_performance(config: &PodCoordinationConfig) -> f64 {
        let base_performance = config.num_devices as f64;

        let strategy_factor = match config.batch_strategy {
            BatchParallelizationStrategy::DataParallel => 0.95,
            BatchParallelizationStrategy::ModelParallel => 0.85,
            BatchParallelizationStrategy::PipelineParallel => 0.90,
            BatchParallelizationStrategy::Hybrid => 0.92,
            BatchParallelizationStrategy::Adaptive => 0.97,
            _ => 0.88,
        };

        let communication_factor = match config.communication_pattern {
            CommunicationPattern::AllReduce => 0.90,
            CommunicationPattern::Ring => 0.85,
            CommunicationPattern::AllToAll => 0.75,
            _ => 0.80,
        };

        let coordination_factor = match config.coordination_strategy {
            CoordinationStrategy::Centralized => 0.85,
            CoordinationStrategy::Decentralized => 0.92,
            CoordinationStrategy::Hierarchical => 0.89,
            CoordinationStrategy::Adaptive => 0.95,
            _ => 0.87,
        };

        base_performance * strategy_factor * communication_factor * coordination_factor
    }
}

/// Presets for common use cases
pub mod presets {
    use super::*;

    /// High-performance data parallel training preset
    pub fn high_performance_data_parallel() -> PodCoordinationBuilder {
        PodCoordinationBuilder::new()
            .coordination_strategy(CoordinationStrategy::Adaptive)
            .communication_pattern(CommunicationPattern::AllReduce)
            .synchronization_mode(SynchronizationMode::Synchronous)
            .batch_strategy(BatchParallelizationStrategy::DataParallel)
            .gradient_aggregation(GradientAggregationMethod::Average)
            .load_balancing_strategy(LoadBalancingStrategy::Dynamic)
            .enable_fault_tolerance(true)
            .enable_performance_monitoring(true)
            .adaptive_optimization(true)
            .heartbeat_interval(500)
            .operation_timeout(15000)
    }

    /// Low-latency inference preset
    pub fn low_latency_inference() -> PodCoordinationBuilder {
        PodCoordinationBuilder::new()
            .coordination_strategy(CoordinationStrategy::Decentralized)
            .communication_pattern(CommunicationPattern::Broadcast)
            .synchronization_mode(SynchronizationMode::Asynchronous)
            .batch_strategy(BatchParallelizationStrategy::ModelParallel)
            .load_balancing_strategy(LoadBalancingStrategy::LatencyAware)
            .enable_fault_tolerance(false)
            .enable_performance_monitoring(true)
            .adaptive_optimization(true)
            .heartbeat_interval(100)
            .operation_timeout(5000)
    }

    /// Fault-tolerant distributed training preset
    pub fn fault_tolerant_training() -> PodCoordinationBuilder {
        PodCoordinationBuilder::new()
            .coordination_strategy(CoordinationStrategy::Hierarchical)
            .communication_pattern(CommunicationPattern::AllReduce)
            .synchronization_mode(SynchronizationMode::BulkSynchronous)
            .batch_strategy(BatchParallelizationStrategy::Hybrid)
            .gradient_aggregation(GradientAggregationMethod::WeightedAverage)
            .load_balancing_strategy(LoadBalancingStrategy::Adaptive)
            .enable_fault_tolerance(true)
            .enable_performance_monitoring(true)
            .adaptive_optimization(true)
            .heartbeat_interval(2000)
            .operation_timeout(60000)
    }

    /// Large-scale pipeline parallel preset
    pub fn large_scale_pipeline() -> PodCoordinationBuilder {
        PodCoordinationBuilder::new()
            .coordination_strategy(CoordinationStrategy::Hierarchical)
            .communication_pattern(CommunicationPattern::Ring)
            .synchronization_mode(SynchronizationMode::BulkSynchronous)
            .batch_strategy(BatchParallelizationStrategy::PipelineParallel)
            .gradient_aggregation(GradientAggregationMethod::LocalSGD)
            .load_balancing_strategy(LoadBalancingStrategy::BandwidthAware)
            .enable_fault_tolerance(true)
            .enable_performance_monitoring(true)
            .adaptive_optimization(true)
            .heartbeat_interval(1000)
            .operation_timeout(45000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_coordination_builder() {
        let config = PodCoordinationBuilder::new()
            .topology(PodTopology::Pod4x4)
            .num_devices(16)
            .batch_strategy(BatchParallelizationStrategy::DataParallel)
            .get_config()
            .clone();

        assert_eq!(config.topology, PodTopology::Pod4x4);
        assert_eq!(config.num_devices, 16);
        assert_eq!(config.batch_strategy, BatchParallelizationStrategy::DataParallel);
    }

    #[test]
    fn test_configuration_validation() {
        let valid_config = utils::create_data_parallel_config(16);
        assert!(utils::validate_config(&valid_config).is_ok());

        let invalid_config = PodCoordinationConfig {
            topology: PodTopology::Pod4x4,
            num_devices: 8, // Mismatch with topology
            ..valid_config
        };
        assert!(utils::validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_optimal_device_calculation() {
        let device_count = utils::calculate_optimal_device_count(
            1024 * 1024 * 1024, // 1GB workload
            256 * 1024 * 1024,  // 256MB per device
            0.8,                // 80% utilization
        );

        assert!(device_count > 0);
        assert!(device_count <= 1024);
    }

    #[test]
    fn test_performance_estimation() {
        let config = utils::create_data_parallel_config(16);
        let performance = utils::estimate_performance(&config);

        assert!(performance > 0.0);
        assert!(performance <= 16.0); // Should not exceed number of devices
    }

    #[test]
    fn test_presets() {
        let hp_config = presets::high_performance_data_parallel().get_config().clone();
        assert_eq!(hp_config.coordination_strategy, CoordinationStrategy::Adaptive);
        assert_eq!(hp_config.batch_strategy, BatchParallelizationStrategy::DataParallel);

        let lt_config = presets::low_latency_inference().get_config().clone();
        assert_eq!(lt_config.coordination_strategy, CoordinationStrategy::Decentralized);
        assert_eq!(lt_config.synchronization_mode, SynchronizationMode::Asynchronous);
    }

    #[tokio::test]
    async fn test_optimization_step() {
        let step = OptimizationStep::<f32>::new(|_partition| {
            Ok(vec![Array::ones(IxDyn(&[2, 2]))])
        });

        let partition = BatchPartition {
            data: Array::zeros(IxDyn(&[2, 2])),
            indices: vec![0, 1],
            status: PartitionStatus::Assigned,
            device: DeviceId(0),
            dependencies: Vec::new(),
            created_at: Instant::now(),
            processing_start: None,
            completed_at: None,
        };

        let result = step.execute(partition).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }
}