//! TPU Pod Coordination for Batch Parallelization
//!
//! This module implements comprehensive coordination mechanisms for TPU pods,
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

// Re-export all functionality from the modular implementation
mod pod_coordination;

pub use pod_coordination::*;