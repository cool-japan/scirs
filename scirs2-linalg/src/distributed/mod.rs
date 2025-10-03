//! Distributed linear algebra operations
//!
//! This module provides distributed implementations of linear algebra operations
//! that can scale across multiple nodes or computing devices. It integrates with
//! the SIMD vectorization framework and provides efficient communication primitives
//! for distributed computing workloads.
//!
//! # Features
//!
//! - **Distributed matrix operations**: Matrix multiplication, decompositions, and solvers
//! - **Load balancing**: Automatic work distribution and load balancing across nodes
//! - **Communication optimization**: Efficient data transfer with minimal overhead
//! - **SIMD integration**: Leverages SIMD operations for maximum performance per node
//! - **Fault tolerance**: Graceful handling of node failures and recovery
//! - **Memory efficiency**: Optimized memory usage for large-scale computations
//!
//! # Architecture
//!
//! The distributed computing framework consists of several layers:
//!
//! 1. **Communication Layer**: Handles data transfer between nodes
//! 2. **Distribution Layer**: Manages data partitioning and work distribution
//! 3. **Computation Layer**: Executes local computations using SIMD acceleration
//! 4. **Coordination Layer**: Synchronizes operations across nodes
//!
//! # Example
//!
//! ```rust
//! use scirs2_linalg::distributed::{DistributedConfig, DistributedMatrix};
//! use scirs2_core::ndarray::Array2;
//!
//! // Create a distributed matrix
//! let matrix = Array2::from_shape_fn((1000, 1000), |(i, j)| (i + j) as f64);
//! let config = DistributedConfig::default().with_num_nodes(4);
//! let distmatrix = DistributedMatrix::from_local(matrix, config)?;
//!
//! // Perform distributed matrix multiplication
//! let result = distmatrix.distributed_matmul(&distmatrix)?;
//!
//! // Gather results back to local matrix
//! let local_result = result.gather()?;
//! ```

// Core modules
pub mod communication;
pub mod distribution;
pub mod computation;
pub mod coordination;
pub mod matrix;
pub mod solvers;
pub mod decomposition;
pub mod mpi;

// New modular components
pub mod config;
pub mod stats;
pub mod load_balancer;
pub mod fault_tolerance;
pub mod redundancy;
pub mod monitoring;
pub mod capacity_planning;
pub mod topology;
pub mod context;
pub mod ops;
pub mod framework;

// Re-export main types for convenience
pub use communication::{CommunicationBackend, DistributedCommunicator, MessageTag};
pub use coordination::{DistributedCoordinator, SynchronizationBarrier};
pub use distribution::{DataDistribution, DistributionStrategy, LoadBalancer};
pub use matrix::{DistributedMatrix, DistributedVector};
pub use mpi::{
    MPIBackend, MPIConfig, MPIImplementation, BufferStrategy, CollectiveHints,
    MPIErrorHandling, MPIPerformanceTuning, MPICommunicator
};

// Re-export new modular components
pub use config::{DistributedConfig, CompressionConfig, CompressionAlgorithm};
pub use stats::DistributedStats;
pub use load_balancer::{AdaptiveLoadBalancer, LoadPredictionModel, WorkloadType};
pub use fault_tolerance::{FaultToleranceManager, NodeHealthMonitor, CheckpointManager, RecoveryStrategy};
pub use redundancy::{RedundancyManager, RedundancyPolicy};
pub use monitoring::{ResourceMonitor, MetricsCollector, AlertSystem};
pub use capacity_planning::{CapacityPlanner, DemandForecastModel};
pub use topology::{NetworkTopologyAnalyzer, TopologySnapshot};
pub use context::{DistributedContext, initialize_distributed, finalize_distributed};
pub use ops::DistributedLinalgOps;
pub use framework::AdvancedDistributedFramework;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig::default()
            .with_num_nodes(4)
            .with_node_rank(0)
            .with_blocksize(512)
            .with_simd(true);

        assert_eq!(config.num_nodes, 4);
        assert_eq!(config.node_rank, 0);
        assert_eq!(config.blocksize, 512);
        assert!(config.enable_simd);
    }

    #[test]
    fn test_compression_config() {
        let compression_config = CompressionConfig::default();
        assert_eq!(compression_config.algorithm, CompressionAlgorithm::LZ4);
    }

    #[test]
    fn test_distributed_stats() {
        let mut stats = DistributedStats::new();

        stats.record_communication(1024, 10);
        stats.record_computation(50);

        assert_eq!(stats.bytes_transferred, 1024);
        assert_eq!(stats.comm_time_ms, 10);
        assert_eq!(stats.compute_time_ms, 50);
        assert_eq!(stats.comm_events, 1);
        assert_eq!(stats.operations_count, 1);

        assert_eq!(stats.comm_compute_ratio(), 0.2);
        assert_eq!(stats.bandwidth_utilization(), 102.4);
    }
}