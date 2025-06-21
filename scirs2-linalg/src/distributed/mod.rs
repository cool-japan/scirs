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
//! use ndarray::Array2;
//!
//! // Create a distributed matrix
//! let matrix = Array2::from_shape_fn((1000, 1000), |(i, j)| (i + j) as f64);
//! let config = DistributedConfig::default().with_num_nodes(4);
//! let dist_matrix = DistributedMatrix::from_local(matrix, config)?;
//!
//! // Perform distributed matrix multiplication
//! let result = dist_matrix.distributed_matmul(&dist_matrix)?;
//!
//! // Gather results back to local matrix
//! let local_result = result.gather()?;
//! ```

pub mod communication;
pub mod distribution;
pub mod computation;
pub mod coordination;
pub mod matrix;
pub mod solvers;
pub mod decomposition;

// Re-export main types for convenience
pub use communication::{CommunicationBackend, DistributedCommunicator, MessageTag};
pub use coordination::{DistributedCoordinator, SynchronizationBarrier};
pub use distribution::{DataDistribution, DistributionStrategy, LoadBalancer};
pub use matrix::{DistributedMatrix, DistributedVector};

use crate::error::{LinalgError, LinalgResult};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for distributed linear algebra operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Number of compute nodes
    pub num_nodes: usize,
    
    /// Rank of the current node (0-indexed)
    pub node_rank: usize,
    
    /// Communication backend to use
    pub backend: CommunicationBackend,
    
    /// Data distribution strategy
    pub distribution: DistributionStrategy,
    
    /// Block size for tiled operations
    pub block_size: usize,
    
    /// Enable SIMD acceleration for local computations
    pub enable_simd: bool,
    
    /// Number of threads per node
    pub threads_per_node: usize,
    
    /// Communication timeout in milliseconds
    pub comm_timeout_ms: u64,
    
    /// Enable fault tolerance
    pub fault_tolerance: bool,
    
    /// Memory limit per node in bytes
    pub memory_limit_bytes: Option<usize>,
    
    /// Compression settings for data transfer
    pub compression: CompressionConfig,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_nodes: 1,
            node_rank: 0,
            backend: CommunicationBackend::InMemory,
            distribution: DistributionStrategy::RowWise,
            block_size: 256,
            enable_simd: true,
            threads_per_node: num_cpus::get(),
            comm_timeout_ms: 30000,
            fault_tolerance: false,
            memory_limit_bytes: None,
            compression: CompressionConfig::default(),
        }
    }
}

impl DistributedConfig {
    /// Builder methods
    pub fn with_num_nodes(mut self, num_nodes: usize) -> Self {
        self.num_nodes = num_nodes;
        self
    }
    
    pub fn with_node_rank(mut self, rank: usize) -> Self {
        self.node_rank = rank;
        self
    }
    
    pub fn with_backend(mut self, backend: CommunicationBackend) -> Self {
        self.backend = backend;
        self
    }
    
    pub fn with_distribution(mut self, strategy: DistributionStrategy) -> Self {
        self.distribution = strategy;
        self
    }
    
    pub fn with_block_size(mut self, size: usize) -> Self {
        self.block_size = size;
        self
    }
    
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }
    
    pub fn with_threads_per_node(mut self, threads: usize) -> Self {
        self.threads_per_node = threads;
        self
    }
    
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.comm_timeout_ms = timeout_ms;
        self
    }
    
    pub fn with_fault_tolerance(mut self, enable: bool) -> Self {
        self.fault_tolerance = enable;
        self
    }
    
    pub fn with_memory_limit(mut self, limit_bytes: usize) -> Self {
        self.memory_limit_bytes = Some(limit_bytes);
        self
    }
    
    pub fn with_compression(mut self, compression: CompressionConfig) -> Self {
        self.compression = compression;
        self
    }
}

/// Configuration for data compression during communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    
    /// Compression level (1-9, where 9 is highest compression)
    pub level: u8,
    
    /// Minimum data size to compress (bytes)
    pub min_size_bytes: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::LZ4,
            level: 3,
            min_size_bytes: 1024,
        }
    }
}

/// Supported compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// LZ4 compression (fast)
    LZ4,
    /// Zstd compression (balanced)
    Zstd,
    /// Gzip compression (small size)
    Gzip,
}

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

/// High-level distributed linear algebra operations
pub struct DistributedLinalgOps;

impl DistributedLinalgOps {
    /// Distributed matrix multiplication: C = A * B
    pub fn distributed_matmul<T>(
        a: &DistributedMatrix<T>,
        b: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: num_traits::Float + Send + Sync + 'static,
    {
        // Check matrix dimensions
        let (m, k) = a.global_shape();
        let (k2, n) = b.global_shape();
        
        if k != k2 {
            return Err(LinalgError::DimensionError(format!(
                "Matrix dimensions don't match for multiplication: ({}, {}) x ({}, {})",
                m, k, k2, n
            )));
        }
        
        // Execute distributed matrix multiplication
        a.multiply(b)
    }
    
    /// Distributed matrix addition: C = A + B
    pub fn distributed_add<T>(
        a: &DistributedMatrix<T>,
        b: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: num_traits::Float + Send + Sync + 'static,
    {
        // Check matrix dimensions
        if a.global_shape() != b.global_shape() {
            return Err(LinalgError::DimensionError(format!(
                "Matrix dimensions don't match for addition: {:?} vs {:?}",
                a.global_shape(),
                b.global_shape()
            )));
        }
        
        // Execute distributed matrix addition
        a.add(b)
    }
    
    /// Distributed matrix transpose: B = A^T
    pub fn distributed_transpose<T>(
        matrix: &DistributedMatrix<T>,
    ) -> LinalgResult<DistributedMatrix<T>>
    where
        T: num_traits::Float + Send + Sync + 'static,
    {
        matrix.transpose()
    }
    
    /// Distributed solve linear system: Ax = b
    pub fn distributed_solve<T>(
        a: &DistributedMatrix<T>,
        b: &DistributedVector<T>,
    ) -> LinalgResult<DistributedVector<T>>
    where
        T: num_traits::Float + Send + Sync + 'static,
    {
        solvers::solve_linear_system(a, b)
    }
    
    /// Distributed LU decomposition
    pub fn distributed_lu<T>(
        matrix: &DistributedMatrix<T>,
    ) -> LinalgResult<(DistributedMatrix<T>, DistributedMatrix<T>)>
    where
        T: num_traits::Float + Send + Sync + 'static,
    {
        decomposition::lu_decomposition(matrix)
    }
    
    /// Distributed QR decomposition
    pub fn distributed_qr<T>(
        matrix: &DistributedMatrix<T>,
    ) -> LinalgResult<(DistributedMatrix<T>, DistributedMatrix<T>)>
    where
        T: num_traits::Float + Send + Sync + 'static,
    {
        decomposition::qr_decomposition(matrix)
    }
}

/// Initialize distributed computing environment
pub fn initialize_distributed(config: DistributedConfig) -> LinalgResult<DistributedContext> {
    DistributedContext::new(config)
}

/// Shutdown distributed computing environment
pub fn finalize_distributed(context: DistributedContext) -> LinalgResult<DistributedStats> {
    context.finalize()
}

/// Context for distributed linear algebra operations
pub struct DistributedContext {
    /// Configuration
    pub config: DistributedConfig,
    
    /// Communicator
    pub communicator: DistributedCommunicator,
    
    /// Coordinator
    pub coordinator: DistributedCoordinator,
    
    /// Load balancer
    pub load_balancer: LoadBalancer,
    
    /// Statistics tracker
    pub stats: DistributedStats,
}

impl DistributedContext {
    /// Create new distributed context
    pub fn new(config: DistributedConfig) -> LinalgResult<Self> {
        let communicator = DistributedCommunicator::new(&config)?;
        let coordinator = DistributedCoordinator::new(&config)?;
        let load_balancer = LoadBalancer::new(&config)?;
        let stats = DistributedStats::new();
        
        Ok(Self {
            config,
            communicator,
            coordinator,
            load_balancer,
            stats,
        })
    }
    
    /// Finalize and return statistics
    pub fn finalize(mut self) -> LinalgResult<DistributedStats> {
        // Synchronize all nodes before shutdown
        self.coordinator.barrier()?;
        
        // Finalize communication
        self.communicator.finalize()?;
        
        Ok(self.stats)
    }
    
    /// Get current statistics
    pub fn get_stats(&self) -> &DistributedStats {
        &self.stats
    }
    
    /// Update statistics
    pub fn update_stats(&mut self, update: impl FnOnce(&mut DistributedStats)) {
        update(&mut self.stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    
    #[test]
    fn test_distributed_config() {
        let config = DistributedConfig::default()
            .with_num_nodes(4)
            .with_node_rank(0)
            .with_block_size(512)
            .with_simd(true);
            
        assert_eq!(config.num_nodes, 4);
        assert_eq!(config.node_rank, 0);
        assert_eq!(config.block_size, 512);
        assert!(config.enable_simd);
    }
    
    #[test] 
    fn test_compression_config() {
        let compression = CompressionConfig::default()
            .enabled
            .algorithm;
            
        assert_eq!(compression, CompressionAlgorithm::LZ4);
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