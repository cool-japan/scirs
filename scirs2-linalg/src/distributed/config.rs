//! Configuration types for distributed linear algebra operations

use crate::distributed::{CommunicationBackend, DistributionStrategy};
use crate::distributed::mpi::{MPIConfig, MPIImplementation, BufferStrategy, CollectiveHints, MPIErrorHandling, MPIPerformanceTuning};
use serde::{Deserialize, Serialize};

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
    pub blocksize: usize,

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

    /// MPI-specific configuration (when using MPI backend)
    pub mpi_config: Option<MPIConfig>,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            num_nodes: 1,
            node_rank: 0,
            backend: CommunicationBackend::InMemory,
            distribution: DistributionStrategy::RowWise,
            blocksize: 256,
            enable_simd: true,
            threads_per_node: num_cpus::get(),
            comm_timeout_ms: 30000,
            fault_tolerance: false,
            memory_limit_bytes: None,
            compression: CompressionConfig::default(),
            mpi_config: None,
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

    pub fn with_blocksize(mut self, size: usize) -> Self {
        self.blocksize = size;
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

    pub fn with_mpi_config(mut self, mpi_config: MPIConfig) -> Self {
        self.mpi_config = Some(mpi_config);
        self
    }

    /// Create a default MPI configuration
    pub fn with_mpi(mut self, implementation: MPIImplementation) -> Self {
        self.backend = CommunicationBackend::MPI;
        self.mpi_config = Some(MPIConfig {
            implementation,
            non_blocking: true,
            persistent_requests: false,
            enable_mpi_io: false,
            enable_rma: false,
            buffer_strategy: BufferStrategy::Automatic,
            collective_hints: CollectiveHints {
                allreduce_algorithm: None,
                allgather_algorithm: None,
                broadcast_algorithm: None,
                enable_pipelining: true,
                pipeline_chunksize: 64 * 1024,
                enable_hierarchical: true,
            },
            error_handling: MPIErrorHandling::FaultTolerant,
            performance_tuning: MPIPerformanceTuning {
                eager_threshold: 8192,
                rendezvous_threshold: 65536,
                max_segmentsize: 1024 * 1024,
                comm_threads: 1,
                numa_binding: true,
                cpu_affinity: Vec::new(),
                memory_alignment: 64,
            },
        });
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
    pub minsize_bytes: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::LZ4,
            level: 3,
            minsize_bytes: 1024,
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