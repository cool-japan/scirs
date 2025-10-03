//! Configuration structures for advanced distributed systems
//!
//! This module provides comprehensive configuration options for:
//! - Advanced cluster configuration
//! - Consensus algorithm settings
//! - Data sharding strategies
//! - Fault tolerance parameters
//! - Auto-scaling policies
//! - Locality optimization
//! - Performance optimization

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Advanced distributed cluster configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvancedClusterConfig {
    /// Basic cluster settings
    pub basic_config: super::super::distributed::DistributedConfig,
    /// Consensus algorithm configuration
    pub consensus_config: ConsensusConfig,
    /// Data sharding strategy
    pub sharding_config: ShardingConfig,
    /// Fault tolerance settings
    pub fault_tolerance: FaultToleranceConfig,
    /// Auto-scaling configuration
    pub auto_scaling: AutoScalingConfig,
    /// Data locality optimization
    pub locality_config: LocalityConfig,
    /// Performance optimization settings
    pub optimization_config: OptimizationConfig,
}

/// Consensus algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Algorithm to use
    pub algorithm: ConsensusAlgorithm,
    /// Minimum number of nodes for quorum
    pub quorum_size: usize,
    /// Election timeout (milliseconds)
    pub election_timeout_ms: u64,
    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,
    /// Maximum entries per append
    pub max_entries_per_append: usize,
    /// Log compaction threshold
    pub log_compaction_threshold: usize,
    /// Snapshot creation interval
    pub snapshot_interval: Duration,
    /// Node identifier (optional)
    pub node_id: Option<String>,
    /// List of peer nodes (optional)
    pub peers: Option<Vec<String>>,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            algorithm: ConsensusAlgorithm::Raft,
            quorum_size: 3,
            election_timeout_ms: 5000,
            heartbeat_interval_ms: 1000,
            max_entries_per_append: 100,
            log_compaction_threshold: 10000,
            snapshot_interval: Duration::from_secs(3600),
            node_id: None,
            peers: None,
        }
    }
}

/// Consensus algorithms supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConsensusAlgorithm {
    /// Raft consensus algorithm
    Raft,
    /// Practical Byzantine Fault Tolerance
    Pbft,
    /// Proof of Stake consensus
    ProofOfStake,
    /// Delegated Proof of Stake
    DelegatedProofOfStake,
    /// Simple majority voting
    SimpleMajority,
    /// No consensus (for testing)
    None,
}

/// Data sharding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Sharding strategy
    pub strategy: ShardingStrategy,
    /// Number of shards
    pub shard_count: usize,
    /// Replication factor per shard
    pub replication_factor: usize,
    /// Hash function for consistent hashing
    pub hash_function: HashFunction,
    /// Virtual nodes per physical node
    pub virtual_nodes: usize,
    /// Enable dynamic resharding
    pub dynamic_resharding: bool,
    /// Shard migration threshold
    pub migration_threshold: f64,
}

impl Default for ShardingConfig {
    fn default() -> Self {
        Self {
            strategy: ShardingStrategy::ConsistentHash,
            shard_count: 16,
            replication_factor: 3,
            hash_function: HashFunction::Murmur3,
            virtual_nodes: 256,
            dynamic_resharding: true,
            migration_threshold: 0.8,
        }
    }
}

/// Data sharding strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShardingStrategy {
    /// Hash-based sharding
    Hash,
    /// Range-based sharding
    Range,
    /// Directory-based sharding
    Directory,
    /// Consistent hashing
    ConsistentHash,
    /// Geographic sharding
    Geographic,
    /// Feature-based sharding
    FeatureBased,
    /// Custom sharding logic
    Custom(String),
}

/// Hash functions for sharding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HashFunction {
    /// MD5 hash
    Md5,
    /// SHA-256 hash
    Sha256,
    /// CRC32 hash
    Crc32,
    /// MurmurHash3
    Murmur3,
    /// xxHash
    XxHash,
    /// Custom hash function
    Custom(String),
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Maximum node failures to tolerate
    pub max_failures: usize,
    /// Enable automatic recovery
    pub auto_recovery: bool,
    /// Recovery timeout (seconds)
    pub recovery_timeout: u64,
    /// Health check interval (seconds)
    pub health_check_interval: u64,
    /// Node replacement strategy
    pub replacement_strategy: NodeReplacementStrategy,
    /// Data backup interval
    pub backup_interval: Duration,
    /// Enable rolling updates
    pub rolling_updates: bool,
    /// Graceful shutdown timeout
    pub graceful_shutdown_timeout: Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            max_failures: 1,
            auto_recovery: true,
            recovery_timeout: 300,
            health_check_interval: 30,
            replacement_strategy: NodeReplacementStrategy::HotStandby,
            backup_interval: Duration::from_secs(3600),
            rolling_updates: true,
            graceful_shutdown_timeout: Duration::from_secs(60),
        }
    }
}

/// Node replacement strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeReplacementStrategy {
    /// Immediate replacement
    Immediate,
    /// Delayed replacement
    Delayed { delay: Duration },
    /// Manual replacement only
    Manual,
    /// Hot standby nodes
    HotStandby,
    /// Cold standby nodes
    ColdStandby,
}

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// CPU threshold for scale-up
    pub scale_up_cpu_threshold: f64,
    /// CPU threshold for scale-down
    pub scale_down_cpu_threshold: f64,
    /// Memory threshold for scale-up
    pub scale_up_memory_threshold: f64,
    /// Memory threshold for scale-down
    pub scale_down_memory_threshold: f64,
    /// Cooldown period between scaling operations
    pub cooldown_period: Duration,
    /// Scaling policies
    pub policies: Vec<ScalingPolicy>,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_nodes: 2,
            max_nodes: 10,
            scale_up_cpu_threshold: 80.0,
            scale_down_cpu_threshold: 30.0,
            scale_up_memory_threshold: 85.0,
            scale_down_memory_threshold: 40.0,
            cooldown_period: Duration::from_secs(300),
            policies: vec![],
        }
    }
}

/// Scaling policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingPolicy {
    /// Policy name
    pub name: String,
    /// Metric to monitor
    pub metric: String,
    /// Threshold value
    pub threshold: f64,
    /// Scaling action
    pub action: ScalingAction,
    /// Adjustment type
    pub adjustment_type: AdjustmentType,
    /// Adjustment value
    pub adjustment_value: f64,
    /// Cooldown period
    pub cooldown: Duration,
}

/// Scaling actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingAction {
    /// Scale up (add nodes)
    ScaleUp,
    /// Scale down (remove nodes)
    ScaleDown,
    /// No action
    None,
}

/// Adjustment types for scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdjustmentType {
    /// Change by a specific count
    ChangeInCapacity,
    /// Change to exact capacity
    ExactCapacity,
    /// Change by percentage
    PercentChangeInCapacity,
}

/// Data locality optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalityConfig {
    /// Enable locality optimization
    pub enabled: bool,
    /// Locality strategy
    pub strategy: LocalityStrategy,
    /// Network topology
    pub topology: NetworkTopology,
    /// Affinity rules
    pub affinity_rules: Vec<AffinityRule>,
    /// Cache configuration
    pub cache_config: CacheConfig,
}

impl Default for LocalityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: LocalityStrategy::NetworkAware,
            topology: NetworkTopology::default(),
            affinity_rules: vec![],
            cache_config: CacheConfig::default(),
        }
    }
}

/// Locality optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LocalityStrategy {
    /// Network-aware placement
    NetworkAware,
    /// Geographic placement
    Geographic,
    /// Latency-based placement
    LatencyBased,
    /// Bandwidth-aware placement
    BandwidthAware,
    /// Cost-optimized placement
    CostOptimized,
    /// Custom placement strategy
    Custom(String),
}

/// Network topology representation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkTopology {
    /// Topology levels (e.g., rack, datacenter, region)
    pub levels: Vec<TopologyLevel>,
    /// Node locations
    pub node_locations: std::collections::HashMap<String, Location>,
}

/// Location in network topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    /// Location coordinates per topology level
    pub coordinates: Vec<String>,
    /// Geographic coordinates (latitude, longitude)
    pub geo_coordinates: Option<(f64, f64)>,
    /// Available bandwidth (Gbps)
    pub bandwidth: f64,
    /// Network latency characteristics
    pub latency_ms: f64,
}

/// Topology level definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyLevel {
    /// Level name (e.g., "rack", "datacenter")
    pub name: String,
    /// Level index (0 = lowest level)
    pub index: usize,
}

/// Affinity rule for data/compute placement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffinityRule {
    /// Rule name
    pub name: String,
    /// Source entity pattern
    pub source_pattern: String,
    /// Target entity pattern
    pub target_pattern: String,
    /// Affinity type
    pub affinity_type: AffinityType,
    /// Rule weight (higher = more important)
    pub weight: f64,
}

/// Affinity rule types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AffinityType {
    /// Entities should be placed together
    Attract,
    /// Entities should be separated
    Repel,
    /// Entities should be in same topology level
    SameLevel(usize),
    /// Custom affinity logic
    Custom(String),
}

/// Cache configuration for locality optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable distributed caching
    pub enabled: bool,
    /// Cache size per node (MB)
    pub cache_size_mb: u64,
    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Cache coherence protocol
    pub coherence_protocol: CoherenceProtocol,
    /// Cache invalidation strategy
    pub invalidation_strategy: InvalidationStrategy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_size_mb: 1024,
            eviction_policy: EvictionPolicy::Lru,
            coherence_protocol: CoherenceProtocol::WriteInvalidate,
            invalidation_strategy: InvalidationStrategy::LazyInvalidation,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// Least Recently Used
    Lru,
    /// Least Frequently Used
    Lfu,
    /// First In, First Out
    Fifo,
    /// Random eviction
    Random,
    /// Time-based eviction
    Ttl,
}

/// Cache coherence protocols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoherenceProtocol {
    /// Write-invalidate protocol
    WriteInvalidate,
    /// Write-update protocol
    WriteUpdate,
    /// Directory-based coherence
    DirectoryBased,
    /// Snooping-based coherence
    SnoopingBased,
}

/// Cache invalidation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InvalidationStrategy {
    /// Immediate invalidation
    ImmediateInvalidation,
    /// Lazy invalidation
    LazyInvalidation,
    /// Lease-based invalidation
    LeaseBased,
    /// Version-based invalidation
    VersionBased,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable performance optimization
    pub enabled: bool,
    /// Batch optimization settings
    pub batch_optimization: BatchOptimization,
    /// Network optimization settings
    pub network_optimization: NetworkOptimization,
    /// Compute optimization settings
    pub compute_optimization: ComputeOptimization,
    /// Memory optimization settings
    pub memory_optimization: MemoryOptimization,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            batch_optimization: BatchOptimization::default(),
            network_optimization: NetworkOptimization::default(),
            compute_optimization: ComputeOptimization::default(),
            memory_optimization: MemoryOptimization::default(),
        }
    }
}

/// Batch processing optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchOptimization {
    /// Enable adaptive batching
    pub adaptive_batching: bool,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Batch timeout (milliseconds)
    pub batch_timeout_ms: u64,
    /// Adaptive algorithm
    pub algorithm: AdaptiveBatchingAlgorithm,
}

impl Default for BatchOptimization {
    fn default() -> Self {
        Self {
            adaptive_batching: true,
            max_batch_size: 1000,
            min_batch_size: 10,
            batch_timeout_ms: 100,
            algorithm: AdaptiveBatchingAlgorithm::LoadBased,
        }
    }
}

/// Adaptive batching algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptiveBatchingAlgorithm {
    /// Load-based batching
    LoadBased,
    /// Latency-based batching
    LatencyBased,
    /// Throughput-based batching
    ThroughputBased,
    /// Machine learning-based batching
    MlBased,
    /// Custom batching algorithm
    Custom(String),
}

/// Network optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkOptimization {
    /// Enable compression
    pub compression: CompressionConfig,
    /// Enable prefetching
    pub prefetching: bool,
    /// Prefetching strategy
    pub prefetching_strategy: PrefetchingStrategy,
    /// Connection pooling
    pub connection_pooling: bool,
    /// Maximum connections per node
    pub max_connections: usize,
}

impl Default for NetworkOptimization {
    fn default() -> Self {
        Self {
            compression: CompressionConfig::default(),
            prefetching: true,
            prefetching_strategy: PrefetchingStrategy::AdaptivePrefetching,
            connection_pooling: true,
            max_connections: 100,
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub level: u8,
    /// Minimum data size for compression (bytes)
    pub min_size_bytes: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Zstd,
            level: 3,
            min_size_bytes: 1024,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// LZ4 compression
    Lz4,
    /// Zstandard compression
    Zstd,
    /// Snappy compression
    Snappy,
    /// Brotli compression
    Brotli,
}

/// Prefetching strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrefetchingStrategy {
    /// No prefetching
    None,
    /// Sequential prefetching
    Sequential,
    /// Pattern-based prefetching
    PatternBased,
    /// Adaptive prefetching
    AdaptivePrefetching,
    /// Machine learning-based prefetching
    MlBased,
}

/// Compute optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeOptimization {
    /// Enable algorithm selection
    pub algorithm_selection: bool,
    /// Selection strategy
    pub selection_strategy: AlgorithmSelectionStrategy,
    /// Precision optimization
    pub precision_optimization: PrecisionOptimization,
    /// Enable vectorization
    pub vectorization: bool,
    /// Enable parallelization
    pub parallelization: bool,
}

impl Default for ComputeOptimization {
    fn default() -> Self {
        Self {
            algorithm_selection: true,
            selection_strategy: AlgorithmSelectionStrategy::PerformanceBased,
            precision_optimization: PrecisionOptimization::default(),
            vectorization: true,
            parallelization: true,
        }
    }
}

/// Algorithm selection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmSelectionStrategy {
    /// Performance-based selection
    PerformanceBased,
    /// Memory-based selection
    MemoryBased,
    /// Energy-based selection
    EnergyBased,
    /// Cost-based selection
    CostBased,
    /// Machine learning-based selection
    MlBased,
}

/// Precision optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionOptimization {
    /// Enable mixed precision
    pub mixed_precision: bool,
    /// Default precision level
    pub default_precision: PrecisionLevel,
    /// Adaptive precision adjustment
    pub adaptive_precision: bool,
}

impl Default for PrecisionOptimization {
    fn default() -> Self {
        Self {
            mixed_precision: true,
            default_precision: PrecisionLevel::Float32,
            adaptive_precision: true,
        }
    }
}

/// Precision levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrecisionLevel {
    /// 16-bit floating point
    Float16,
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
    /// Brain float 16
    BFloat16,
    /// 8-bit integer
    Int8,
    /// 16-bit integer
    Int16,
    /// 32-bit integer
    Int32,
}

/// Memory optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimization {
    /// Enable memory pooling
    pub memory_pooling: bool,
    /// Pool size (MB)
    pub pool_size_mb: u64,
    /// Enable garbage collection optimization
    pub gc_optimization: bool,
    /// Memory allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

impl Default for MemoryOptimization {
    fn default() -> Self {
        Self {
            memory_pooling: true,
            pool_size_mb: 512,
            gc_optimization: true,
            allocation_strategy: AllocationStrategy::PoolBased,
        }
    }
}

/// Memory allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Pool-based allocation
    PoolBased,
    /// Arena-based allocation
    ArenaBased,
    /// Stack-based allocation
    StackBased,
    /// Heap-based allocation
    HeapBased,
}
