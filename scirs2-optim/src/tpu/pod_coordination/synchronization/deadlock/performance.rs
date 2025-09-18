//! Performance Monitoring and Statistics for Deadlock Detection
//!
//! This module provides comprehensive performance monitoring, metrics collection,
//! and statistical analysis for deadlock detection systems.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive deadlock detection statistics
#[derive(Debug, Clone)]
pub struct DeadlockStatistics {
    /// Total detection cycles
    pub total_cycles: usize,
    /// Deadlocks detected
    pub deadlocks_detected: usize,
    /// False positives
    pub false_positives: usize,
    /// False negatives
    pub false_negatives: usize,
    /// Detection time statistics
    pub detection_time: DetectionTimeStatistics,
    /// Prevention effectiveness
    pub prevention_effectiveness: f64,
    /// Recovery effectiveness
    pub recovery_effectiveness: f64,
    /// System impact
    pub system_impact: SystemImpactStatistics,
}

/// Detection time statistics and analysis
#[derive(Debug, Clone)]
pub struct DetectionTimeStatistics {
    /// Average detection time
    pub average: Duration,
    /// Minimum detection time
    pub minimum: Duration,
    /// Maximum detection time
    pub maximum: Duration,
    /// Detection time variance
    pub variance: f64,
    /// Detection time percentiles
    pub percentiles: DetectionTimePercentiles,
}

/// Detection time percentiles for detailed analysis
#[derive(Debug, Clone)]
pub struct DetectionTimePercentiles {
    /// 50th percentile (median)
    pub p50: Duration,
    /// 90th percentile
    pub p90: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile
    pub p999: Duration,
}

/// System impact metrics
#[derive(Debug, Clone)]
pub struct SystemImpactStatistics {
    /// Performance impact
    pub performance_impact: f64,
    /// Resource overhead
    pub resource_overhead: f64,
    /// Availability impact
    pub availability_impact: f64,
    /// User impact
    pub user_impact: f64,
}

/// Performance configuration for deadlock detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockPerformanceConfig {
    /// Performance monitoring
    pub monitoring: PerformanceMonitoring,
    /// Optimization settings
    pub optimization: PerformanceOptimization,
    /// Resource management
    pub resource_management: ResourceManagement,
    /// Scalability settings
    pub scalability: ScalabilityConfiguration,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoring {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring metrics
    pub metrics: Vec<PerformanceMetric>,
    /// Monitoring frequency
    pub frequency: Duration,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
}

/// Performance metrics to collect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Detection latency
    DetectionLatency,
    /// CPU utilization
    CpuUtilization,
    /// Memory utilization
    MemoryUtilization,
    /// Throughput
    Throughput,
    /// Error rate
    ErrorRate,
    /// Resource contention
    ResourceContention,
    /// Network latency
    NetworkLatency,
    /// Disk I/O
    DiskIo,
    /// Custom metric
    Custom { name: String },
}

/// Performance threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum detection latency
    pub max_detection_latency: Duration,
    /// Maximum CPU utilization
    pub max_cpu_utilization: f64,
    /// Maximum memory utilization
    pub max_memory_utilization: f64,
    /// Minimum throughput
    pub min_throughput: f64,
    /// Maximum error rate
    pub max_error_rate: f64,
}

/// Performance optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimization {
    /// Optimization techniques
    pub techniques: Vec<OptimizationTechnique>,
    /// Auto-tuning
    pub auto_tuning: AutoTuning,
    /// Caching strategies
    pub caching: CachingStrategies,
}

/// Available optimization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTechnique {
    /// Lazy evaluation
    LazyEvaluation,
    /// Memoization
    Memoization,
    /// Parallel processing
    ParallelProcessing,
    /// Cache optimization
    CacheOptimization,
    /// Compression
    Compression,
    /// Batch processing
    BatchProcessing,
}

/// Auto-tuning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoTuning {
    /// Enable auto-tuning
    pub enabled: bool,
    /// Tuning interval
    pub interval: Duration,
    /// Performance targets
    pub targets: PerformanceTargets,
    /// Adaptation rate
    pub adaptation_rate: f64,
}

/// Performance targets for auto-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target latency
    pub target_latency: Duration,
    /// Target CPU usage
    pub target_cpu: f64,
    /// Target memory usage
    pub target_memory: f64,
    /// Target throughput
    pub target_throughput: f64,
}

/// Caching strategies configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingStrategies {
    /// Result caching
    pub result_caching: ResultCaching,
    /// Graph caching
    pub graph_caching: GraphCaching,
    /// Computation caching
    pub computation_caching: ComputationCaching,
}

/// Result caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultCaching {
    /// Cache size
    pub cache_size: usize,
    /// TTL (time to live)
    pub ttl: Duration,
    /// Cache policy
    pub policy: CachePolicy,
}

/// Graph caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphCaching {
    /// Cache size
    pub cache_size: usize,
    /// Cache policy
    pub policy: CachePolicy,
    /// Update strategy
    pub update_strategy: UpdateStrategy,
}

/// Cache policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachePolicy {
    /// Least recently used
    LRU,
    /// Least frequently used
    LFU,
    /// Most recently used
    MRU,
    /// Random replacement
    Random,
    /// Optimal (Belady's algorithm)
    Optimal,
}

/// Cache update strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpdateStrategy {
    /// Write-through
    WriteThrough,
    /// Write-back
    WriteBack,
    /// Write-around
    WriteAround,
}

/// Computation caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationCaching {
    /// Enable caching
    pub enabled: bool,
    /// Cache capacity
    pub capacity: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionPolicy {
    /// FIFO (First In, First Out)
    FIFO,
    /// LIFO (Last In, First Out)
    LIFO,
    /// LRU (Least Recently Used)
    LRU,
    /// LFU (Least Frequently Used)
    LFU,
    /// Random
    Random,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagement {
    /// Memory management
    pub memory: MemoryManagement,
    /// CPU management
    pub cpu: CpuManagement,
    /// I/O management
    pub io: IoManagement,
}

/// Memory management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagement {
    /// Memory limits
    pub limits: MemoryLimits,
    /// Garbage collection
    pub gc: GarbageCollection,
    /// Memory monitoring
    pub monitoring: MemoryMonitoring,
}

/// Memory limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum heap size
    pub max_heap: usize,
    /// Maximum stack size
    pub max_stack: usize,
    /// Memory warning threshold
    pub warning_threshold: f64,
    /// Memory critical threshold
    pub critical_threshold: f64,
}

/// Garbage collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollection {
    /// Enable aggressive GC
    pub aggressive: bool,
    /// GC interval
    pub interval: Duration,
    /// GC strategy
    pub strategy: GcStrategy,
}

/// Garbage collection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GcStrategy {
    /// Mark and sweep
    MarkAndSweep,
    /// Reference counting
    ReferenceCounting,
    /// Generational
    Generational,
    /// Incremental
    Incremental,
}

/// Memory monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMonitoring {
    /// Monitor allocation rate
    pub allocation_rate: bool,
    /// Monitor fragmentation
    pub fragmentation: bool,
    /// Monitor leaks
    pub leaks: bool,
}

/// CPU management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuManagement {
    /// CPU limits
    pub limits: CpuLimits,
    /// Thread management
    pub threads: ThreadManagement,
    /// CPU monitoring
    pub monitoring: CpuMonitoring,
}

/// CPU limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum CPU usage percentage
    pub max_usage: f64,
    /// CPU time quota
    pub time_quota: Duration,
    /// Priority level
    pub priority: i32,
}

/// Thread management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadManagement {
    /// Maximum thread count
    pub max_threads: usize,
    /// Thread pool size
    pub pool_size: usize,
    /// Thread spawning strategy
    pub spawning_strategy: SpawningStrategy,
}

/// Thread spawning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpawningStrategy {
    /// On-demand spawning
    OnDemand,
    /// Pre-allocated pool
    PreAllocated,
    /// Dynamic sizing
    Dynamic,
}

/// CPU monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuMonitoring {
    /// Monitor utilization
    pub utilization: bool,
    /// Monitor load average
    pub load_average: bool,
    /// Monitor context switches
    pub context_switches: bool,
}

/// I/O management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoManagement {
    /// I/O limits
    pub limits: IoLimits,
    /// I/O optimization
    pub optimization: IoOptimization,
    /// I/O monitoring
    pub monitoring: IoMonitoring,
}

/// I/O limits configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoLimits {
    /// Maximum I/O operations per second
    pub max_iops: usize,
    /// Maximum bandwidth
    pub max_bandwidth: usize,
    /// I/O timeout
    pub timeout: Duration,
}

/// I/O optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoOptimization {
    /// Enable buffering
    pub buffering: bool,
    /// Buffer size
    pub buffer_size: usize,
    /// I/O scheduling strategy
    pub scheduling: IoScheduling,
}

/// I/O scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IoScheduling {
    /// First-come, first-served
    FCFS,
    /// Shortest seek time first
    SSTF,
    /// SCAN
    SCAN,
    /// C-SCAN
    CSCAN,
}

/// I/O monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoMonitoring {
    /// Monitor I/O latency
    pub latency: bool,
    /// Monitor I/O throughput
    pub throughput: bool,
    /// Monitor I/O errors
    pub errors: bool,
}

/// Scalability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityConfiguration {
    /// Horizontal scaling
    pub horizontal: HorizontalScaling,
    /// Vertical scaling
    pub vertical: VerticalScaling,
    /// Load balancing
    pub load_balancing: LoadBalancing,
}

/// Horizontal scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalScaling {
    /// Enable auto-scaling
    pub auto_scaling: bool,
    /// Minimum instances
    pub min_instances: usize,
    /// Maximum instances
    pub max_instances: usize,
    /// Scaling triggers
    pub triggers: Vec<ScalingTrigger>,
}

/// Scaling triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingTrigger {
    /// CPU utilization threshold
    CpuUtilization { threshold: f64 },
    /// Memory utilization threshold
    MemoryUtilization { threshold: f64 },
    /// Request queue length
    QueueLength { threshold: usize },
    /// Response time threshold
    ResponseTime { threshold: Duration },
}

/// Vertical scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalScaling {
    /// Enable auto-scaling
    pub auto_scaling: bool,
    /// Resource limits
    pub limits: VerticalLimits,
    /// Performance monitoring
    pub monitoring: VerticalPerformanceMonitoring,
}

/// Vertical scaling limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalLimits {
    /// Maximum CPU cores
    pub max_cpu_cores: usize,
    /// Maximum memory
    pub max_memory: usize,
    /// Resource allocation strategy
    pub allocation_strategy: AllocationStrategy,
}

/// Resource allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Conservative allocation
    Conservative,
    /// Aggressive allocation
    Aggressive,
    /// Adaptive allocation
    Adaptive,
}

/// Vertical performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalPerformanceMonitoring {
    /// Monitoring metrics
    pub metrics: Vec<VerticalMetric>,
    /// Scaling triggers
    pub triggers: Vec<ScalingTrigger>,
    /// Cooldown period
    pub cooldown: Duration,
}

/// Vertical scaling metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerticalMetric {
    /// Resource utilization
    ResourceUtilization,
    /// Performance degradation
    PerformanceDegradation,
    /// Bottleneck detection
    BottleneckDetection,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancing {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Health checking
    pub health_checking: HealthChecking,
    /// Monitoring
    pub monitoring: LoadBalancingMonitoring,
}

/// Load balancing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted round robin
    WeightedRoundRobin { weights: Vec<f64> },
    /// IP hash
    IpHash,
    /// Least response time
    LeastResponseTime,
}

/// Health checking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthChecking {
    /// Health check interval
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Health check retries
    pub retries: usize,
    /// Health check endpoint
    pub endpoint: String,
}

/// Load balancing monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingMonitoring {
    /// Monitoring metrics
    pub metrics: Vec<LoadBalancingMetric>,
    /// Monitoring frequency
    pub frequency: Duration,
    /// Alert thresholds
    pub thresholds: LoadBalancingThresholds,
}

/// Load balancing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingMetric {
    /// Load distribution
    LoadDistribution,
    /// Response time variance
    ResponseTimeVariance,
    /// Connection count
    ConnectionCount,
    /// Error rate
    ErrorRate,
}

/// Load balancing alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingThresholds {
    /// Maximum load imbalance
    pub max_load_imbalance: f64,
    /// Maximum response time variance
    pub max_response_variance: Duration,
    /// Maximum error rate
    pub max_error_rate: f64,
}

impl Default for DeadlockStatistics {
    fn default() -> Self {
        Self {
            total_cycles: 0,
            deadlocks_detected: 0,
            false_positives: 0,
            false_negatives: 0,
            detection_time: DetectionTimeStatistics::default(),
            prevention_effectiveness: 0.0,
            recovery_effectiveness: 0.0,
            system_impact: SystemImpactStatistics::default(),
        }
    }
}

impl Default for DetectionTimeStatistics {
    fn default() -> Self {
        Self {
            average: Duration::from_millis(0),
            minimum: Duration::from_millis(0),
            maximum: Duration::from_millis(0),
            variance: 0.0,
            percentiles: DetectionTimePercentiles::default(),
        }
    }
}

impl Default for DetectionTimePercentiles {
    fn default() -> Self {
        Self {
            p50: Duration::from_millis(0),
            p90: Duration::from_millis(0),
            p95: Duration::from_millis(0),
            p99: Duration::from_millis(0),
            p999: Duration::from_millis(0),
        }
    }
}

impl Default for SystemImpactStatistics {
    fn default() -> Self {
        Self {
            performance_impact: 0.0,
            resource_overhead: 0.0,
            availability_impact: 0.0,
            user_impact: 0.0,
        }
    }
}

impl Default for DeadlockPerformanceConfig {
    fn default() -> Self {
        Self {
            monitoring: PerformanceMonitoring::default(),
            optimization: PerformanceOptimization::default(),
            resource_management: ResourceManagement::default(),
            scalability: ScalabilityConfiguration::default(),
        }
    }
}

impl Default for PerformanceMonitoring {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics: vec![
                PerformanceMetric::DetectionLatency,
                PerformanceMetric::CpuUtilization,
                PerformanceMetric::MemoryUtilization,
            ],
            frequency: Duration::from_secs(10),
            thresholds: PerformanceThresholds::default(),
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_detection_latency: Duration::from_millis(100),
            max_cpu_utilization: 80.0,
            max_memory_utilization: 90.0,
            min_throughput: 100.0,
            max_error_rate: 1.0,
        }
    }
}

impl Default for PerformanceOptimization {
    fn default() -> Self {
        Self {
            techniques: vec![
                OptimizationTechnique::CacheOptimization,
                OptimizationTechnique::ParallelProcessing,
            ],
            auto_tuning: AutoTuning::default(),
            caching: CachingStrategies::default(),
        }
    }
}

impl Default for AutoTuning {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(60),
            targets: PerformanceTargets::default(),
            adaptation_rate: 0.1,
        }
    }
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            target_latency: Duration::from_millis(50),
            target_cpu: 70.0,
            target_memory: 80.0,
            target_throughput: 1000.0,
        }
    }
}

impl Default for CachingStrategies {
    fn default() -> Self {
        Self {
            result_caching: ResultCaching::default(),
            graph_caching: GraphCaching::default(),
            computation_caching: ComputationCaching::default(),
        }
    }
}

impl Default for ResultCaching {
    fn default() -> Self {
        Self {
            cache_size: 1000,
            ttl: Duration::from_secs(300),
            policy: CachePolicy::LRU,
        }
    }
}

impl Default for GraphCaching {
    fn default() -> Self {
        Self {
            cache_size: 500,
            policy: CachePolicy::LRU,
            update_strategy: UpdateStrategy::WriteThrough,
        }
    }
}

impl Default for ComputationCaching {
    fn default() -> Self {
        Self {
            enabled: true,
            capacity: 2000,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

impl Default for ResourceManagement {
    fn default() -> Self {
        Self {
            memory: MemoryManagement::default(),
            cpu: CpuManagement::default(),
            io: IoManagement::default(),
        }
    }
}

impl Default for MemoryManagement {
    fn default() -> Self {
        Self {
            limits: MemoryLimits::default(),
            gc: GarbageCollection::default(),
            monitoring: MemoryMonitoring::default(),
        }
    }
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_heap: 1024 * 1024 * 1024, // 1GB
            max_stack: 8 * 1024 * 1024,   // 8MB
            warning_threshold: 0.8,
            critical_threshold: 0.95,
        }
    }
}

impl Default for GarbageCollection {
    fn default() -> Self {
        Self {
            aggressive: false,
            interval: Duration::from_secs(30),
            strategy: GcStrategy::Incremental,
        }
    }
}

impl Default for MemoryMonitoring {
    fn default() -> Self {
        Self {
            allocation_rate: true,
            fragmentation: true,
            leaks: true,
        }
    }
}

impl Default for CpuManagement {
    fn default() -> Self {
        Self {
            limits: CpuLimits::default(),
            threads: ThreadManagement::default(),
            monitoring: CpuMonitoring::default(),
        }
    }
}

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_usage: 80.0,
            time_quota: Duration::from_secs(1),
            priority: 0,
        }
    }
}

impl Default for ThreadManagement {
    fn default() -> Self {
        Self {
            max_threads: num_cpus::get(),
            pool_size: num_cpus::get(),
            spawning_strategy: SpawningStrategy::Dynamic,
        }
    }
}

impl Default for CpuMonitoring {
    fn default() -> Self {
        Self {
            utilization: true,
            load_average: true,
            context_switches: false,
        }
    }
}

impl Default for IoManagement {
    fn default() -> Self {
        Self {
            limits: IoLimits::default(),
            optimization: IoOptimization::default(),
            monitoring: IoMonitoring::default(),
        }
    }
}

impl Default for IoLimits {
    fn default() -> Self {
        Self {
            max_iops: 10000,
            max_bandwidth: 100 * 1024 * 1024, // 100 MB/s
            timeout: Duration::from_secs(5),
        }
    }
}

impl Default for IoOptimization {
    fn default() -> Self {
        Self {
            buffering: true,
            buffer_size: 64 * 1024, // 64KB
            scheduling: IoScheduling::SCAN,
        }
    }
}

impl Default for IoMonitoring {
    fn default() -> Self {
        Self {
            latency: true,
            throughput: true,
            errors: true,
        }
    }
}

impl Default for ScalabilityConfiguration {
    fn default() -> Self {
        Self {
            horizontal: HorizontalScaling::default(),
            vertical: VerticalScaling::default(),
            load_balancing: LoadBalancing::default(),
        }
    }
}

impl Default for HorizontalScaling {
    fn default() -> Self {
        Self {
            auto_scaling: false,
            min_instances: 1,
            max_instances: 10,
            triggers: vec![
                ScalingTrigger::CpuUtilization { threshold: 80.0 },
                ScalingTrigger::MemoryUtilization { threshold: 85.0 },
            ],
        }
    }
}

impl Default for VerticalScaling {
    fn default() -> Self {
        Self {
            auto_scaling: false,
            limits: VerticalLimits::default(),
            monitoring: VerticalPerformanceMonitoring::default(),
        }
    }
}

impl Default for VerticalLimits {
    fn default() -> Self {
        Self {
            max_cpu_cores: num_cpus::get() * 2,
            max_memory: 16 * 1024 * 1024 * 1024, // 16GB
            allocation_strategy: AllocationStrategy::Adaptive,
        }
    }
}

impl Default for VerticalPerformanceMonitoring {
    fn default() -> Self {
        Self {
            metrics: vec![
                VerticalMetric::ResourceUtilization,
                VerticalMetric::PerformanceDegradation,
            ],
            triggers: vec![
                ScalingTrigger::CpuUtilization { threshold: 90.0 },
            ],
            cooldown: Duration::from_secs(300),
        }
    }
}

impl Default for LoadBalancing {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::RoundRobin,
            health_checking: HealthChecking::default(),
            monitoring: LoadBalancingMonitoring::default(),
        }
    }
}

impl Default for HealthChecking {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            retries: 3,
            endpoint: "/health".to_string(),
        }
    }
}

impl Default for LoadBalancingMonitoring {
    fn default() -> Self {
        Self {
            metrics: vec![
                LoadBalancingMetric::LoadDistribution,
                LoadBalancingMetric::ResponseTimeVariance,
            ],
            frequency: Duration::from_secs(10),
            thresholds: LoadBalancingThresholds::default(),
        }
    }
}

impl Default for LoadBalancingThresholds {
    fn default() -> Self {
        Self {
            max_load_imbalance: 0.2,
            max_response_variance: Duration::from_millis(100),
            max_error_rate: 0.05,
        }
    }
}