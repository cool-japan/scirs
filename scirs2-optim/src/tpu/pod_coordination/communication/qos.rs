//! Quality of Service Management
//!
//! This module provides QoS management for TPU communication including
//! traffic classification, bandwidth allocation, and priority scheduling.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Quality of Service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSConfig {
    /// Enable QoS
    pub enabled: bool,
    /// Traffic classes
    pub traffic_classes: Vec<TrafficClass>,
    /// Bandwidth allocation
    pub bandwidth_allocation: BandwidthAllocation,
    /// Priority scheduling
    pub priority_scheduling: PriorityScheduling,
    /// Flow control
    pub flow_control: FlowControl,
}

/// QoS class enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QoSClass {
    /// Best effort
    BestEffort,
    /// Assured forwarding
    AssuredForwarding,
    /// Expedited forwarding
    ExpeditedForwarding,
    /// Voice
    Voice,
    /// Video
    Video,
    /// Control
    Control,
    /// Custom class
    Custom { name: String, priority: u8 },
}

/// Traffic class definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficClass {
    /// Class name
    pub name: String,
    /// Traffic priority
    pub priority: TrafficPriority,
    /// Bandwidth guarantee
    pub bandwidth_guarantee: f64,
    /// Maximum bandwidth
    pub max_bandwidth: f64,
    /// Latency requirements
    pub latency_requirements: LatencyRequirements,
    /// Jitter requirements
    pub jitter_requirements: JitterRequirements,
    /// Packet loss tolerance
    pub packet_loss_tolerance: f64,
}

/// Traffic priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrafficPriority {
    /// Real-time critical
    RealTimeCritical = 7,
    /// Real-time
    RealTime = 6,
    /// Network control
    NetworkControl = 5,
    /// Interactive multimedia
    InteractiveMultimedia = 4,
    /// Streaming multimedia
    StreamingMultimedia = 3,
    /// Broadcast video
    BroadcastVideo = 2,
    /// Bulk data
    BulkData = 1,
    /// Best effort
    BestEffort = 0,
}

/// Latency requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyRequirements {
    /// Maximum latency
    pub max_latency: Duration,
    /// Target latency
    pub target_latency: Duration,
    /// Latency variation tolerance
    pub variation_tolerance: Duration,
}

/// Jitter requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterRequirements {
    /// Maximum jitter
    pub max_jitter: Duration,
    /// Target jitter
    pub target_jitter: Duration,
    /// Jitter buffer size
    pub buffer_size: Duration,
}

/// Bandwidth allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BandwidthAllocation {
    /// Total bandwidth
    pub total_bandwidth: f64,
    /// Allocation strategy
    pub strategy: BandwidthAllocationStrategy,
    /// Fair sharing configuration
    pub fair_sharing: FairSharingConfig,
}

/// Bandwidth allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BandwidthAllocationStrategy {
    /// Static allocation
    Static,
    /// Dynamic allocation
    Dynamic,
    /// Proportional allocation
    Proportional,
    /// Priority-based allocation
    PriorityBased,
    /// Fair queuing
    FairQueuing,
}

/// Fair sharing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairSharingConfig {
    /// Enable fair sharing
    pub enabled: bool,
    /// Fairness algorithm
    pub algorithm: FairnessAlgorithm,
    /// Sharing granularity
    pub granularity: SharingGranularity,
    /// Fairness monitoring
    pub monitoring: FairnessMonitoring,
    /// Corrective actions
    pub corrective_actions: CorrectiveActions,
}

/// Fairness algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessAlgorithm {
    /// Weighted fair queuing
    WeightedFairQueuing,
    /// Deficit round robin
    DeficitRoundRobin,
    /// Hierarchical fair service curve
    HierarchicalFairServiceCurve,
    /// Start-time fair queuing
    StartTimeFairQueuing,
}

/// Sharing granularity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingGranularity {
    /// Per flow
    PerFlow,
    /// Per class
    PerClass,
    /// Per user
    PerUser,
    /// Per application
    PerApplication,
}

/// Fairness monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessMonitoring {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Fairness metrics
    pub metrics: Vec<String>,
    /// Violation thresholds
    pub thresholds: HashMap<String, f64>,
}

/// Corrective actions for fairness violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectiveActions {
    /// Enable corrective actions
    pub enabled: bool,
    /// Correction strategy
    pub strategy: CorrectionStrategy,
    /// Action triggers
    pub triggers: Vec<String>,
    /// Recovery time
    pub recovery_time: Duration,
}

/// Correction strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CorrectionStrategy {
    /// Rate limiting
    RateLimiting,
    /// Priority adjustment
    PriorityAdjustment,
    /// Queue reordering
    QueueReordering,
    /// Resource reallocation
    ResourceReallocation,
}

/// Priority scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityScheduling {
    /// Enable priority scheduling
    pub enabled: bool,
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    /// Queue configuration
    pub queue_config: QueueConfiguration,
    /// Preemption settings
    pub preemption: PreemptionSettings,
}

/// Scheduling algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    /// Strict priority
    StrictPriority,
    /// Weighted round robin
    WeightedRoundRobin,
    /// Deficit weighted round robin
    DeficitWeightedRoundRobin,
    /// Hierarchical scheduling
    Hierarchical,
    /// Custom algorithm
    Custom { name: String, parameters: HashMap<String, f64> },
}

/// Queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueConfiguration {
    /// Number of queues
    pub num_queues: usize,
    /// Queue sizes
    pub queue_sizes: HashMap<TrafficPriority, usize>,
    /// Queue management
    pub management: QueueManagement,
}

/// Queue management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueManagement {
    /// Drop policy
    pub drop_policy: DropPolicy,
    /// Congestion control
    pub congestion_control: QueueCongestionControl,
    /// Buffer management
    pub buffer_management: QueueBufferManagement,
    /// Memory management
    pub memory_management: QueueMemoryManagement,
}

/// Drop policies for queue management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DropPolicy {
    /// Drop tail
    DropTail,
    /// Random early detection
    RandomEarlyDetection,
    /// Weighted random early detection
    WeightedRandomEarlyDetection,
    /// Blue
    Blue,
    /// Adaptive virtual queue
    AdaptiveVirtualQueue,
}

/// Queue congestion control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueCongestionControl {
    /// Enable congestion control
    pub enabled: bool,
    /// Detection method
    pub detection_method: CongestionDetectionMethod,
    /// Response strategy
    pub response_strategy: CongestionResponseStrategy,
    /// Recovery mechanism
    pub recovery_mechanism: CongestionRecoveryMechanism,
}

/// Congestion detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionDetectionMethod {
    /// Queue length based
    QueueLength { threshold: f64 },
    /// Delay based
    DelayBased { threshold: Duration },
    /// Loss based
    LossBased { threshold: f64 },
    /// Hybrid detection
    Hybrid,
}

/// Congestion response strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionResponseStrategy {
    /// Drop packets
    DropPackets,
    /// Mark packets
    MarkPackets,
    /// Rate limiting
    RateLimiting,
    /// Priority adjustment
    PriorityAdjustment,
}

/// Congestion recovery mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CongestionRecoveryMechanism {
    /// Gradual recovery
    Gradual { rate: f64 },
    /// Immediate recovery
    Immediate,
    /// Adaptive recovery
    Adaptive,
}

/// Queue buffer management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueBufferManagement {
    /// Buffer allocation strategy
    pub allocation_strategy: BufferAllocationStrategy,
    /// Shared buffer settings
    pub shared_buffer: SharedBufferSettings,
    /// Buffer isolation settings
    pub isolation: BufferIsolationSettings,
}

/// Buffer allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferAllocationStrategy {
    /// Static allocation
    Static,
    /// Dynamic allocation
    Dynamic,
    /// Adaptive allocation
    Adaptive,
}

/// Shared buffer settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedBufferSettings {
    /// Enable shared buffers
    pub enabled: bool,
    /// Sharing policy
    pub policy: BufferSharingPolicy,
    /// Maximum shared size
    pub max_shared_size: usize,
}

/// Buffer sharing policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BufferSharingPolicy {
    /// No sharing
    NoSharing,
    /// Full sharing
    FullSharing,
    /// Threshold sharing
    ThresholdSharing { threshold: f64 },
    /// Priority sharing
    PrioritySharing,
}

/// Buffer isolation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferIsolationSettings {
    /// Enable isolation
    pub enabled: bool,
    /// Isolation method
    pub method: IsolationMethod,
}

/// Isolation methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationMethod {
    /// Physical isolation
    Physical,
    /// Virtual isolation
    Virtual,
    /// Logical isolation
    Logical,
}

/// Queue memory management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueMemoryManagement {
    /// Memory allocation strategy
    pub allocation_strategy: MemoryAllocationStrategy,
    /// Garbage collection settings
    pub garbage_collection: GarbageCollectionSettings,
    /// Memory optimization settings
    pub optimization: MemoryOptimizationSettings,
}

/// Memory allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryAllocationStrategy {
    /// Pre-allocated
    PreAllocated,
    /// On-demand
    OnDemand,
    /// Pool-based
    PoolBased,
    /// Hybrid
    Hybrid,
}

/// Garbage collection settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarbageCollectionSettings {
    /// Enable garbage collection
    pub enabled: bool,
    /// Collection strategy
    pub strategy: GarbageCollectionStrategy,
    /// Collection frequency
    pub frequency: Duration,
    /// Memory threshold
    pub memory_threshold: f64,
}

/// Garbage collection strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GarbageCollectionStrategy {
    /// Mark and sweep
    MarkAndSweep,
    /// Reference counting
    ReferenceCounting,
    /// Generational
    Generational,
}

/// Memory optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationSettings {
    /// Enable optimization
    pub enabled: bool,
    /// Optimization strategy
    pub strategy: MemoryOptimizationStrategy,
    /// Optimization interval
    pub interval: Duration,
}

/// Memory optimization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryOptimizationStrategy {
    /// Compaction
    Compaction,
    /// Defragmentation
    Defragmentation,
    /// Compression
    Compression,
}

/// Preemption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionSettings {
    /// Enable preemption
    pub enabled: bool,
    /// Preemption policy
    pub policy: PreemptionPolicy,
    /// Preemption thresholds
    pub thresholds: PreemptionThresholds,
    /// Recovery settings
    pub recovery: PreemptionRecovery,
}

/// Preemption policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionPolicy {
    /// Priority-based preemption
    PriorityBased,
    /// Time-based preemption
    TimeBased,
    /// Resource-based preemption
    ResourceBased,
    /// Adaptive preemption
    Adaptive,
}

/// Preemption thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionThresholds {
    /// Priority difference threshold
    pub priority_threshold: u8,
    /// Resource utilization threshold
    pub resource_threshold: f64,
    /// Time threshold
    pub time_threshold: Duration,
}

/// Preemption recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionRecovery {
    /// Recovery strategy
    pub strategy: RecoveryStrategy,
    /// Compensation mechanism
    pub compensation: CompensationMechanism,
    /// Recovery timeout
    pub timeout: Duration,
}

/// Recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Restart
    Restart,
    /// Resume
    Resume,
    /// Reschedule
    Reschedule,
}

/// Compensation mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompensationMechanism {
    /// Priority boost
    PriorityBoost,
    /// Resource allocation
    ResourceAllocation,
    /// Time extension
    TimeExtension,
}

/// Flow control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowControl {
    /// Enable flow control
    pub enabled: bool,
    /// Flow control mechanism
    pub mechanism: FlowControlMechanism,
    /// Window settings
    pub window_settings: WindowSettings,
    /// Credit-based settings
    pub credit_settings: CreditBasedSettings,
    /// Back pressure settings
    pub back_pressure: BackPressureSettings,
}

/// Flow control mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowControlMechanism {
    /// Window-based flow control
    WindowBased,
    /// Credit-based flow control
    CreditBased,
    /// Rate-based flow control
    RateBased,
    /// Hybrid flow control
    Hybrid,
}

/// Window settings for flow control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSettings {
    /// Initial window size
    pub initial_window_size: usize,
    /// Maximum window size
    pub max_window_size: usize,
    /// Adaptive window sizing
    pub adaptive_sizing: AdaptiveWindowSizing,
}

/// Adaptive window sizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveWindowSizing {
    /// Enable adaptive sizing
    pub enabled: bool,
    /// Adaptation algorithm
    pub algorithm: WindowAdaptationAlgorithm,
    /// Adaptation parameters
    pub parameters: WindowAdaptationParameters,
}

/// Window adaptation algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WindowAdaptationAlgorithm {
    /// Additive increase, multiplicative decrease
    AIMD,
    /// Binary increase congestion control
    BIC,
    /// CUBIC
    CUBIC,
    /// Custom algorithm
    Custom { name: String },
}

/// Window adaptation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowAdaptationParameters {
    /// Increase factor
    pub increase_factor: f64,
    /// Decrease factor
    pub decrease_factor: f64,
    /// Adaptation interval
    pub adaptation_interval: Duration,
}

/// Credit-based settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditBasedSettings {
    /// Initial credits
    pub initial_credits: usize,
    /// Maximum credits
    pub max_credits: usize,
    /// Credit management
    pub management: CreditManagement,
    /// Credit monitoring
    pub monitoring: CreditMonitoring,
}

/// Credit management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditManagement {
    /// Credit allocation strategy
    pub allocation_strategy: CreditAllocationStrategy,
    /// Credit recovery mechanism
    pub recovery_mechanism: CreditRecoveryMechanism,
}

/// Credit allocation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreditAllocationStrategy {
    /// Static allocation
    Static,
    /// Dynamic allocation
    Dynamic,
    /// Proportional allocation
    Proportional,
}

/// Credit recovery mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreditRecoveryMechanism {
    /// Time-based recovery
    TimeBased { interval: Duration },
    /// Acknowledgment-based recovery
    AcknowledgmentBased,
    /// Hybrid recovery
    Hybrid,
}

/// Credit monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditMonitoring {
    /// Enable monitoring
    pub enabled: bool,
    /// Monitoring interval
    pub interval: Duration,
    /// Credit exhaustion handling
    pub exhaustion_handling: CreditExhaustionHandling,
}

/// Credit exhaustion handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CreditExhaustionHandling {
    /// Block sender
    BlockSender,
    /// Drop packets
    DropPackets,
    /// Queue packets
    QueuePackets,
}

/// Back pressure settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackPressureSettings {
    /// Enable back pressure
    pub enabled: bool,
    /// Propagation strategy
    pub propagation: BackPressurePropagation,
    /// Recovery settings
    pub recovery: BackPressureRecovery,
}

/// Back pressure propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackPressurePropagation {
    /// Hop-by-hop propagation
    HopByHop,
    /// End-to-end propagation
    EndToEnd,
    /// Selective propagation
    Selective,
}

/// Back pressure recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackPressureRecovery {
    /// Recovery strategy
    pub strategy: BackPressureRecoveryStrategy,
    /// Recovery timeout
    pub timeout: Duration,
}

/// Back pressure recovery strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BackPressureRecoveryStrategy {
    /// Gradual recovery
    Gradual,
    /// Immediate recovery
    Immediate,
    /// Adaptive recovery
    Adaptive,
}

/// QoS requirements for communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QoSRequirements {
    /// Required QoS class
    pub qos_class: QoSClass,
    /// Bandwidth requirements
    pub bandwidth: Option<f64>,
    /// Latency requirements
    pub latency: Option<Duration>,
    /// Jitter tolerance
    pub jitter: Option<Duration>,
    /// Packet loss tolerance
    pub packet_loss: Option<f64>,
    /// Priority level
    pub priority: Option<TrafficPriority>,
}

/// Reliability requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityRequirements {
    /// Delivery guarantee
    pub delivery_guarantee: DeliveryGuarantee,
    /// Ordering guarantee
    pub ordering_guarantee: OrderingGuarantee,
    /// Duplicate handling
    pub duplicate_handling: DuplicateHandling,
    /// Error detection
    pub error_detection: ErrorDetection,
    /// Recovery mechanism
    pub recovery_mechanism: RecoveryMechanism,
}

/// Delivery guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    /// Best effort
    BestEffort,
    /// At most once
    AtMostOnce,
    /// At least once
    AtLeastOnce,
    /// Exactly once
    ExactlyOnce,
}

/// Ordering guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OrderingGuarantee {
    /// No ordering guarantee
    None,
    /// FIFO ordering
    FIFO,
    /// Causal ordering
    Causal,
    /// Total ordering
    Total,
}

/// Duplicate handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DuplicateHandling {
    /// Allow duplicates
    Allow,
    /// Detect duplicates
    Detect,
    /// Suppress duplicates
    Suppress,
}

/// Error detection mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorDetection {
    /// Checksum
    Checksum,
    /// CRC
    CRC,
    /// Hash-based
    HashBased,
    /// Custom detection
    Custom { method: String },
}

/// Recovery mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryMechanism {
    /// Retransmission
    Retransmission,
    /// Forward error correction
    ForwardErrorCorrection,
    /// Hybrid recovery
    Hybrid,
    /// Custom recovery
    Custom { method: String },
}

impl Default for QoSConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            traffic_classes: vec![
                TrafficClass {
                    name: "critical".to_string(),
                    priority: TrafficPriority::RealTimeCritical,
                    bandwidth_guarantee: 0.3,
                    max_bandwidth: 0.5,
                    latency_requirements: LatencyRequirements {
                        max_latency: Duration::from_micros(100),
                        target_latency: Duration::from_micros(50),
                        variation_tolerance: Duration::from_micros(10),
                    },
                    jitter_requirements: JitterRequirements {
                        max_jitter: Duration::from_micros(20),
                        target_jitter: Duration::from_micros(5),
                        buffer_size: Duration::from_millis(1),
                    },
                    packet_loss_tolerance: 0.001,
                },
                TrafficClass {
                    name: "normal".to_string(),
                    priority: TrafficPriority::BestEffort,
                    bandwidth_guarantee: 0.4,
                    max_bandwidth: 0.7,
                    latency_requirements: LatencyRequirements {
                        max_latency: Duration::from_millis(10),
                        target_latency: Duration::from_millis(5),
                        variation_tolerance: Duration::from_millis(1),
                    },
                    jitter_requirements: JitterRequirements {
                        max_jitter: Duration::from_millis(1),
                        target_jitter: Duration::from_micros(100),
                        buffer_size: Duration::from_millis(10),
                    },
                    packet_loss_tolerance: 0.01,
                },
            ],
            bandwidth_allocation: BandwidthAllocation::default(),
            priority_scheduling: PriorityScheduling::default(),
            flow_control: FlowControl::default(),
        }
    }
}

impl Default for BandwidthAllocation {
    fn default() -> Self {
        Self {
            total_bandwidth: 10_000_000_000.0, // 10 Gbps
            strategy: BandwidthAllocationStrategy::PriorityBased,
            fair_sharing: FairSharingConfig {
                enabled: true,
                algorithm: FairnessAlgorithm::WeightedFairQueuing,
                granularity: SharingGranularity::PerFlow,
                monitoring: FairnessMonitoring {
                    enabled: true,
                    interval: Duration::from_secs(60),
                    metrics: vec!["fairness_index".to_string(), "utilization".to_string()],
                    thresholds: HashMap::new(),
                },
                corrective_actions: CorrectiveActions {
                    enabled: true,
                    strategy: CorrectionStrategy::ResourceReallocation,
                    triggers: vec!["fairness_violation".to_string()],
                    recovery_time: Duration::from_secs(30),
                },
            },
        }
    }
}

impl Default for PriorityScheduling {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: SchedulingAlgorithm::WeightedRoundRobin,
            queue_config: QueueConfiguration {
                num_queues: 8,
                queue_sizes: HashMap::new(),
                management: QueueManagement {
                    drop_policy: DropPolicy::WeightedRandomEarlyDetection,
                    congestion_control: QueueCongestionControl {
                        enabled: true,
                        detection_method: CongestionDetectionMethod::QueueLength { threshold: 0.8 },
                        response_strategy: CongestionResponseStrategy::MarkPackets,
                        recovery_mechanism: CongestionRecoveryMechanism::Gradual { rate: 0.1 },
                    },
                    buffer_management: QueueBufferManagement {
                        allocation_strategy: BufferAllocationStrategy::Dynamic,
                        shared_buffer: SharedBufferSettings {
                            enabled: true,
                            policy: BufferSharingPolicy::ThresholdSharing { threshold: 0.7 },
                            max_shared_size: 1024 * 1024, // 1MB
                        },
                        isolation: BufferIsolationSettings {
                            enabled: false,
                            method: IsolationMethod::Virtual,
                        },
                    },
                    memory_management: QueueMemoryManagement {
                        allocation_strategy: MemoryAllocationStrategy::PoolBased,
                        garbage_collection: GarbageCollectionSettings {
                            enabled: true,
                            strategy: GarbageCollectionStrategy::MarkAndSweep,
                            frequency: Duration::from_secs(300),
                            memory_threshold: 0.8,
                        },
                        optimization: MemoryOptimizationSettings {
                            enabled: true,
                            strategy: MemoryOptimizationStrategy::Compaction,
                            interval: Duration::from_secs(600),
                        },
                    },
                },
            },
            preemption: PreemptionSettings {
                enabled: true,
                policy: PreemptionPolicy::PriorityBased,
                thresholds: PreemptionThresholds {
                    priority_threshold: 2,
                    resource_threshold: 0.9,
                    time_threshold: Duration::from_millis(100),
                },
                recovery: PreemptionRecovery {
                    strategy: RecoveryStrategy::Reschedule,
                    compensation: CompensationMechanism::PriorityBoost,
                    timeout: Duration::from_secs(30),
                },
            },
        }
    }
}

impl Default for FlowControl {
    fn default() -> Self {
        Self {
            enabled: true,
            mechanism: FlowControlMechanism::WindowBased,
            window_settings: WindowSettings {
                initial_window_size: 64 * 1024, // 64KB
                max_window_size: 1024 * 1024, // 1MB
                adaptive_sizing: AdaptiveWindowSizing {
                    enabled: true,
                    algorithm: WindowAdaptationAlgorithm::CUBIC,
                    parameters: WindowAdaptationParameters {
                        increase_factor: 1.2,
                        decrease_factor: 0.8,
                        adaptation_interval: Duration::from_millis(100),
                    },
                },
            },
            credit_settings: CreditBasedSettings {
                initial_credits: 1000,
                max_credits: 10000,
                management: CreditManagement {
                    allocation_strategy: CreditAllocationStrategy::Dynamic,
                    recovery_mechanism: CreditRecoveryMechanism::AcknowledgmentBased,
                },
                monitoring: CreditMonitoring {
                    enabled: true,
                    interval: Duration::from_secs(10),
                    exhaustion_handling: CreditExhaustionHandling::QueuePackets,
                },
            },
            back_pressure: BackPressureSettings {
                enabled: true,
                propagation: BackPressurePropagation::HopByHop,
                recovery: BackPressureRecovery {
                    strategy: BackPressureRecoveryStrategy::Gradual,
                    timeout: Duration::from_secs(60),
                },
            },
        }
    }
}

impl Default for QoSRequirements {
    fn default() -> Self {
        Self {
            qos_class: QoSClass::BestEffort,
            bandwidth: None,
            latency: None,
            jitter: None,
            packet_loss: None,
            priority: None,
        }
    }
}

impl Default for ReliabilityRequirements {
    fn default() -> Self {
        Self {
            delivery_guarantee: DeliveryGuarantee::BestEffort,
            ordering_guarantee: OrderingGuarantee::FIFO,
            duplicate_handling: DuplicateHandling::Detect,
            error_detection: ErrorDetection::CRC,
            recovery_mechanism: RecoveryMechanism::Retransmission,
        }
    }
}