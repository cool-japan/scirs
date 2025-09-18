//! Core Communication Management
//!
//! This module implements the central communication manager and core coordination
//! logic for TPU pod communication. It provides the main interface for managing
//! communications between TPU devices in a pod configuration.

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::buffer_management::*;
use super::compression::*;
use super::network_config::*;
use super::qos::*;
use super::reliability::*;
use super::monitoring::*;
use super::routing::*;
use crate::error::{OptimError, Result};

// Re-export from tpu_backend
use crate::tpu::tpu_backend::DeviceId;

// Type aliases for communication management
pub type MessageId = u64;
pub type CommunicationId = u64;
pub type CommunicationStatistics = HashMap<String, f64>;

/// Communication manager for TPU devices
///
/// Central coordinator for all communication between TPU devices in a pod.
/// Manages message routing, buffering, compression, quality of service,
/// and performance monitoring.
#[derive(Debug)]
pub struct CommunicationManager<T: Float> {
    /// Communication configuration
    pub config: CommunicationConfig,
    /// Active communications tracking
    pub active_communications: HashMap<CommunicationId, ActiveCommunication<T>>,
    /// Message buffer pool
    pub buffer_pool: MessageBufferPool<T>,
    /// Communication scheduler
    pub scheduler: CommunicationScheduler,
    /// Compression engine
    pub compression_engine: CompressionEngine<T>,
    /// Network monitor
    pub network_monitor: NetworkMonitor,
    /// Performance statistics
    pub statistics: CommunicationStatistics,
    /// Message routing table
    pub routing_table: RoutingTable,
}

/// Configuration for communication management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Maximum number of active communications
    pub max_active_communications: usize,
    /// Default message timeout
    pub default_timeout: Duration,
    /// Buffer pool configuration
    pub buffer_pool_config: BufferPoolConfig,
    /// Compression settings
    pub compression_config: CompressionConfig,
    /// Network configuration
    pub network_config: NetworkConfig,
    /// Quality of service settings
    pub qos_config: QoSConfig,
    /// Reliability settings
    pub reliability_config: ReliabilityConfig,
    /// Performance optimization settings
    pub optimization_config: OptimizationConfig,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable automatic optimization
    pub enable_auto_optimization: bool,
    /// Optimization strategies
    pub optimization_strategies: Vec<OptimizationStrategy>,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
    /// Optimization monitoring
    pub monitoring_config: OptimizationMonitoring,
    /// Adaptation settings
    pub adaptation_config: AdaptationConfig,
}

/// Optimization strategies available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Minimize latency
    MinimizeLatency,
    /// Maximize throughput
    MaximizeThroughput,
    /// Minimize energy consumption
    MinimizeEnergy,
    /// Balance latency and throughput
    BalanceLatencyThroughput { weight: f64 },
    /// Custom optimization function
    Custom { name: String, parameters: HashMap<String, f64> },
}

/// Performance targets for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Target latency (microseconds)
    pub target_latency_us: Option<f64>,
    /// Target throughput (messages/sec)
    pub target_throughput_mps: Option<f64>,
    /// Target bandwidth utilization
    pub target_bandwidth_utilization: Option<f64>,
    /// Maximum acceptable packet loss
    pub max_packet_loss_rate: f64,
    /// Maximum acceptable jitter
    pub max_jitter_us: f64,
}

/// Optimization monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMonitoring {
    /// Monitoring interval
    pub interval: Duration,
    /// Performance history window
    pub history_window_size: usize,
    /// Metrics to track
    pub tracked_metrics: Vec<PerformanceMetric>,
    /// Anomaly detection settings
    pub anomaly_detection: AnomalyDetectionConfig,
}

/// Performance metrics to track
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Average latency
    AverageLatency,
    /// 95th percentile latency
    P95Latency,
    /// 99th percentile latency
    P99Latency,
    /// Message throughput
    MessageThroughput,
    /// Bandwidth utilization
    BandwidthUtilization,
    /// Packet loss rate
    PacketLossRate,
    /// Jitter
    Jitter,
    /// CPU utilization
    CpuUtilization,
    /// Memory utilization
    MemoryUtilization,
    /// Network interface utilization
    NetworkInterfaceUtilization,
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detection algorithm
    pub algorithm: AnomalyDetectionAlgorithm,
    /// Sensitivity settings
    pub sensitivity: f64,
    /// Response actions
    pub response_actions: Vec<AnomalyResponseAction>,
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical outlier detection
    StatisticalOutlier { threshold: f64 },
    /// Moving average based
    MovingAverage { window_size: usize, threshold: f64 },
    /// Exponential smoothing
    ExponentialSmoothing { alpha: f64, threshold: f64 },
    /// Machine learning based
    MachineLearning { model_type: String, parameters: HashMap<String, f64> },
}

/// Actions to take when anomalies are detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyResponseAction {
    /// Log the anomaly
    Log,
    /// Send alert notification
    Alert { severity: AlertSeverity },
    /// Trigger adaptation
    TriggerAdaptation,
    /// Increase monitoring frequency
    IncreaseMonitoring { factor: f64 },
    /// Execute custom action
    Custom { action: String, parameters: HashMap<String, String> },
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical alert
    Critical,
}

/// Adaptation configuration for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    /// Enable adaptive optimization
    pub enabled: bool,
    /// Adaptation algorithm
    pub algorithm: AdaptationAlgorithm,
    /// Adaptation rate
    pub adaptation_rate: f64,
    /// Stability requirements
    pub stability_requirements: StabilityRequirements,
    /// Rollback settings
    pub rollback_config: RollbackConfig,
}

/// Adaptation algorithms available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationAlgorithm {
    /// Gradient descent optimization
    GradientDescent { learning_rate: f64, momentum: f64 },
    /// Genetic algorithm
    GeneticAlgorithm { population_size: usize, mutation_rate: f64 },
    /// Simulated annealing
    SimulatedAnnealing { initial_temperature: f64, cooling_rate: f64 },
    /// Reinforcement learning
    ReinforcementLearning { algorithm: String, parameters: HashMap<String, f64> },
}

/// Stability requirements for adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityRequirements {
    /// Minimum stability period before adaptation
    pub min_stability_period: Duration,
    /// Maximum allowed performance variance
    pub max_performance_variance: f64,
    /// Convergence criteria
    pub convergence_criteria: ConvergenceCriteria,
}

/// Convergence criteria for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    /// Maximum improvement threshold
    pub max_improvement_threshold: f64,
    /// Minimum improvement threshold
    pub min_improvement_threshold: f64,
    /// Evaluation window for convergence
    pub evaluation_window: Duration,
    /// Consecutive evaluations required
    pub consecutive_evaluations: usize,
}

/// Rollback configuration for failed adaptations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollbackConfig {
    /// Enable automatic rollback
    pub enable_auto_rollback: bool,
    /// Performance degradation threshold for rollback
    pub performance_degradation_threshold: f64,
    /// Rollback evaluation period
    pub evaluation_period: Duration,
    /// Number of backup configurations to maintain
    pub backup_configurations: usize,
}

/// Active communication tracking
#[derive(Debug, Clone)]
pub struct ActiveCommunication<T: Float> {
    /// Communication ID
    pub id: CommunicationId,
    /// Source device
    pub source: DeviceId,
    /// Destination device
    pub destination: DeviceId,
    /// Message type
    pub message_type: MessageType,
    /// Priority level
    pub priority: Priority,
    /// Start time
    pub start_time: Instant,
    /// Expected completion time
    pub expected_completion: Option<Instant>,
    /// Current status
    pub status: CommunicationStatus,
    /// Associated resources
    pub resources: CommunicationResources<T>,
    /// Performance metrics
    pub metrics: CommunicationMetrics<T>,
}

/// Message types for communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Control message
    Control,
    /// Data transfer
    DataTransfer,
    /// Synchronization message
    Synchronization,
    /// Heartbeat message
    Heartbeat,
    /// Error/fault message
    Error,
    /// Status update
    StatusUpdate,
    /// Configuration change
    ConfigurationChange,
    /// Custom message type
    Custom { name: String },
}

/// Priority levels for communications
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

/// Communication status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationStatus {
    /// Initializing communication
    Initializing,
    /// In progress
    InProgress { progress: f64 },
    /// Waiting for resources
    WaitingForResources,
    /// Completed successfully
    Completed,
    /// Failed with error
    Failed { error: String },
    /// Cancelled
    Cancelled,
    /// Timeout occurred
    Timeout,
}

/// Resources associated with a communication
#[derive(Debug, Clone)]
pub struct CommunicationResources<T: Float> {
    /// Allocated buffers
    pub buffers: Vec<BufferId>,
    /// Network bandwidth allocation
    pub bandwidth_allocation: T,
    /// CPU resources allocated
    pub cpu_allocation: T,
    /// Memory allocation
    pub memory_allocation: T,
    /// Quality of service class
    pub qos_class: QoSClass,
}

/// Communication performance metrics
#[derive(Debug, Clone)]
pub struct CommunicationMetrics<T: Float> {
    /// Latency measurements
    pub latency_us: T,
    /// Throughput measurements
    pub throughput_mbps: T,
    /// Bytes transferred
    pub bytes_transferred: u64,
    /// Retry count
    pub retry_count: u32,
    /// Error count
    pub error_count: u32,
    /// Compression ratio achieved
    pub compression_ratio: T,
    /// Resource utilization metrics
    pub resource_utilization: ResourceUtilizationMetrics<T>,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilizationMetrics<T: Float> {
    /// CPU utilization percentage
    pub cpu_utilization: T,
    /// Memory utilization percentage
    pub memory_utilization: T,
    /// Network bandwidth utilization
    pub bandwidth_utilization: T,
    /// Buffer pool utilization
    pub buffer_utilization: T,
}

/// Communication scheduler for managing message ordering and timing
#[derive(Debug)]
pub struct CommunicationScheduler {
    /// Scheduling algorithm
    pub algorithm: SchedulingAlgorithm,
    /// Scheduled communications queue
    pub scheduled_queue: VecDeque<ScheduledCommunication>,
    /// Priority queues for different priority levels
    pub priority_queues: HashMap<Priority, VecDeque<ScheduledCommunication>>,
    /// Scheduler statistics
    pub statistics: SchedulerStatistics,
    /// Configuration for scheduler behavior
    pub config: SchedulerConfig,
}

/// Scheduling algorithms available
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingAlgorithm {
    /// First-come, first-served
    FCFS,
    /// Shortest job first
    SJF,
    /// Priority-based scheduling
    Priority,
    /// Round-robin scheduling
    RoundRobin { time_quantum: Duration },
    /// Weighted fair queuing
    WeightedFairQueuing { weights: HashMap<Priority, f64> },
    /// Earliest deadline first
    EarliestDeadlineFirst,
    /// Custom scheduling algorithm
    Custom { name: String, parameters: HashMap<String, f64> },
}

/// Scheduled communication entry
#[derive(Debug, Clone)]
pub struct ScheduledCommunication {
    /// Communication request
    pub communication: CommunicationRequest,
    /// Scheduled start time
    pub scheduled_time: Instant,
    /// Deadline for completion
    pub deadline: Option<Instant>,
    /// Estimated duration
    pub estimated_duration: Duration,
    /// Scheduling metadata
    pub metadata: SchedulingMetadata,
}

/// Communication request structure
#[derive(Debug, Clone)]
pub struct CommunicationRequest {
    /// Source device
    pub source: DeviceId,
    /// Destination device
    pub destination: DeviceId,
    /// Message data
    pub data: Vec<u8>,
    /// Message type
    pub message_type: MessageType,
    /// Priority
    pub priority: Priority,
    /// Quality of service requirements
    pub qos_requirements: QoSRequirements,
    /// Reliability requirements
    pub reliability_requirements: ReliabilityRequirements,
}

/// Scheduling metadata
#[derive(Debug, Clone)]
pub struct SchedulingMetadata {
    /// Submission time
    pub submission_time: Instant,
    /// Estimated resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Dependencies on other communications
    pub dependencies: Vec<CommunicationId>,
    /// Scheduling constraints
    pub constraints: Vec<SchedulingConstraint>,
}

/// Resource requirements for communication
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Required bandwidth
    pub bandwidth_mbps: f64,
    /// Required buffer space
    pub buffer_size: usize,
    /// Required CPU resources
    pub cpu_percentage: f64,
    /// Required memory
    pub memory_mb: f64,
    /// Maximum latency tolerance
    pub max_latency_us: f64,
}

/// Scheduling constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingConstraint {
    /// Must start after specific time
    StartAfter { time: Instant },
    /// Must complete before specific time
    CompleteBefore { time: Instant },
    /// Must not overlap with specific communication
    NoOverlapWith { communication_id: CommunicationId },
    /// Requires specific resources
    RequiresResource { resource: String },
    /// Custom constraint
    Custom { name: String, parameters: HashMap<String, String> },
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics {
    /// Total communications scheduled
    pub total_scheduled: u64,
    /// Total communications completed
    pub total_completed: u64,
    /// Average scheduling delay
    pub avg_scheduling_delay: Duration,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Queue utilization by priority
    pub queue_utilization: HashMap<Priority, f64>,
    /// Throughput statistics
    pub throughput_stats: ThroughputStatistics,
}

/// Throughput statistics
#[derive(Debug, Clone)]
pub struct ThroughputStatistics {
    /// Messages per second
    pub messages_per_second: f64,
    /// Bytes per second
    pub bytes_per_second: f64,
    /// Peak throughput achieved
    pub peak_throughput: f64,
    /// Average throughput over time window
    pub average_throughput: f64,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum queue size per priority
    pub max_queue_size: HashMap<Priority, usize>,
    /// Scheduling interval
    pub scheduling_interval: Duration,
    /// Preemption settings
    pub preemption_config: PreemptionConfig,
    /// Load balancing settings
    pub load_balancing_config: LoadBalancingConfig,
}

/// Preemption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreemptionConfig {
    /// Enable preemption
    pub enabled: bool,
    /// Preemption priorities
    pub preemption_priorities: HashMap<Priority, bool>,
    /// Preemption cost threshold
    pub cost_threshold: f64,
    /// Recovery strategy after preemption
    pub recovery_strategy: PreemptionRecoveryStrategy,
}

/// Recovery strategies after preemption
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreemptionRecoveryStrategy {
    /// Restart from beginning
    RestartFromBeginning,
    /// Resume from checkpoint
    ResumeFromCheckpoint,
    /// Reschedule with higher priority
    RescheduleHigherPriority,
    /// Custom recovery strategy
    Custom { strategy: String },
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable load balancing
    pub enabled: bool,
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    /// Rebalancing frequency
    pub rebalancing_frequency: Duration,
    /// Load threshold for rebalancing
    pub load_threshold: f64,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    /// Round robin
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Weighted round robin
    WeightedRoundRobin { weights: HashMap<String, f64> },
    /// Resource-based balancing
    ResourceBased,
    /// Custom algorithm
    Custom { name: String, parameters: HashMap<String, f64> },
}

impl<T: Float> CommunicationManager<T> {
    /// Create a new communication manager
    pub fn new(config: CommunicationConfig) -> Result<Self> {
        let buffer_pool = MessageBufferPool::new(config.buffer_pool_config.clone())?;
        let scheduler = CommunicationScheduler::new(SchedulerConfig::default())?;
        let compression_engine = CompressionEngine::new(config.compression_config.clone())?;
        let network_monitor = NetworkMonitor::new(config.network_config.clone())?;
        let routing_table = RoutingTable::new()?;

        Ok(Self {
            config,
            active_communications: HashMap::new(),
            buffer_pool,
            scheduler,
            compression_engine,
            network_monitor,
            statistics: HashMap::new(),
            routing_table,
        })
    }

    /// Start communication between two devices
    pub fn start_communication(
        &mut self,
        request: CommunicationRequest,
    ) -> Result<CommunicationId> {
        // Generate unique communication ID
        let comm_id = self.generate_communication_id();

        // Validate communication request
        self.validate_communication_request(&request)?;

        // Schedule the communication
        self.scheduler.schedule_communication(request.clone(), comm_id)?;

        // Create active communication tracking
        let active_comm = ActiveCommunication {
            id: comm_id,
            source: request.source,
            destination: request.destination,
            message_type: request.message_type,
            priority: request.priority,
            start_time: Instant::now(),
            expected_completion: None,
            status: CommunicationStatus::Initializing,
            resources: CommunicationResources {
                buffers: Vec::new(),
                bandwidth_allocation: T::zero(),
                cpu_allocation: T::zero(),
                memory_allocation: T::zero(),
                qos_class: QoSClass::BestEffort,
            },
            metrics: CommunicationMetrics {
                latency_us: T::zero(),
                throughput_mbps: T::zero(),
                bytes_transferred: 0,
                retry_count: 0,
                error_count: 0,
                compression_ratio: T::one(),
                resource_utilization: ResourceUtilizationMetrics {
                    cpu_utilization: T::zero(),
                    memory_utilization: T::zero(),
                    bandwidth_utilization: T::zero(),
                    buffer_utilization: T::zero(),
                },
            },
        };

        self.active_communications.insert(comm_id, active_comm);
        Ok(comm_id)
    }

    /// Get status of a communication
    pub fn get_communication_status(&self, comm_id: CommunicationId) -> Option<&CommunicationStatus> {
        self.active_communications
            .get(&comm_id)
            .map(|comm| &comm.status)
    }

    /// Cancel a communication
    pub fn cancel_communication(&mut self, comm_id: CommunicationId) -> Result<()> {
        if let Some(comm) = self.active_communications.get_mut(&comm_id) {
            comm.status = CommunicationStatus::Cancelled;
            // Clean up resources
            self.cleanup_communication_resources(comm_id)?;
        }
        Ok(())
    }

    /// Update communication statistics
    pub fn update_statistics(&mut self) {
        // Update global statistics based on active communications
        let mut total_latency = T::zero();
        let mut total_throughput = T::zero();
        let mut active_count = 0;

        for comm in self.active_communications.values() {
            if matches!(comm.status, CommunicationStatus::InProgress { .. }) {
                total_latency = total_latency + comm.metrics.latency_us;
                total_throughput = total_throughput + comm.metrics.throughput_mbps;
                active_count += 1;
            }
        }

        if active_count > 0 {
            let avg_latency = total_latency / num_traits::cast::cast(active_count).unwrap_or_else(|| T::zero());
            let avg_throughput = total_throughput / num_traits::cast::cast(active_count).unwrap_or_else(|| T::zero());

            self.statistics.insert("avg_latency_us".to_string(), avg_latency.to_f64().unwrap());
            self.statistics.insert("avg_throughput_mbps".to_string(), avg_throughput.to_f64().unwrap());
        }

        self.statistics.insert("active_communications".to_string(), active_count as f64);
        self.statistics.insert("total_communications".to_string(), self.active_communications.len() as f64);
    }

    /// Get communication statistics
    pub fn get_statistics(&self) -> &CommunicationStatistics {
        &self.statistics
    }

    // Private helper methods
    fn generate_communication_id(&self) -> CommunicationId {
        // Simple ID generation - in practice would use more sophisticated approach
        self.active_communications.len() as u64 + 1
    }

    fn validate_communication_request(&self, _request: &CommunicationRequest) -> Result<()> {
        // Validation logic would go here
        Ok(())
    }

    fn cleanup_communication_resources(&mut self, _comm_id: CommunicationId) -> Result<()> {
        // Resource cleanup logic would go here
        Ok(())
    }
}

impl CommunicationScheduler {
    /// Create a new communication scheduler
    pub fn new(config: SchedulerConfig) -> Result<Self> {
        Ok(Self {
            algorithm: SchedulingAlgorithm::Priority,
            scheduled_queue: VecDeque::new(),
            priority_queues: HashMap::new(),
            statistics: SchedulerStatistics::default(),
            config,
        })
    }

    /// Schedule a communication
    pub fn schedule_communication(
        &mut self,
        request: CommunicationRequest,
        comm_id: CommunicationId,
    ) -> Result<()> {
        let scheduled_comm = ScheduledCommunication {
            communication: request.clone(),
            scheduled_time: Instant::now(),
            deadline: None,
            estimated_duration: Duration::from_millis(100), // Default estimate
            metadata: SchedulingMetadata {
                submission_time: Instant::now(),
                resource_requirements: ResourceRequirements {
                    bandwidth_mbps: 100.0, // Default requirements
                    buffer_size: 1024,
                    cpu_percentage: 10.0,
                    memory_mb: 10.0,
                    max_latency_us: 1000.0,
                },
                dependencies: Vec::new(),
                constraints: Vec::new(),
            },
        };

        // Add to appropriate queue based on priority
        let priority = request.priority;
        let queue = self.priority_queues.entry(priority).or_insert_with(VecDeque::new);
        queue.push_back(scheduled_comm);

        self.statistics.total_scheduled += 1;
        Ok(())
    }

    /// Get next scheduled communication
    pub fn get_next_scheduled(&mut self) -> Option<ScheduledCommunication> {
        // Process queues in priority order
        for priority in [Priority::Critical, Priority::High, Priority::Normal, Priority::Low] {
            if let Some(queue) = self.priority_queues.get_mut(&priority) {
                if let Some(scheduled_comm) = queue.pop_front() {
                    return Some(scheduled_comm);
                }
            }
        }
        None
    }
}

impl Default for SchedulerStatistics {
    fn default() -> Self {
        Self {
            total_scheduled: 0,
            total_completed: 0,
            avg_scheduling_delay: Duration::from_millis(0),
            avg_wait_time: Duration::from_millis(0),
            queue_utilization: HashMap::new(),
            throughput_stats: ThroughputStatistics {
                messages_per_second: 0.0,
                bytes_per_second: 0.0,
                peak_throughput: 0.0,
                average_throughput: 0.0,
            },
        }
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        let mut max_queue_size = HashMap::new();
        max_queue_size.insert(Priority::Critical, 100);
        max_queue_size.insert(Priority::High, 200);
        max_queue_size.insert(Priority::Normal, 500);
        max_queue_size.insert(Priority::Low, 1000);

        Self {
            max_queue_size,
            scheduling_interval: Duration::from_millis(10),
            preemption_config: PreemptionConfig {
                enabled: true,
                preemption_priorities: HashMap::new(),
                cost_threshold: 0.1,
                recovery_strategy: PreemptionRecoveryStrategy::RescheduleHigherPriority,
            },
            load_balancing_config: LoadBalancingConfig {
                enabled: true,
                algorithm: LoadBalancingAlgorithm::ResourceBased,
                rebalancing_frequency: Duration::from_secs(30),
                load_threshold: 0.8,
            },
        }
    }
}