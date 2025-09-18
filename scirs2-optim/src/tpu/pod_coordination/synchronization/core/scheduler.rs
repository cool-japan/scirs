//! Coordination scheduler for TPU pod synchronization operations
//!
//! This module provides scheduling and orchestration of synchronization operations
//! across TPU devices, including operation queuing, resource allocation, and
//! execution management.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::tpu::tpu_backend::DeviceId;
use crate::error::{Result, OptimError};

use super::config::*;
use super::state::*;

/// Operation identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperationId(pub u64);

/// Coordination scheduler for managing synchronization operations
#[derive(Debug)]
pub struct CoordinationScheduler {
    /// Scheduler configuration
    pub config: SchedulerConfig,
    /// Scheduled operations
    pub scheduled_operations: HashMap<OperationId, ScheduledOperation>,
    /// Operation queue
    pub operation_queue: VecDeque<QueuedOperation>,
    /// Execution history
    pub execution_history: Vec<ExecutionRecord>,
    /// Resource manager
    pub resource_manager: ResourceManager,
    /// Next operation ID
    next_operation_id: u64,
    /// Active executions
    active_executions: HashMap<OperationId, ActiveExecution>,
    /// Scheduler state
    scheduler_state: SchedulerState,
}

/// Scheduled operation
#[derive(Debug)]
pub struct ScheduledOperation {
    /// Operation ID
    pub id: OperationId,
    /// Operation type
    pub operation_type: OperationType,
    /// Target devices
    pub target_devices: Vec<DeviceId>,
    /// Scheduled time
    pub scheduled_time: Instant,
    /// Operation parameters
    pub parameters: OperationParameters,
    /// Operation status
    pub status: OperationStatus,
    /// Creation time
    pub created_at: Instant,
    /// Priority
    pub priority: u8,
    /// Dependencies
    pub dependencies: Vec<OperationId>,
}

/// Operation types
#[derive(Debug, Clone)]
pub enum OperationType {
    /// Barrier synchronization
    BarrierSync { barrier_id: String },
    /// Event synchronization
    EventSync { event_id: String },
    /// Clock synchronization
    ClockSync,
    /// Deadlock detection
    DeadlockDetection,
    /// Consensus operation
    Consensus { proposal_id: String },
    /// Global synchronization
    GlobalSync,
    /// Device recovery
    DeviceRecovery { device_id: DeviceId },
    /// Resource reallocation
    ResourceReallocation,
    /// Health check
    HealthCheck,
    /// Performance optimization
    PerformanceOptimization,
    /// Custom operation
    Custom { operation: String },
}

/// Operation parameters
#[derive(Debug, Clone)]
pub struct OperationParameters {
    /// Operation timeout
    pub timeout: Duration,
    /// Priority
    pub priority: u8,
    /// Retry settings
    pub retry_settings: RetrySettings,
    /// Custom parameters
    pub custom_params: HashMap<String, String>,
    /// Resource requirements
    pub resource_requirements: Option<ResourceRequirements>,
    /// Quality requirements
    pub quality_requirements: Option<QualityRequirements>,
}

/// Quality requirements for operations
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    /// Minimum success rate
    pub min_success_rate: f64,
    /// Maximum latency
    pub max_latency: Duration,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
    /// Durability requirements
    pub durability: bool,
}

/// Consistency levels
#[derive(Debug, Clone)]
pub enum ConsistencyLevel {
    /// Eventual consistency
    Eventual,
    /// Strong consistency
    Strong,
    /// Weak consistency
    Weak,
    /// Sequential consistency
    Sequential,
}

/// Operation status
#[derive(Debug, Clone, PartialEq)]
pub enum OperationStatus {
    /// Operation scheduled
    Scheduled,
    /// Operation queued
    Queued,
    /// Operation running
    Running,
    /// Operation completed successfully
    Completed,
    /// Operation failed
    Failed { reason: String },
    /// Operation cancelled
    Cancelled,
    /// Operation timed out
    TimedOut,
    /// Operation waiting for dependencies
    WaitingForDependencies,
    /// Operation waiting for resources
    WaitingForResources,
}

/// Queued operation
#[derive(Debug, Clone)]
pub struct QueuedOperation {
    /// Operation ID
    pub operation_id: OperationId,
    /// Queue time
    pub queued_at: Instant,
    /// Estimated execution time
    pub estimated_duration: Duration,
    /// Dependencies
    pub dependencies: Vec<OperationId>,
    /// Resource requirements
    pub resource_requirements: Option<ResourceRequirements>,
    /// Priority
    pub priority: u8,
}

/// Active execution
#[derive(Debug)]
pub struct ActiveExecution {
    /// Operation ID
    pub operation_id: OperationId,
    /// Start time
    pub start_time: Instant,
    /// Allocated resources
    pub allocated_resources: AllocatedResources,
    /// Execution context
    pub context: ExecutionContext,
    /// Progress tracker
    pub progress: ExecutionProgress,
}

/// Execution context
#[derive(Debug)]
pub struct ExecutionContext {
    /// Execution environment
    pub environment: HashMap<String, String>,
    /// Target devices
    pub target_devices: Vec<DeviceId>,
    /// Operation metadata
    pub metadata: HashMap<String, String>,
    /// Execution constraints
    pub constraints: ExecutionConstraints,
}

/// Execution constraints
#[derive(Debug, Clone)]
pub struct ExecutionConstraints {
    /// Maximum execution time
    pub max_execution_time: Duration,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Failure tolerance
    pub failure_tolerance: FailureTolerance,
    /// Rollback policy
    pub rollback_policy: RollbackPolicy,
}

/// Failure tolerance settings
#[derive(Debug, Clone)]
pub struct FailureTolerance {
    /// Maximum failures allowed
    pub max_failures: usize,
    /// Failure threshold percentage
    pub failure_threshold: f64,
    /// Retry on failure
    pub retry_on_failure: bool,
    /// Continue on partial failure
    pub continue_on_partial_failure: bool,
}

/// Rollback policies
#[derive(Debug, Clone)]
pub enum RollbackPolicy {
    /// No rollback
    None,
    /// Automatic rollback on failure
    AutomaticOnFailure,
    /// Manual rollback only
    Manual,
    /// Conditional rollback
    Conditional { conditions: Vec<String> },
}

/// Execution progress
#[derive(Debug)]
pub struct ExecutionProgress {
    /// Progress percentage (0.0 to 1.0)
    pub progress: f64,
    /// Current phase
    pub current_phase: String,
    /// Completed phases
    pub completed_phases: Vec<String>,
    /// Progress messages
    pub messages: Vec<ProgressMessage>,
    /// Last update time
    pub last_update: Instant,
}

/// Progress message
#[derive(Debug, Clone)]
pub struct ProgressMessage {
    /// Message timestamp
    pub timestamp: Instant,
    /// Message text
    pub message: String,
    /// Message severity
    pub severity: MessageSeverity,
}

/// Message severity levels
#[derive(Debug, Clone)]
pub enum MessageSeverity {
    /// Information message
    Info,
    /// Warning message
    Warning,
    /// Error message
    Error,
    /// Debug message
    Debug,
}

/// Execution record
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Operation ID
    pub operation_id: OperationId,
    /// Operation type
    pub operation_type: OperationType,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Execution duration
    pub duration: Option<Duration>,
    /// Result status
    pub result: OperationResult,
    /// Performance metrics
    pub metrics: ExecutionMetrics,
    /// Target devices
    pub target_devices: Vec<DeviceId>,
    /// Resource usage
    pub resource_usage: ResourceUsageRecord,
}

/// Operation result
#[derive(Debug, Clone)]
pub enum OperationResult {
    /// Operation successful
    Success { data: Vec<u8> },
    /// Operation failed
    Failure { error: String },
    /// Operation partial success
    PartialSuccess { completed: usize, failed: usize },
    /// Operation cancelled
    Cancelled { reason: String },
    /// Operation timed out
    TimedOut,
}

/// Execution metrics
#[derive(Debug, Clone)]
pub struct ExecutionMetrics {
    /// CPU usage
    pub cpu_usage: f64,
    /// Memory usage
    pub memory_usage: f64,
    /// Network I/O
    pub network_io: NetworkIOMetrics,
    /// Synchronization overhead
    pub sync_overhead: Duration,
    /// Quality metrics
    pub quality_metrics: OperationQualityMetrics,
}

/// Network I/O metrics
#[derive(Debug, Clone)]
pub struct NetworkIOMetrics {
    /// Bytes sent
    pub bytes_sent: u64,
    /// Bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Network latency
    pub network_latency: Duration,
}

/// Operation quality metrics
#[derive(Debug, Clone)]
pub struct OperationQualityMetrics {
    /// Success rate
    pub success_rate: f64,
    /// Latency
    pub latency: Duration,
    /// Throughput
    pub throughput: f64,
    /// Consistency achieved
    pub consistency_achieved: bool,
    /// Error count
    pub error_count: usize,
}

/// Resource usage record
#[derive(Debug, Clone)]
pub struct ResourceUsageRecord {
    /// Peak CPU usage
    pub peak_cpu: f64,
    /// Peak memory usage
    pub peak_memory: f64,
    /// Total network bytes
    pub total_network_bytes: u64,
    /// Resource efficiency
    pub resource_efficiency: f64,
}

/// Resource manager for coordination
#[derive(Debug)]
pub struct ResourceManager {
    /// Resource configuration
    pub config: ResourceConfig,
    /// Available resources
    pub available_resources: ResourcePool,
    /// Allocated resources
    pub allocated_resources: HashMap<OperationId, AllocatedResources>,
    /// Resource usage statistics
    pub usage_statistics: ResourceUsageStatistics,
    /// Resource reservations
    pub reservations: HashMap<OperationId, ResourceReservation>,
    /// Resource policies
    pub policies: ResourcePolicies,
}

/// Resource reservation
#[derive(Debug, Clone)]
pub struct ResourceReservation {
    /// Operation ID
    pub operation_id: OperationId,
    /// Reserved resources
    pub resources: ResourceRequirements,
    /// Reservation time
    pub reserved_at: Instant,
    /// Expiration time
    pub expires_at: Instant,
    /// Reservation priority
    pub priority: u8,
}

/// Resource policies
#[derive(Debug, Clone)]
pub struct ResourcePolicies {
    /// Allocation policies
    pub allocation_policies: Vec<AllocationPolicy>,
    /// Reclamation policies
    pub reclamation_policies: Vec<ReclamationPolicy>,
    /// Priority policies
    pub priority_policies: Vec<PriorityPolicy>,
}

/// Allocation policy
#[derive(Debug, Clone)]
pub struct AllocationPolicy {
    /// Policy name
    pub name: String,
    /// Policy type
    pub policy_type: AllocationPolicyType,
    /// Policy parameters
    pub parameters: HashMap<String, f64>,
    /// Policy conditions
    pub conditions: Vec<String>,
}

/// Allocation policy types
#[derive(Debug, Clone)]
pub enum AllocationPolicyType {
    /// First-fit allocation
    FirstFit,
    /// Best-fit allocation
    BestFit,
    /// Priority-based allocation
    PriorityBased,
    /// Load-balanced allocation
    LoadBalanced,
    /// Custom allocation policy
    Custom { policy: String },
}

/// Reclamation policy
#[derive(Debug, Clone)]
pub struct ReclamationPolicy {
    /// Policy name
    pub name: String,
    /// Reclamation strategy
    pub strategy: ReclamationStrategy,
    /// Trigger conditions
    pub triggers: Vec<ReclamationTrigger>,
    /// Grace period
    pub grace_period: Duration,
}

/// Reclamation strategies
#[derive(Debug, Clone)]
pub enum ReclamationStrategy {
    /// Preempt lowest priority
    PreemptLowestPriority,
    /// Least recently used
    LeastRecentlyUsed,
    /// Least frequently used
    LeastFrequentlyUsed,
    /// Custom strategy
    Custom { strategy: String },
}

/// Reclamation triggers
#[derive(Debug, Clone)]
pub enum ReclamationTrigger {
    /// Resource utilization threshold
    UtilizationThreshold { threshold: f64 },
    /// Time-based trigger
    TimeBased { interval: Duration },
    /// Demand-based trigger
    DemandBased { demand_threshold: f64 },
    /// Custom trigger
    Custom { trigger: String },
}

/// Priority policy
#[derive(Debug, Clone)]
pub struct PriorityPolicy {
    /// Policy name
    pub name: String,
    /// Priority calculation
    pub calculation: PriorityCalculation,
    /// Aging factor
    pub aging_factor: f64,
    /// Boost conditions
    pub boost_conditions: Vec<String>,
}

/// Priority calculation methods
#[derive(Debug, Clone)]
pub enum PriorityCalculation {
    /// Static priority
    Static,
    /// Dynamic priority
    Dynamic { factors: Vec<String> },
    /// Weighted priority
    Weighted { weights: HashMap<String, f64> },
    /// Custom calculation
    Custom { calculation: String },
}

/// Scheduler state
#[derive(Debug, Clone)]
pub struct SchedulerState {
    /// Scheduler status
    pub status: SchedulerStatus,
    /// Active operation count
    pub active_operations: usize,
    /// Queued operation count
    pub queued_operations: usize,
    /// Total operations processed
    pub total_operations: usize,
    /// Scheduler statistics
    pub statistics: SchedulerStatistics,
    /// Last operation time
    pub last_operation_time: Option<Instant>,
}

/// Scheduler status
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerStatus {
    /// Scheduler is running
    Running,
    /// Scheduler is paused
    Paused,
    /// Scheduler is stopped
    Stopped,
    /// Scheduler is in maintenance mode
    Maintenance,
    /// Scheduler has encountered an error
    Error { error: String },
}

/// Scheduler statistics
#[derive(Debug, Clone)]
pub struct SchedulerStatistics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Average operation latency
    pub avg_operation_latency: Duration,
    /// Success rate
    pub success_rate: f64,
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
    /// Queue length statistics
    pub queue_stats: QueueStatistics,
}

/// Queue statistics
#[derive(Debug, Clone)]
pub struct QueueStatistics {
    /// Average queue length
    pub avg_queue_length: f64,
    /// Maximum queue length
    pub max_queue_length: usize,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Maximum wait time
    pub max_wait_time: Duration,
}

// Implementations

impl CoordinationScheduler {
    /// Create new coordination scheduler
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: SchedulerConfig::default(),
            scheduled_operations: HashMap::new(),
            operation_queue: VecDeque::new(),
            execution_history: Vec::new(),
            resource_manager: ResourceManager::new()?,
            next_operation_id: 1,
            active_executions: HashMap::new(),
            scheduler_state: SchedulerState::new(),
        })
    }

    /// Start scheduler
    pub fn start(&mut self) -> Result<()> {
        self.scheduler_state.status = SchedulerStatus::Running;
        Ok(())
    }

    /// Stop scheduler
    pub fn stop(&mut self) -> Result<()> {
        self.scheduler_state.status = SchedulerStatus::Stopped;

        // Cancel all active operations
        self.cancel_all_operations()?;

        Ok(())
    }

    /// Schedule operation
    pub fn schedule_operation(&mut self, operation: ScheduledOperation) -> Result<OperationId> {
        let operation_id = operation.id;

        // Validate operation
        self.validate_operation(&operation)?;

        // Add to scheduled operations
        self.scheduled_operations.insert(operation_id, operation);

        // Queue for execution
        self.queue_operation(operation_id)?;

        Ok(operation_id)
    }

    /// Create new operation
    pub fn create_operation(&mut self, operation_type: OperationType, target_devices: Vec<DeviceId>, parameters: OperationParameters) -> Result<OperationId> {
        let operation_id = OperationId(self.next_operation_id);
        self.next_operation_id += 1;

        let operation = ScheduledOperation {
            id: operation_id,
            operation_type,
            target_devices,
            scheduled_time: Instant::now(),
            parameters,
            status: OperationStatus::Scheduled,
            created_at: Instant::now(),
            priority: 5, // Default priority
            dependencies: Vec::new(),
        };

        self.schedule_operation(operation)
    }

    /// Queue operation for execution
    fn queue_operation(&mut self, operation_id: OperationId) -> Result<()> {
        if let Some(operation) = self.scheduled_operations.get_mut(&operation_id) {
            let queued_operation = QueuedOperation {
                operation_id,
                queued_at: Instant::now(),
                estimated_duration: operation.parameters.timeout,
                dependencies: operation.dependencies.clone(),
                resource_requirements: operation.parameters.resource_requirements.clone(),
                priority: operation.priority,
            };

            // Insert based on priority
            self.insert_by_priority(queued_operation);

            operation.status = OperationStatus::Queued;
            self.scheduler_state.queued_operations += 1;
        }

        Ok(())
    }

    /// Insert operation in queue based on priority
    fn insert_by_priority(&mut self, operation: QueuedOperation) {
        let mut insertion_index = self.operation_queue.len();

        for (index, queued_op) in self.operation_queue.iter().enumerate() {
            if operation.priority > queued_op.priority {
                insertion_index = index;
                break;
            }
        }

        self.operation_queue.insert(insertion_index, operation);
    }

    /// Execute scheduled operations
    pub fn execute_operations(&mut self) -> Result<()> {
        while let Some(queued_operation) = self.operation_queue.pop_front() {
            if self.can_execute_operation(&queued_operation)? {
                self.execute_operation(queued_operation.operation_id)?;
            } else {
                // Put back in queue if can't execute
                self.operation_queue.push_front(queued_operation);
                break;
            }
        }

        // Update active executions
        self.update_active_executions()?;

        Ok(())
    }

    /// Check if operation can be executed
    fn can_execute_operation(&self, operation: &QueuedOperation) -> Result<bool> {
        // Check dependencies
        if !self.dependencies_satisfied(&operation.dependencies) {
            return Ok(false);
        }

        // Check resource availability
        if let Some(requirements) = &operation.resource_requirements {
            if !self.resource_manager.can_allocate(requirements) {
                return Ok(false);
            }
        }

        // Check scheduler limits
        if self.scheduler_state.active_operations >= self.config.max_concurrent_operations {
            return Ok(false);
        }

        Ok(true)
    }

    /// Check if dependencies are satisfied
    fn dependencies_satisfied(&self, dependencies: &[OperationId]) -> bool {
        dependencies.iter().all(|dep_id| {
            self.scheduled_operations.get(dep_id)
                .map(|op| op.status == OperationStatus::Completed)
                .unwrap_or(false)
        })
    }

    /// Execute operation
    fn execute_operation(&mut self, operation_id: OperationId) -> Result<()> {
        if let Some(operation) = self.scheduled_operations.get_mut(&operation_id) {
            // Allocate resources
            let allocated_resources = if let Some(requirements) = &operation.parameters.resource_requirements {
                self.resource_manager.allocate_resources(operation_id, requirements.clone())?
            } else {
                AllocatedResources::default()
            };

            // Create execution context
            let context = ExecutionContext {
                environment: HashMap::new(),
                target_devices: operation.target_devices.clone(),
                metadata: HashMap::new(),
                constraints: ExecutionConstraints::default(),
            };

            // Create active execution
            let active_execution = ActiveExecution {
                operation_id,
                start_time: Instant::now(),
                allocated_resources,
                context,
                progress: ExecutionProgress::new(),
            };

            // Update operation status
            operation.status = OperationStatus::Running;

            // Add to active executions
            self.active_executions.insert(operation_id, active_execution);
            self.scheduler_state.active_operations += 1;
            self.scheduler_state.queued_operations -= 1;

            // Start execution (in real implementation, this would spawn a task)
            self.start_operation_execution(operation_id)?;
        }

        Ok(())
    }

    /// Start operation execution
    fn start_operation_execution(&mut self, operation_id: OperationId) -> Result<()> {
        // In real implementation, this would start the actual operation
        // For now, we'll simulate completion
        if let Some(operation) = self.scheduled_operations.get_mut(&operation_id) {
            operation.status = OperationStatus::Completed;
        }

        self.complete_operation(operation_id, OperationResult::Success { data: vec![] })?;

        Ok(())
    }

    /// Update active executions
    fn update_active_executions(&mut self) -> Result<()> {
        let mut completed_operations = Vec::new();

        for (operation_id, execution) in &mut self.active_executions {
            // Check for timeout
            if execution.start_time.elapsed() > Duration::from_secs(300) { // 5 minutes default timeout
                completed_operations.push((*operation_id, OperationResult::TimedOut));
            }

            // Update progress
            execution.progress.last_update = Instant::now();
        }

        // Complete timed out operations
        for (operation_id, result) in completed_operations {
            self.complete_operation(operation_id, result)?;
        }

        Ok(())
    }

    /// Complete operation
    fn complete_operation(&mut self, operation_id: OperationId, result: OperationResult) -> Result<()> {
        if let Some(execution) = self.active_executions.remove(&operation_id) {
            // Create execution record
            let record = ExecutionRecord {
                operation_id,
                operation_type: self.scheduled_operations.get(&operation_id)
                    .map(|op| op.operation_type.clone())
                    .unwrap_or(OperationType::Custom { operation: "unknown".to_string() }),
                start_time: execution.start_time,
                end_time: Some(Instant::now()),
                duration: Some(execution.start_time.elapsed()),
                result,
                metrics: ExecutionMetrics::default(),
                target_devices: execution.context.target_devices,
                resource_usage: ResourceUsageRecord::default(),
            };

            // Add to history
            self.execution_history.push(record);

            // Release resources
            self.resource_manager.release_resources(operation_id)?;

            // Update scheduler state
            self.scheduler_state.active_operations -= 1;
            self.scheduler_state.total_operations += 1;

            // Update operation status
            if let Some(operation) = self.scheduled_operations.get_mut(&operation_id) {
                operation.status = match &self.execution_history.last().unwrap().result {
                    OperationResult::Success { .. } => OperationStatus::Completed,
                    OperationResult::Failure { error } => OperationStatus::Failed { reason: error.clone() },
                    OperationResult::TimedOut => OperationStatus::TimedOut,
                    OperationResult::Cancelled { reason } => OperationStatus::Cancelled,
                    OperationResult::PartialSuccess { .. } => OperationStatus::Completed,
                };
            }
        }

        Ok(())
    }

    /// Cancel operation
    pub fn cancel_operation(&mut self, operation_id: OperationId) -> Result<()> {
        // Remove from queue if queued
        self.operation_queue.retain(|op| op.operation_id != operation_id);

        // Cancel if active
        if self.active_executions.contains_key(&operation_id) {
            self.complete_operation(operation_id, OperationResult::Cancelled { reason: "Manual cancellation".to_string() })?;
        }

        // Update operation status
        if let Some(operation) = self.scheduled_operations.get_mut(&operation_id) {
            operation.status = OperationStatus::Cancelled;
        }

        Ok(())
    }

    /// Cancel all operations
    fn cancel_all_operations(&mut self) -> Result<()> {
        let operation_ids: Vec<OperationId> = self.active_executions.keys().cloned().collect();

        for operation_id in operation_ids {
            self.cancel_operation(operation_id)?;
        }

        self.operation_queue.clear();
        self.scheduler_state.queued_operations = 0;

        Ok(())
    }

    /// Validate operation
    fn validate_operation(&self, operation: &ScheduledOperation) -> Result<()> {
        // Check if devices exist
        if operation.target_devices.is_empty() {
            return Err(OptimError::InvalidInput("Target devices cannot be empty".to_string()));
        }

        // Check timeout
        if operation.parameters.timeout.as_secs() == 0 {
            return Err(OptimError::InvalidInput("Operation timeout cannot be zero".to_string()));
        }

        Ok(())
    }

    /// Calculate scheduler efficiency
    pub fn calculate_efficiency(&self) -> f64 {
        if self.execution_history.is_empty() {
            return 0.0;
        }

        let successful_operations = self.execution_history.iter()
            .filter(|record| matches!(record.result, OperationResult::Success { .. }))
            .count();

        successful_operations as f64 / self.execution_history.len() as f64
    }

    /// Get operation status
    pub fn get_operation_status(&self, operation_id: OperationId) -> Option<&OperationStatus> {
        self.scheduled_operations.get(&operation_id).map(|op| &op.status)
    }

    /// Get scheduler statistics
    pub fn get_statistics(&self) -> &SchedulerStatistics {
        &self.scheduler_state.statistics
    }

    /// Update scheduler statistics
    pub fn update_statistics(&mut self) {
        let total_ops = self.execution_history.len();
        if total_ops == 0 {
            return;
        }

        let successful_ops = self.execution_history.iter()
            .filter(|record| matches!(record.result, OperationResult::Success { .. }))
            .count();

        self.scheduler_state.statistics.success_rate = successful_ops as f64 / total_ops as f64;

        // Calculate average latency
        let total_duration: Duration = self.execution_history.iter()
            .filter_map(|record| record.duration)
            .sum();

        self.scheduler_state.statistics.avg_operation_latency =
            total_duration / total_ops.max(1) as u32;

        // Update queue statistics
        self.scheduler_state.statistics.queue_stats.avg_queue_length =
            self.operation_queue.len() as f64;
    }
}

impl ResourceManager {
    /// Create new resource manager
    pub fn new() -> Result<Self> {
        Ok(Self {
            config: ResourceConfig::default(),
            available_resources: ResourcePool::default(),
            allocated_resources: HashMap::new(),
            usage_statistics: ResourceUsageStatistics::default(),
            reservations: HashMap::new(),
            policies: ResourcePolicies::default(),
        })
    }

    /// Check if resources can be allocated
    pub fn can_allocate(&self, requirements: &ResourceRequirements) -> bool {
        self.available_resources.cpu_cores as f64 >= requirements.cpu &&
        self.available_resources.memory as f64 >= requirements.memory &&
        self.available_resources.network_bandwidth as f64 >= requirements.network
    }

    /// Allocate resources for operation
    pub fn allocate_resources(&mut self, operation_id: OperationId, requirements: ResourceRequirements) -> Result<AllocatedResources> {
        if !self.can_allocate(&requirements) {
            return Err(OptimError::ResourceUnavailable("Insufficient resources".to_string()));
        }

        let allocation = AllocatedResources {
            cpu_cores: requirements.cpu as usize,
            memory: requirements.memory as usize,
            network_bandwidth: requirements.network as u64,
            allocated_at: Instant::now(),
            custom_allocations: HashMap::new(),
        };

        // Update available resources
        self.available_resources.cpu_cores =
            (self.available_resources.cpu_cores as f64 - requirements.cpu) as usize;
        self.available_resources.memory =
            (self.available_resources.memory as f64 - requirements.memory) as usize;
        self.available_resources.network_bandwidth =
            (self.available_resources.network_bandwidth as f64 - requirements.network) as u64;

        self.allocated_resources.insert(operation_id, allocation.clone());
        Ok(allocation)
    }

    /// Release resources
    pub fn release_resources(&mut self, operation_id: OperationId) -> Result<()> {
        if let Some(allocation) = self.allocated_resources.remove(&operation_id) {
            // Return resources to available pool
            self.available_resources.cpu_cores += allocation.cpu_cores;
            self.available_resources.memory += allocation.memory;
            self.available_resources.network_bandwidth += allocation.network_bandwidth;
        }

        Ok(())
    }
}

impl SchedulerState {
    /// Create new scheduler state
    pub fn new() -> Self {
        Self {
            status: SchedulerStatus::Stopped,
            active_operations: 0,
            queued_operations: 0,
            total_operations: 0,
            statistics: SchedulerStatistics::default(),
            last_operation_time: None,
        }
    }
}

impl ExecutionProgress {
    /// Create new execution progress
    pub fn new() -> Self {
        Self {
            progress: 0.0,
            current_phase: "Initializing".to_string(),
            completed_phases: Vec::new(),
            messages: Vec::new(),
            last_update: Instant::now(),
        }
    }

    /// Update progress
    pub fn update_progress(&mut self, progress: f64, phase: String) {
        self.progress = progress.clamp(0.0, 1.0);

        if phase != self.current_phase {
            self.completed_phases.push(self.current_phase.clone());
            self.current_phase = phase;
        }

        self.last_update = Instant::now();
    }

    /// Add progress message
    pub fn add_message(&mut self, message: String, severity: MessageSeverity) {
        self.messages.push(ProgressMessage {
            timestamp: Instant::now(),
            message,
            severity,
        });
    }
}

// Default implementations
impl Default for SchedulerStatistics {
    fn default() -> Self {
        Self {
            ops_per_second: 0.0,
            avg_operation_latency: Duration::from_millis(0),
            success_rate: 0.0,
            resource_efficiency: 0.0,
            queue_stats: QueueStatistics::default(),
        }
    }
}

impl Default for QueueStatistics {
    fn default() -> Self {
        Self {
            avg_queue_length: 0.0,
            max_queue_length: 0,
            avg_wait_time: Duration::from_millis(0),
            max_wait_time: Duration::from_millis(0),
        }
    }
}

impl Default for ExecutionMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            network_io: NetworkIOMetrics::default(),
            sync_overhead: Duration::from_millis(0),
            quality_metrics: OperationQualityMetrics::default(),
        }
    }
}

impl Default for NetworkIOMetrics {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            network_latency: Duration::from_millis(0),
        }
    }
}

impl Default for OperationQualityMetrics {
    fn default() -> Self {
        Self {
            success_rate: 0.0,
            latency: Duration::from_millis(0),
            throughput: 0.0,
            consistency_achieved: false,
            error_count: 0,
        }
    }
}

impl Default for ResourceUsageRecord {
    fn default() -> Self {
        Self {
            peak_cpu: 0.0,
            peak_memory: 0.0,
            total_network_bytes: 0,
            resource_efficiency: 0.0,
        }
    }
}

impl Default for AllocatedResources {
    fn default() -> Self {
        Self {
            cpu_cores: 0,
            memory: 0,
            network_bandwidth: 0,
            allocated_at: Instant::now(),
            custom_allocations: HashMap::new(),
        }
    }
}

impl Default for ResourcePolicies {
    fn default() -> Self {
        Self {
            allocation_policies: Vec::new(),
            reclamation_policies: Vec::new(),
            priority_policies: Vec::new(),
        }
    }
}

impl Default for ExecutionConstraints {
    fn default() -> Self {
        Self {
            max_execution_time: Duration::from_secs(300), // 5 minutes
            resource_limits: ResourceLimits::default(),
            failure_tolerance: FailureTolerance::default(),
            rollback_policy: RollbackPolicy::None,
        }
    }
}

impl Default for FailureTolerance {
    fn default() -> Self {
        Self {
            max_failures: 3,
            failure_threshold: 0.1, // 10%
            retry_on_failure: true,
            continue_on_partial_failure: false,
        }
    }
}

/// Scheduler utilities
pub mod utils {
    use super::*;

    /// Create test scheduler
    pub fn create_test_scheduler() -> Result<CoordinationScheduler> {
        let mut scheduler = CoordinationScheduler::new()?;
        scheduler.config.max_concurrent_operations = 5;
        scheduler.config.operation_timeout = Duration::from_secs(30);
        Ok(scheduler)
    }

    /// Create test operation
    pub fn create_test_operation(device_ids: Vec<DeviceId>) -> ScheduledOperation {
        ScheduledOperation {
            id: OperationId(1),
            operation_type: OperationType::GlobalSync,
            target_devices: device_ids,
            scheduled_time: Instant::now(),
            parameters: OperationParameters {
                timeout: Duration::from_secs(30),
                priority: 5,
                retry_settings: RetrySettings {
                    max_attempts: 3,
                    interval: Duration::from_secs(5),
                    backoff: BackoffStrategy::Fixed,
                    conditions: vec![RetryCondition::OnTimeout],
                },
                custom_params: HashMap::new(),
                resource_requirements: None,
                quality_requirements: None,
            },
            status: OperationStatus::Scheduled,
            created_at: Instant::now(),
            priority: 5,
            dependencies: Vec::new(),
        }
    }

    /// Calculate operation efficiency
    pub fn calculate_operation_efficiency(records: &[ExecutionRecord]) -> f64 {
        if records.is_empty() {
            return 0.0;
        }

        let successful = records.iter()
            .filter(|r| matches!(r.result, OperationResult::Success { .. }))
            .count();

        successful as f64 / records.len() as f64
    }
}