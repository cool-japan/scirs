//! State management for TPU pod coordination synchronization
//!
//! This module provides structures for managing global synchronization state,
//! device states, and quality metrics across the TPU pod coordination system.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::tpu::tpu_backend::DeviceId;

/// Global synchronization state
#[derive(Debug)]
pub struct GlobalSynchronizationState {
    /// Overall synchronization status
    pub status: GlobalSyncStatus,
    /// Participating devices
    pub participants: HashSet<DeviceId>,
    /// Synchronization quality metrics
    pub quality_metrics: GlobalQualityMetrics,
    /// Last global synchronization
    pub last_global_sync: Option<Instant>,
    /// Synchronization epochs
    pub current_epoch: u64,
    /// Device states
    pub device_states: HashMap<DeviceId, DeviceSyncState>,
    /// Synchronization barriers
    pub active_sync_barriers: HashMap<String, GlobalBarrier>,
}

/// Global synchronization status
#[derive(Debug, Clone, PartialEq)]
pub enum GlobalSyncStatus {
    /// System is not synchronized
    NotSynchronized,
    /// System is synchronizing
    Synchronizing { progress: f64 },
    /// System is synchronized
    Synchronized { quality: f64 },
    /// Synchronization is degraded
    Degraded { reason: String },
    /// Synchronization has failed
    Failed { error: String },
}

/// Global quality metrics for synchronization
#[derive(Debug, Clone)]
pub struct GlobalQualityMetrics {
    /// Overall synchronization quality
    pub overall_quality: f64,
    /// Clock synchronization quality
    pub clock_quality: f64,
    /// Event synchronization quality
    pub event_quality: f64,
    /// Barrier synchronization quality
    pub barrier_quality: f64,
    /// Consensus quality
    pub consensus_quality: f64,
    /// Deadlock prevention quality
    pub deadlock_prevention_quality: f64,
    /// Coordination efficiency
    pub coordination_efficiency: f64,
}

/// Device synchronization state
#[derive(Debug, Clone)]
pub struct DeviceSyncState {
    /// Device ID
    pub device_id: DeviceId,
    /// Synchronization status
    pub status: DeviceSyncStatus,
    /// Last synchronization time
    pub last_sync: Option<Instant>,
    /// Synchronization quality
    pub quality: f64,
    /// Participation count
    pub participation_count: usize,
    /// Performance metrics
    pub performance: DevicePerformanceMetrics,
}

/// Device synchronization status
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceSyncStatus {
    /// Device is synchronized
    Synchronized,
    /// Device is synchronizing
    Synchronizing,
    /// Device synchronization failed
    Failed { reason: String },
    /// Device is offline
    Offline,
    /// Device status unknown
    Unknown,
}

/// Device performance metrics
#[derive(Debug, Clone)]
pub struct DevicePerformanceMetrics {
    /// Synchronization latency
    pub sync_latency: Duration,
    /// Throughput
    pub throughput: f64,
    /// Success rate
    pub success_rate: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// CPU utilization
    pub cpu: f64,
    /// Memory utilization
    pub memory: f64,
    /// Network bandwidth utilization
    pub network_bandwidth: f64,
    /// Storage utilization
    pub storage: f64,
}

/// Global barrier for synchronization
#[derive(Debug, Clone)]
pub struct GlobalBarrier {
    /// Barrier ID
    pub id: String,
    /// Expected participants
    pub expected_participants: HashSet<DeviceId>,
    /// Arrived participants
    pub arrived_participants: HashSet<DeviceId>,
    /// Barrier timeout
    pub timeout: Duration,
    /// Creation time
    pub created_at: Instant,
    /// Completion time
    pub completed_at: Option<Instant>,
}

/// Resource pool
#[derive(Debug, Clone)]
pub struct ResourcePool {
    /// Available CPU cores
    pub cpu_cores: usize,
    /// Available memory
    pub memory: usize,
    /// Available network bandwidth
    pub network_bandwidth: u64,
    /// Custom resources
    pub custom_resources: HashMap<String, u64>,
}

/// Allocated resources
#[derive(Debug, Clone)]
pub struct AllocatedResources {
    /// Allocated CPU cores
    pub cpu_cores: usize,
    /// Allocated memory
    pub memory: usize,
    /// Allocated network bandwidth
    pub network_bandwidth: u64,
    /// Allocation timestamp
    pub allocated_at: Instant,
    /// Custom allocations
    pub custom_allocations: HashMap<String, u64>,
}

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceUsageStatistics {
    /// CPU usage statistics
    pub cpu_usage: UsageStatistics,
    /// Memory usage statistics
    pub memory_usage: UsageStatistics,
    /// Network usage statistics
    pub network_usage: UsageStatistics,
    /// Overall efficiency
    pub overall_efficiency: f64,
}

/// Usage statistics
#[derive(Debug, Clone)]
pub struct UsageStatistics {
    /// Current usage
    pub current_usage: f64,
    /// Average usage
    pub average_usage: f64,
    /// Peak usage
    pub peak_usage: f64,
    /// Usage variance
    pub variance: f64,
    /// Usage history
    pub history: Vec<(Instant, f64)>,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Latency metrics
    pub latency: LatencyMetrics,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Error rate metrics
    pub error_rate: ErrorRateMetrics,
    /// Synchronization metrics
    pub sync_metrics: SyncQualityMetrics,
}

/// Latency metrics
#[derive(Debug, Clone)]
pub struct LatencyMetrics {
    /// Average latency
    pub average: Duration,
    /// P50 latency
    pub p50: Duration,
    /// P90 latency
    pub p90: Duration,
    /// P99 latency
    pub p99: Duration,
    /// Maximum latency
    pub max: Duration,
}

/// Throughput metrics
#[derive(Debug, Clone)]
pub struct ThroughputMetrics {
    /// Operations per second
    pub ops_per_second: f64,
    /// Bytes per second
    pub bytes_per_second: f64,
    /// Peak throughput
    pub peak_throughput: f64,
    /// Sustained throughput
    pub sustained_throughput: f64,
}

/// Error rate metrics
#[derive(Debug, Clone)]
pub struct ErrorRateMetrics {
    /// Total errors
    pub total_errors: usize,
    /// Error rate
    pub error_rate: f64,
    /// Error types
    pub error_types: HashMap<String, usize>,
    /// Critical errors
    pub critical_errors: usize,
}

/// Synchronization quality metrics
#[derive(Debug, Clone)]
pub struct SyncQualityMetrics {
    /// Synchronization accuracy
    pub accuracy: f64,
    /// Consistency level
    pub consistency: f64,
    /// Fault tolerance
    pub fault_tolerance: f64,
    /// Recovery time
    pub recovery_time: Duration,
}

/// Optimization state
#[derive(Debug, Clone)]
pub struct OptimizationState {
    /// Current configuration
    pub current_config: HashMap<String, f64>,
    /// Best configuration found
    pub best_config: Option<HashMap<String, f64>>,
    /// Optimization iteration
    pub iteration: usize,
    /// Convergence status
    pub convergence_status: ConvergenceStatus,
}

/// Convergence status
#[derive(Debug, Clone, PartialEq)]
pub enum ConvergenceStatus {
    /// Not converged
    NotConverged,
    /// Converged
    Converged,
    /// Diverged
    Diverged,
    /// Stalled
    Stalled,
}

/// Resource requirements
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// CPU requirements
    pub cpu: f64,
    /// Memory requirements
    pub memory: f64,
    /// Network requirements
    pub network: f64,
    /// Custom requirements
    pub custom: HashMap<String, f64>,
}

/// Synchronization epoch information
#[derive(Debug, Clone)]
pub struct SynchronizationEpoch {
    /// Epoch number
    pub epoch: u64,
    /// Start time
    pub start_time: Instant,
    /// End time
    pub end_time: Option<Instant>,
    /// Participating devices
    pub participants: HashSet<DeviceId>,
    /// Epoch quality
    pub quality: f64,
    /// Synchronization events
    pub events: Vec<SyncEvent>,
}

/// Synchronization event
#[derive(Debug, Clone)]
pub struct SyncEvent {
    /// Event ID
    pub id: u64,
    /// Event type
    pub event_type: SyncEventType,
    /// Timestamp
    pub timestamp: Instant,
    /// Device ID
    pub device_id: Option<DeviceId>,
    /// Event data
    pub data: HashMap<String, String>,
}

/// Synchronization event types
#[derive(Debug, Clone)]
pub enum SyncEventType {
    /// Device joined synchronization
    DeviceJoined,
    /// Device left synchronization
    DeviceLeft,
    /// Synchronization started
    SyncStarted,
    /// Synchronization completed
    SyncCompleted,
    /// Synchronization failed
    SyncFailed,
    /// Barrier created
    BarrierCreated,
    /// Barrier completed
    BarrierCompleted,
    /// Quality threshold exceeded
    QualityThresholdExceeded,
    /// Custom event
    Custom { event_type: String },
}

/// Device capability information
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device ID
    pub device_id: DeviceId,
    /// Supported synchronization modes
    pub supported_sync_modes: Vec<String>,
    /// Maximum throughput
    pub max_throughput: f64,
    /// Minimum latency
    pub min_latency: Duration,
    /// Resource capacity
    pub resource_capacity: ResourceCapacity,
    /// Reliability metrics
    pub reliability: ReliabilityMetrics,
}

/// Resource capacity
#[derive(Debug, Clone)]
pub struct ResourceCapacity {
    /// CPU cores
    pub cpu_cores: usize,
    /// Memory capacity
    pub memory: usize,
    /// Network bandwidth
    pub network_bandwidth: u64,
    /// Storage capacity
    pub storage: usize,
}

/// Reliability metrics
#[derive(Debug, Clone)]
pub struct ReliabilityMetrics {
    /// Uptime percentage
    pub uptime: f64,
    /// Mean time between failures
    pub mtbf: Duration,
    /// Mean time to recovery
    pub mttr: Duration,
    /// Failure rate
    pub failure_rate: f64,
}

/// Synchronization health snapshot
#[derive(Debug, Clone)]
pub struct SyncHealthSnapshot {
    /// Snapshot timestamp
    pub timestamp: Instant,
    /// Global health score
    pub global_health: f64,
    /// Device health scores
    pub device_health: HashMap<DeviceId, f64>,
    /// Component health scores
    pub component_health: HashMap<String, f64>,
    /// Active issues
    pub active_issues: Vec<HealthIssue>,
    /// Performance indicators
    pub performance_indicators: PerformanceIndicators,
}

/// Health issue
#[derive(Debug, Clone)]
pub struct HealthIssue {
    /// Issue ID
    pub id: String,
    /// Issue severity
    pub severity: IssueSeverity,
    /// Issue description
    pub description: String,
    /// Affected components
    pub affected_components: Vec<String>,
    /// Detection time
    pub detected_at: Instant,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum IssueSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Performance indicators
#[derive(Debug, Clone)]
pub struct PerformanceIndicators {
    /// Average synchronization time
    pub avg_sync_time: Duration,
    /// Synchronization success rate
    pub sync_success_rate: f64,
    /// Resource utilization efficiency
    pub resource_efficiency: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Error rate trend
    pub error_rate_trend: TrendDirection,
    /// Latency trend
    pub latency_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, PartialEq)]
pub enum TrendDirection {
    /// Improving trend
    Improving,
    /// Stable trend
    Stable,
    /// Degrading trend
    Degrading,
    /// Unknown trend
    Unknown,
}

// Implementations

impl GlobalSynchronizationState {
    /// Create new global synchronization state
    pub fn new() -> Self {
        Self {
            status: GlobalSyncStatus::NotSynchronized,
            participants: HashSet::new(),
            quality_metrics: GlobalQualityMetrics::default(),
            last_global_sync: None,
            current_epoch: 0,
            device_states: HashMap::new(),
            active_sync_barriers: HashMap::new(),
        }
    }

    /// Check if system is synchronized
    pub fn is_synchronized(&self) -> bool {
        matches!(self.status, GlobalSyncStatus::Synchronized { .. })
    }

    /// Get synchronization quality
    pub fn get_quality(&self) -> f64 {
        self.quality_metrics.overall_quality
    }

    /// Update quality metrics
    pub fn update_quality_metrics(&mut self, metrics: GlobalQualityMetrics) {
        self.quality_metrics = metrics;
    }

    /// Add device
    pub fn add_device(&mut self, device_id: DeviceId) {
        self.participants.insert(device_id);

        let device_state = DeviceSyncState {
            device_id,
            status: DeviceSyncStatus::Unknown,
            last_sync: None,
            quality: 0.0,
            participation_count: 0,
            performance: DevicePerformanceMetrics::default(),
        };

        self.device_states.insert(device_id, device_state);
    }

    /// Remove device
    pub fn remove_device(&mut self, device_id: DeviceId) {
        self.participants.remove(&device_id);
        self.device_states.remove(&device_id);
    }

    /// Get synchronized device count
    pub fn get_synchronized_count(&self) -> usize {
        self.device_states.values()
            .filter(|state| state.status == DeviceSyncStatus::Synchronized)
            .count()
    }

    /// Get failed device count
    pub fn get_failed_count(&self) -> usize {
        self.device_states.values()
            .filter(|state| matches!(state.status, DeviceSyncStatus::Failed { .. }))
            .count()
    }

    /// Get synchronization progress
    pub fn get_sync_progress(&self) -> f64 {
        if self.participants.is_empty() {
            return 1.0;
        }

        let synchronized_count = self.get_synchronized_count();
        synchronized_count as f64 / self.participants.len() as f64
    }

    /// Check if all devices are synchronized
    pub fn all_devices_synchronized(&self) -> bool {
        self.device_states.values()
            .all(|state| state.status == DeviceSyncStatus::Synchronized)
    }

    /// Get average device quality
    pub fn get_average_device_quality(&self) -> f64 {
        if self.device_states.is_empty() {
            return 0.0;
        }

        let total_quality: f64 = self.device_states.values()
            .map(|state| state.quality)
            .sum();

        total_quality / self.device_states.len() as f64
    }
}

impl DeviceSyncState {
    /// Create new device state
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            device_id,
            status: DeviceSyncStatus::Unknown,
            last_sync: None,
            quality: 0.0,
            participation_count: 0,
            performance: DevicePerformanceMetrics::default(),
        }
    }

    /// Check if device is healthy
    pub fn is_healthy(&self) -> bool {
        matches!(self.status, DeviceSyncStatus::Synchronized | DeviceSyncStatus::Synchronizing)
            && self.quality > 0.7
            && self.performance.success_rate > 0.9
    }

    /// Update performance metrics
    pub fn update_performance(&mut self, latency: Duration, throughput: f64, success_rate: f64) {
        self.performance.sync_latency = latency;
        self.performance.throughput = throughput;
        self.performance.success_rate = success_rate;
    }

    /// Mark as synchronized
    pub fn mark_synchronized(&mut self) {
        self.status = DeviceSyncStatus::Synchronized;
        self.last_sync = Some(Instant::now());
        self.participation_count += 1;
    }

    /// Mark as failed
    pub fn mark_failed(&mut self, reason: String) {
        self.status = DeviceSyncStatus::Failed { reason };
    }

    /// Get time since last sync
    pub fn time_since_last_sync(&self) -> Option<Duration> {
        self.last_sync.map(|last| last.elapsed())
    }
}

impl GlobalBarrier {
    /// Create new global barrier
    pub fn new(id: String, participants: HashSet<DeviceId>, timeout: Duration) -> Self {
        Self {
            id,
            expected_participants: participants,
            arrived_participants: HashSet::new(),
            timeout,
            created_at: Instant::now(),
            completed_at: None,
        }
    }

    /// Check if barrier is complete
    pub fn is_complete(&self) -> bool {
        self.arrived_participants.len() == self.expected_participants.len()
    }

    /// Check if barrier has timed out
    pub fn has_timed_out(&self) -> bool {
        self.created_at.elapsed() > self.timeout
    }

    /// Add participant arrival
    pub fn add_arrival(&mut self, device_id: DeviceId) -> bool {
        if self.expected_participants.contains(&device_id) {
            self.arrived_participants.insert(device_id);

            if self.is_complete() {
                self.completed_at = Some(Instant::now());
                return true;
            }
        }
        false
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f64 {
        if self.expected_participants.is_empty() {
            return 1.0;
        }

        self.arrived_participants.len() as f64 / self.expected_participants.len() as f64
    }

    /// Get waiting devices
    pub fn get_waiting_devices(&self) -> HashSet<DeviceId> {
        self.expected_participants.difference(&self.arrived_participants).cloned().collect()
    }
}

impl SyncHealthSnapshot {
    /// Create new health snapshot
    pub fn new() -> Self {
        Self {
            timestamp: Instant::now(),
            global_health: 0.0,
            device_health: HashMap::new(),
            component_health: HashMap::new(),
            active_issues: Vec::new(),
            performance_indicators: PerformanceIndicators::default(),
        }
    }

    /// Add health issue
    pub fn add_issue(&mut self, issue: HealthIssue) {
        self.active_issues.push(issue);
    }

    /// Get critical issues
    pub fn get_critical_issues(&self) -> Vec<&HealthIssue> {
        self.active_issues.iter()
            .filter(|issue| issue.severity == IssueSeverity::Critical)
            .collect()
    }

    /// Calculate overall health score
    pub fn calculate_overall_health(&mut self) {
        if self.device_health.is_empty() {
            self.global_health = 0.0;
            return;
        }

        let device_avg = self.device_health.values().sum::<f64>() / self.device_health.len() as f64;
        let component_avg = if self.component_health.is_empty() {
            1.0
        } else {
            self.component_health.values().sum::<f64>() / self.component_health.len() as f64
        };

        // Weight device health more heavily
        self.global_health = (device_avg * 0.7) + (component_avg * 0.3);

        // Penalize for critical issues
        let critical_penalty = self.get_critical_issues().len() as f64 * 0.1;
        self.global_health = (self.global_health - critical_penalty).max(0.0);
    }
}

// Default implementations
impl Default for GlobalQualityMetrics {
    fn default() -> Self {
        Self {
            overall_quality: 0.0,
            clock_quality: 0.0,
            event_quality: 0.0,
            barrier_quality: 0.0,
            consensus_quality: 0.0,
            deadlock_prevention_quality: 0.0,
            coordination_efficiency: 0.0,
        }
    }
}

impl Default for DevicePerformanceMetrics {
    fn default() -> Self {
        Self {
            sync_latency: Duration::from_millis(0),
            throughput: 0.0,
            success_rate: 0.0,
            resource_utilization: ResourceUtilization::default(),
        }
    }
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu: 0.0,
            memory: 0.0,
            network_bandwidth: 0.0,
            storage: 0.0,
        }
    }
}

impl Default for ResourcePool {
    fn default() -> Self {
        Self {
            cpu_cores: 8,
            memory: 16 * 1024 * 1024 * 1024, // 16 GB
            network_bandwidth: 10_000_000_000, // 10 Gbps
            custom_resources: HashMap::new(),
        }
    }
}

impl Default for ResourceUsageStatistics {
    fn default() -> Self {
        Self {
            cpu_usage: UsageStatistics::default(),
            memory_usage: UsageStatistics::default(),
            network_usage: UsageStatistics::default(),
            overall_efficiency: 0.0,
        }
    }
}

impl Default for UsageStatistics {
    fn default() -> Self {
        Self {
            current_usage: 0.0,
            average_usage: 0.0,
            peak_usage: 0.0,
            variance: 0.0,
            history: Vec::new(),
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            latency: LatencyMetrics::default(),
            throughput: ThroughputMetrics::default(),
            error_rate: ErrorRateMetrics::default(),
            sync_metrics: SyncQualityMetrics::default(),
        }
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            average: Duration::from_millis(0),
            p50: Duration::from_millis(0),
            p90: Duration::from_millis(0),
            p99: Duration::from_millis(0),
            max: Duration::from_millis(0),
        }
    }
}

impl Default for ThroughputMetrics {
    fn default() -> Self {
        Self {
            ops_per_second: 0.0,
            bytes_per_second: 0.0,
            peak_throughput: 0.0,
            sustained_throughput: 0.0,
        }
    }
}

impl Default for ErrorRateMetrics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            error_rate: 0.0,
            error_types: HashMap::new(),
            critical_errors: 0,
        }
    }
}

impl Default for SyncQualityMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.0,
            consistency: 0.0,
            fault_tolerance: 0.0,
            recovery_time: Duration::from_millis(0),
        }
    }
}

impl Default for OptimizationState {
    fn default() -> Self {
        Self {
            current_config: HashMap::new(),
            best_config: None,
            iteration: 0,
            convergence_status: ConvergenceStatus::NotConverged,
        }
    }
}

impl Default for PerformanceIndicators {
    fn default() -> Self {
        Self {
            avg_sync_time: Duration::from_millis(0),
            sync_success_rate: 0.0,
            resource_efficiency: 0.0,
            network_utilization: 0.0,
            error_rate_trend: TrendDirection::Unknown,
            latency_trend: TrendDirection::Unknown,
        }
    }
}

/// State management utilities
pub mod utils {
    use super::*;

    /// Create health snapshot from global state
    pub fn create_health_snapshot(global_state: &GlobalSynchronizationState) -> SyncHealthSnapshot {
        let mut snapshot = SyncHealthSnapshot::new();

        // Calculate device health scores
        for (device_id, device_state) in &global_state.device_states {
            let health_score = calculate_device_health_score(device_state);
            snapshot.device_health.insert(*device_id, health_score);
        }

        // Calculate component health scores
        snapshot.component_health.insert("clock_sync".to_string(), global_state.quality_metrics.clock_quality);
        snapshot.component_health.insert("event_sync".to_string(), global_state.quality_metrics.event_quality);
        snapshot.component_health.insert("barrier_sync".to_string(), global_state.quality_metrics.barrier_quality);
        snapshot.component_health.insert("consensus".to_string(), global_state.quality_metrics.consensus_quality);

        // Calculate overall health
        snapshot.calculate_overall_health();

        snapshot
    }

    /// Calculate device health score
    pub fn calculate_device_health_score(device_state: &DeviceSyncState) -> f64 {
        let status_score = match device_state.status {
            DeviceSyncStatus::Synchronized => 1.0,
            DeviceSyncStatus::Synchronizing => 0.7,
            DeviceSyncStatus::Failed { .. } => 0.1,
            DeviceSyncStatus::Offline => 0.0,
            DeviceSyncStatus::Unknown => 0.3,
        };

        let quality_score = device_state.quality;
        let performance_score = device_state.performance.success_rate;

        // Weighted average
        (status_score * 0.4) + (quality_score * 0.3) + (performance_score * 0.3)
    }

    /// Check for potential issues
    pub fn detect_potential_issues(global_state: &GlobalSynchronizationState) -> Vec<HealthIssue> {
        let mut issues = Vec::new();

        // Check for low overall quality
        if global_state.quality_metrics.overall_quality < 0.5 {
            issues.push(HealthIssue {
                id: "low_overall_quality".to_string(),
                severity: IssueSeverity::High,
                description: format!("Overall synchronization quality is low: {:.2}", global_state.quality_metrics.overall_quality),
                affected_components: vec!["global_sync".to_string()],
                detected_at: Instant::now(),
                recommendations: vec!["Review component health and performance".to_string()],
            });
        }

        // Check for failed devices
        let failed_count = global_state.get_failed_count();
        if failed_count > 0 {
            issues.push(HealthIssue {
                id: "failed_devices".to_string(),
                severity: if failed_count > global_state.participants.len() / 2 {
                    IssueSeverity::Critical
                } else {
                    IssueSeverity::Medium
                },
                description: format!("{} devices have failed synchronization", failed_count),
                affected_components: vec!["device_coordination".to_string()],
                detected_at: Instant::now(),
                recommendations: vec!["Investigate failed devices and restart synchronization".to_string()],
            });
        }

        // Check for stale synchronization
        if let Some(last_sync) = global_state.last_global_sync {
            if last_sync.elapsed() > Duration::from_secs(300) { // 5 minutes
                issues.push(HealthIssue {
                    id: "stale_synchronization".to_string(),
                    severity: IssueSeverity::Medium,
                    description: "Global synchronization is stale".to_string(),
                    affected_components: vec!["global_sync".to_string()],
                    detected_at: Instant::now(),
                    recommendations: vec!["Trigger fresh synchronization".to_string()],
                });
            }
        }

        issues
    }

    /// Get synchronization summary
    pub fn get_sync_summary(global_state: &GlobalSynchronizationState) -> SyncSummary {
        SyncSummary {
            total_devices: global_state.participants.len(),
            synchronized_devices: global_state.get_synchronized_count(),
            failed_devices: global_state.get_failed_count(),
            current_epoch: global_state.current_epoch,
            overall_quality: global_state.quality_metrics.overall_quality,
            sync_progress: global_state.get_sync_progress(),
            active_barriers: global_state.active_sync_barriers.len(),
            last_sync: global_state.last_global_sync,
        }
    }
}

/// Synchronization summary
#[derive(Debug, Clone)]
pub struct SyncSummary {
    /// Total number of devices
    pub total_devices: usize,
    /// Number of synchronized devices
    pub synchronized_devices: usize,
    /// Number of failed devices
    pub failed_devices: usize,
    /// Current synchronization epoch
    pub current_epoch: u64,
    /// Overall synchronization quality
    pub overall_quality: f64,
    /// Synchronization progress (0.0 to 1.0)
    pub sync_progress: f64,
    /// Number of active barriers
    pub active_barriers: usize,
    /// Last synchronization time
    pub last_sync: Option<Instant>,
}

/// State management testing utilities
#[cfg(test)]
pub mod testing {
    use super::*;

    /// Create test global state
    pub fn create_test_global_state() -> GlobalSynchronizationState {
        let mut state = GlobalSynchronizationState::new();

        // Add some test devices
        for i in 0..3 {
            let device_id = DeviceId::from(i);
            state.add_device(device_id);
        }

        state
    }

    /// Create test device state
    pub fn create_test_device_state(device_id: DeviceId) -> DeviceSyncState {
        DeviceSyncState {
            device_id,
            status: DeviceSyncStatus::Synchronized,
            last_sync: Some(Instant::now()),
            quality: 0.9,
            participation_count: 1,
            performance: DevicePerformanceMetrics {
                sync_latency: Duration::from_millis(50),
                throughput: 1000.0,
                success_rate: 0.99,
                resource_utilization: ResourceUtilization {
                    cpu: 0.6,
                    memory: 0.5,
                    network_bandwidth: 0.3,
                    storage: 0.4,
                },
            },
        }
    }

    /// Create test global barrier
    pub fn create_test_barrier(device_ids: &[DeviceId]) -> GlobalBarrier {
        let participants = device_ids.iter().cloned().collect();
        GlobalBarrier::new("test_barrier".to_string(), participants, Duration::from_secs(30))
    }
}