//! TPU Pod Coordination Synchronization
//!
//! This module provides comprehensive synchronization capabilities for TPU pod coordination,
//! including barriers, events, clocks, deadlock detection, consensus protocols, and adaptive
//! optimization. The synchronization system is designed for high-performance, fault-tolerant
//! coordination of distributed TPU devices.
//!
//! # Features
//!
//! * **Barrier Synchronization** - Advanced barrier management with optimization strategies
//! * **Event Synchronization** - Event-driven coordination with filtering and persistence
//! * **Clock Synchronization** - Precise time synchronization with drift compensation
//! * **Deadlock Detection** - Comprehensive deadlock prevention and resolution
//! * **Consensus Protocols** - Raft, PBFT, and Paxos implementations
//! * **Adaptive Optimization** - Machine learning-based performance optimization
//! * **Fault Tolerance** - Comprehensive fault detection and recovery
//! * **Performance Monitoring** - Real-time monitoring with alerts and analytics
//!
//! # Quick Start
//!
//! ```rust
//! use crate::tpu::pod_coordination::synchronization::prelude::*;
//!
//! // Create a synchronization manager
//! let config = SynchronizationConfig::default();
//! let mut sync_manager = SynchronizationManager::new(config)?;
//!
//! // Start synchronization
//! sync_manager.start()?;
//!
//! // Add devices to synchronization
//! sync_manager.add_device(DeviceId(0))?;
//! sync_manager.add_device(DeviceId(1))?;
//!
//! // Perform global synchronization
//! sync_manager.global_sync()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Architecture
//!
//! The synchronization system is organized into several specialized modules:
//!
//! * `config` - Configuration types and enums
//! * `barriers` - Barrier synchronization management
//! * `events` - Event-based synchronization
//! * `clocks` - Clock synchronization and time management
//! * `deadlock` - Deadlock detection and prevention
//! * `consensus` - Consensus protocol implementations
//! * `core` - Main synchronization manager and coordination logic

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use std::sync::Arc;

use crate::tpu::tpu_backend::DeviceId;
use crate::error::{Result, OptimError};

// Module declarations
pub mod config;
pub mod barriers;
pub mod events;
pub mod clocks;
pub mod deadlock;
pub mod consensus;
pub mod core;

// Re-export core types for convenience
pub use self::core::{
    SynchronizationManager,
    SynchronizationConfig,
    SynchronizationMode,
    GlobalSynchronizationState,
    GlobalSyncStatus,
    GlobalQualityMetrics,
    DeviceSyncState,
    DeviceSyncStatus,
    CoordinationScheduler,
    PerformanceMonitor,
    AdaptiveOptimizer,
    SynchronizationStatistics,
};

// Re-export configuration types
pub use self::config::{
    ClockSynchronizationConfig,
    ClockSyncProtocol,
    ClockAccuracyRequirements,
    BarrierConfig,
    BarrierOptimization,
    BarrierFaultTolerance,
    EventSynchronizationConfig,
    DeliveryGuarantees,
    EventOrdering,
    DeadlockDetectionConfig,
    DeadlockDetectionAlgorithm,
    ConsensusConfig,
    ConsensusProtocol,
    SynchronizationOptimization,
    OptimizationStrategy,
};

// Re-export barrier types
pub use self::barriers::{
    BarrierManager,
    BarrierState,
    BarrierStatus,
    BarrierType,
    BarrierStatistics,
    BarrierOptimizer,
    BarrierId,
};

// Re-export event types
pub use self::events::{
    EventSynchronizationManager,
    SyncEvent,
    SyncEventId,
    EventHandler,
    EventQueue,
    EventRouter,
    EventStatistics,
    EventFilter,
    EventPersistenceManager,
};

// Re-export clock types
pub use self::clocks::{
    ClockSynchronizationManager,
    ClockSource,
    ClockSynchronizer,
    ClockStatistics,
    TimeSourceManager,
    ClockQualityMonitor,
    TimeSource,
};

// Re-export deadlock types
pub use self::deadlock::{
    DeadlockDetector,
    DependencyGraph,
    DeadlockStatistics,
    DeadlockPreventionSystem,
    DeadlockRecoverySystem,
    ResourceId,
    TransactionId,
};

// Re-export consensus types
pub use self::consensus::{
    ConsensusProtocolManager,
    ConsensusProtocol as ConsensusProtocolTrait,
    ConsensusResult,
    ConsensusDecision,
    Vote,
    ProposalId,
    RaftConsensus,
    PBFTConsensus,
    PaxosConsensus,
    LeaderElectionManager,
    FaultToleranceManager,
};

/// Prelude module for easy imports
pub mod prelude {
    pub use super::{
        SynchronizationManager,
        SynchronizationConfig,
        SynchronizationMode,
        GlobalSynchronizationState,
        GlobalSyncStatus,
        DeviceId,
    };

    pub use super::barriers::{
        BarrierManager,
        BarrierType,
        BarrierStatus,
        BarrierId,
    };

    pub use super::events::{
        EventSynchronizationManager,
        SyncEvent,
        SyncEventId,
    };

    pub use super::clocks::{
        ClockSynchronizationManager,
        ClockSource,
        TimeSource,
    };

    pub use super::deadlock::{
        DeadlockDetector,
        ResourceId,
        TransactionId,
    };

    pub use super::consensus::{
        ConsensusProtocolManager,
        ConsensusResult,
        Vote,
        ProposalId,
    };

    pub use crate::error::{Result, OptimError};
    pub use std::time::{Duration, Instant};
    pub use std::collections::{HashMap, HashSet};
}

/// Constants for synchronization operations
pub mod constants {
    use std::time::Duration;

    /// Default global synchronization timeout
    pub const DEFAULT_GLOBAL_TIMEOUT: Duration = Duration::from_secs(30);

    /// Default barrier timeout
    pub const DEFAULT_BARRIER_TIMEOUT: Duration = Duration::from_secs(30);

    /// Default event timeout
    pub const DEFAULT_EVENT_TIMEOUT: Duration = Duration::from_secs(10);

    /// Default clock synchronization frequency
    pub const DEFAULT_CLOCK_SYNC_FREQUENCY: Duration = Duration::from_secs(60);

    /// Default consensus election timeout
    pub const DEFAULT_CONSENSUS_ELECTION_TIMEOUT: Duration = Duration::from_millis(150);

    /// Default consensus heartbeat interval
    pub const DEFAULT_CONSENSUS_HEARTBEAT_INTERVAL: Duration = Duration::from_millis(50);

    /// Maximum number of synchronization retries
    pub const MAX_SYNC_RETRIES: usize = 3;

    /// Default deadlock detection interval
    pub const DEFAULT_DEADLOCK_DETECTION_INTERVAL: Duration = Duration::from_millis(100);

    /// Default performance monitoring interval
    pub const DEFAULT_MONITORING_INTERVAL: Duration = Duration::from_secs(10);

    /// Default optimization frequency
    pub const DEFAULT_OPTIMIZATION_FREQUENCY: Duration = Duration::from_secs(300);

    /// Minimum synchronization quality threshold
    pub const MIN_SYNC_QUALITY_THRESHOLD: f64 = 0.8;

    /// Maximum allowed clock skew
    pub const MAX_CLOCK_SKEW: Duration = Duration::from_millis(10);

    /// Default resource allocation timeout
    pub const DEFAULT_RESOURCE_ALLOCATION_TIMEOUT: Duration = Duration::from_secs(5);

    /// Maximum concurrent operations
    pub const MAX_CONCURRENT_OPERATIONS: usize = 100;

    /// Default alert escalation timeout
    pub const DEFAULT_ALERT_ESCALATION_TIMEOUT: Duration = Duration::from_secs(300);
}

/// Type aliases for common synchronization types
pub mod types {
    use super::*;

    /// Synchronization result type
    pub type SyncResult<T> = Result<T>;

    /// Device set type
    pub type DeviceSet = HashSet<DeviceId>;

    /// Synchronization metrics type
    pub type SyncMetrics = HashMap<String, f64>;

    /// Configuration map type
    pub type ConfigMap = HashMap<String, String>;

    /// Performance data type
    pub type PerformanceData = Vec<(Instant, f64)>;

    /// Alert callback type
    pub type AlertCallback = Box<dyn Fn(&str) + Send + Sync>;

    /// Optimization callback type
    pub type OptimizationCallback = Box<dyn Fn(&HashMap<String, f64>) -> Result<HashMap<String, f64>> + Send + Sync>;

    /// Shared synchronization manager type
    pub type SharedSyncManager = Arc<std::sync::Mutex<SynchronizationManager>>;
}

/// Utility functions for synchronization operations
pub mod utils {
    use super::*;

    /// Create a default synchronization configuration
    pub fn create_default_config() -> SynchronizationConfig {
        SynchronizationConfig::default()
    }

    /// Create a minimal synchronization configuration for testing
    pub fn create_test_config() -> SynchronizationConfig {
        SynchronizationConfig {
            sync_mode: SynchronizationMode::BulkSynchronous,
            global_timeout: Duration::from_secs(5),
            clock_sync: config::ClockSynchronizationConfig {
                enable: false,
                ..Default::default()
            },
            barrier_config: config::BarrierConfig {
                default_timeout: Duration::from_secs(5),
                ..Default::default()
            },
            event_config: config::EventSynchronizationConfig::default(),
            deadlock_config: config::DeadlockDetectionConfig {
                enable: false,
                ..Default::default()
            },
            consensus_config: config::ConsensusConfig::default(),
            optimization: config::SynchronizationOptimization {
                enable: false,
                ..Default::default()
            },
        }
    }

    /// Create a high-performance synchronization configuration
    pub fn create_high_performance_config() -> SynchronizationConfig {
        SynchronizationConfig {
            sync_mode: SynchronizationMode::Adaptive {
                strategy: "high_performance".to_string()
            },
            global_timeout: Duration::from_secs(60),
            clock_sync: config::ClockSynchronizationConfig {
                enable: true,
                protocol: config::ClockSyncProtocol::PTP,
                sync_frequency: Duration::from_secs(30),
                accuracy_requirements: config::ClockAccuracyRequirements {
                    max_skew: Duration::from_millis(1),
                    target_accuracy: Duration::from_micros(100),
                    drift_tolerance: 1e-8,
                    ..Default::default()
                },
                ..Default::default()
            },
            barrier_config: config::BarrierConfig {
                optimization: config::BarrierOptimization {
                    enable: true,
                    strategy: config::BarrierOptimizationStrategy::TreeBased { fanout: 8 },
                    ..Default::default()
                },
                ..Default::default()
            },
            consensus_config: config::ConsensusConfig {
                protocol: config::ConsensusProtocol::Raft,
                optimization: config::ConsensusOptimization {
                    enable: true,
                    batching: config::ConsensusBatching {
                        enable: true,
                        batch_size: 1000,
                        timeout: Duration::from_millis(5),
                    },
                    pipelining: config::ConsensusPipelining {
                        enable: true,
                        depth: 20,
                        timeout: Duration::from_millis(50),
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            optimization: config::SynchronizationOptimization {
                enable: true,
                strategies: vec![
                    config::OptimizationStrategy::LockFree,
                    config::OptimizationStrategy::Optimistic,
                ],
                adaptive: config::AdaptiveOptimization {
                    enable: true,
                    learning: config::OptimizationLearning {
                        algorithm: config::LearningAlgorithm::ReinforcementLearning,
                        rate: 0.001,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Calculate synchronization quality from metrics
    pub fn calculate_sync_quality(metrics: &GlobalQualityMetrics) -> f64 {
        let weights = [0.2, 0.2, 0.2, 0.2, 0.2]; // Equal weights
        let values = [
            metrics.clock_quality,
            metrics.event_quality,
            metrics.barrier_quality,
            metrics.consensus_quality,
            metrics.deadlock_prevention_quality,
        ];

        weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum()
    }

    /// Check if synchronization quality meets threshold
    pub fn is_sync_quality_acceptable(quality: f64) -> bool {
        quality >= constants::MIN_SYNC_QUALITY_THRESHOLD
    }

    /// Convert duration to milliseconds as f64
    pub fn duration_to_ms(duration: Duration) -> f64 {
        duration.as_secs_f64() * 1000.0
    }

    /// Convert milliseconds to duration
    pub fn ms_to_duration(ms: f64) -> Duration {
        Duration::from_secs_f64(ms / 1000.0)
    }

    /// Generate a unique operation ID
    pub fn generate_operation_id() -> core::OperationId {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        core::OperationId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Create a device set from a vector of device IDs
    pub fn create_device_set(device_ids: Vec<u32>) -> DeviceSet {
        device_ids.into_iter().map(DeviceId).collect()
    }

    /// Validate synchronization configuration
    pub fn validate_config(config: &SynchronizationConfig) -> Result<()> {
        // Validate timeout values
        if config.global_timeout.as_secs() == 0 {
            return Err(OptimError::invalid_input("Global timeout cannot be zero"));
        }

        if config.barrier_config.default_timeout.as_secs() == 0 {
            return Err(OptimError::invalid_input("Barrier timeout cannot be zero"));
        }

        // Validate clock synchronization settings
        if config.clock_sync.enable {
            if config.clock_sync.sync_frequency.as_secs() == 0 {
                return Err(OptimError::invalid_input("Clock sync frequency cannot be zero"));
            }

            if config.clock_sync.accuracy_requirements.drift_tolerance <= 0.0 {
                return Err(OptimError::invalid_input("Drift tolerance must be positive"));
            }
        }

        // Validate consensus settings
        if config.consensus_config.parameters.quorum_size == 0 {
            return Err(OptimError::invalid_input("Consensus quorum size cannot be zero"));
        }

        Ok(())
    }

    /// Merge two synchronization configurations
    pub fn merge_configs(base: SynchronizationConfig, override_config: SynchronizationConfig) -> SynchronizationConfig {
        SynchronizationConfig {
            sync_mode: override_config.sync_mode,
            global_timeout: if override_config.global_timeout != Duration::from_secs(30) {
                override_config.global_timeout
            } else {
                base.global_timeout
            },
            clock_sync: if override_config.clock_sync.enable != base.clock_sync.enable {
                override_config.clock_sync
            } else {
                base.clock_sync
            },
            barrier_config: override_config.barrier_config,
            event_config: override_config.event_config,
            deadlock_config: override_config.deadlock_config,
            consensus_config: override_config.consensus_config,
            optimization: override_config.optimization,
        }
    }

    /// Get system health summary
    pub fn get_health_summary(sync_manager: &SynchronizationManager) -> HealthSummary {
        let global_state = sync_manager.get_global_state();
        let stats = sync_manager.get_statistics();

        HealthSummary {
            overall_status: global_state.status.clone(),
            device_count: global_state.participants.len(),
            synchronized_devices: global_state.device_states.values()
                .filter(|state| state.status == DeviceSyncStatus::Synchronized)
                .count(),
            average_quality: global_state.quality_metrics.overall_quality,
            last_sync: global_state.last_global_sync,
            performance_metrics: PerformanceSummary {
                avg_latency: stats.get("avg_latency").copied().unwrap_or(0.0),
                throughput: stats.get("throughput").copied().unwrap_or(0.0),
                error_rate: stats.get("error_rate").copied().unwrap_or(0.0),
            },
        }
    }

    /// Wait for synchronization with timeout
    pub fn wait_for_sync(sync_manager: &SynchronizationManager, timeout: Duration) -> Result<()> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            if sync_manager.get_global_state().is_synchronized() {
                return Ok(());
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        Err(OptimError::timeout("Synchronization timeout"))
    }

    /// Perform bulk device operations
    pub fn bulk_add_devices(sync_manager: &mut SynchronizationManager, device_ids: &[u32]) -> Result<()> {
        for &device_id in device_ids {
            sync_manager.add_device(DeviceId(device_id))?;
        }
        Ok(())
    }

    /// Calculate synchronization overhead
    pub fn calculate_sync_overhead(
        total_time: Duration,
        computation_time: Duration,
    ) -> SyncOverhead {
        let sync_time = total_time.saturating_sub(computation_time);
        let overhead_ratio = if total_time.as_nanos() > 0 {
            sync_time.as_secs_f64() / total_time.as_secs_f64()
        } else {
            0.0
        };

        SyncOverhead {
            sync_time,
            computation_time,
            total_time,
            overhead_ratio,
        }
    }
}

/// Health summary structure
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Overall synchronization status
    pub overall_status: GlobalSyncStatus,
    /// Total number of devices
    pub device_count: usize,
    /// Number of synchronized devices
    pub synchronized_devices: usize,
    /// Average synchronization quality
    pub average_quality: f64,
    /// Last synchronization timestamp
    pub last_sync: Option<Instant>,
    /// Performance metrics summary
    pub performance_metrics: PerformanceSummary,
}

/// Performance summary structure
#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    /// Average latency in milliseconds
    pub avg_latency: f64,
    /// Throughput in operations per second
    pub throughput: f64,
    /// Error rate as a percentage
    pub error_rate: f64,
}

/// Synchronization overhead information
#[derive(Debug, Clone)]
pub struct SyncOverhead {
    /// Time spent on synchronization
    pub sync_time: Duration,
    /// Time spent on computation
    pub computation_time: Duration,
    /// Total time
    pub total_time: Duration,
    /// Synchronization overhead ratio
    pub overhead_ratio: f64,
}

/// Builder for synchronization configuration
#[derive(Debug, Default)]
pub struct SynchronizationConfigBuilder {
    config: SynchronizationConfig,
}

impl SynchronizationConfigBuilder {
    /// Create a new configuration builder
    pub fn new() -> Self {
        Self {
            config: SynchronizationConfig::default(),
        }
    }

    /// Set synchronization mode
    pub fn sync_mode(mut self, mode: SynchronizationMode) -> Self {
        self.config.sync_mode = mode;
        self
    }

    /// Set global timeout
    pub fn global_timeout(mut self, timeout: Duration) -> Self {
        self.config.global_timeout = timeout;
        self
    }

    /// Enable clock synchronization
    pub fn enable_clock_sync(mut self, enable: bool) -> Self {
        self.config.clock_sync.enable = enable;
        self
    }

    /// Set clock synchronization protocol
    pub fn clock_protocol(mut self, protocol: config::ClockSyncProtocol) -> Self {
        self.config.clock_sync.protocol = protocol;
        self
    }

    /// Set consensus protocol
    pub fn consensus_protocol(mut self, protocol: config::ConsensusProtocol) -> Self {
        self.config.consensus_config.protocol = protocol;
        self
    }

    /// Enable optimization
    pub fn enable_optimization(mut self, enable: bool) -> Self {
        self.config.optimization.enable = enable;
        self
    }

    /// Enable deadlock detection
    pub fn enable_deadlock_detection(mut self, enable: bool) -> Self {
        self.config.deadlock_config.enable = enable;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<SynchronizationConfig> {
        utils::validate_config(&self.config)?;
        Ok(self.config)
    }
}

/// Synchronization metrics collector
#[derive(Debug)]
pub struct MetricsCollector {
    /// Collected metrics
    metrics: HashMap<String, Vec<(Instant, f64)>>,
    /// Collection start time
    start_time: Instant,
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    /// Add a metric value
    pub fn add_metric(&mut self, name: &str, value: f64) {
        let timestamp = Instant::now();
        self.metrics.entry(name.to_string())
            .or_insert_with(Vec::new)
            .push((timestamp, value));
    }

    /// Get metric history
    pub fn get_metric_history(&self, name: &str) -> Option<&Vec<(Instant, f64)>> {
        self.metrics.get(name)
    }

    /// Calculate average for a metric
    pub fn calculate_average(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).map(|values| {
            if values.is_empty() {
                0.0
            } else {
                let sum: f64 = values.iter().map(|(_, v)| v).sum();
                sum / values.len() as f64
            }
        })
    }

    /// Get collection duration
    pub fn collection_duration(&self) -> Duration {
        self.start_time.elapsed()
    }
}

/// Error types specific to synchronization
#[derive(Debug, Clone)]
pub enum SyncError {
    /// Synchronization timeout
    Timeout { operation: String, duration: Duration },
    /// Device not found
    DeviceNotFound { device_id: DeviceId },
    /// Synchronization quality too low
    QualityTooLow { current: f64, required: f64 },
    /// Resource allocation failed
    ResourceAllocationFailed { resource: String, reason: String },
    /// Consensus failed
    ConsensusFailed { reason: String },
    /// Deadlock detected
    DeadlockDetected { resources: Vec<String> },
    /// Configuration invalid
    InvalidConfiguration { field: String, reason: String },
    /// Unknown error
    Unknown { message: String },
}

impl std::fmt::Display for SyncError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SyncError::Timeout { operation, duration } =>
                write!(f, "Synchronization timeout for {} after {:?}", operation, duration),
            SyncError::DeviceNotFound { device_id } =>
                write!(f, "Device not found: {:?}", device_id),
            SyncError::QualityTooLow { current, required } =>
                write!(f, "Synchronization quality too low: {} < {}", current, required),
            SyncError::ResourceAllocationFailed { resource, reason } =>
                write!(f, "Resource allocation failed for {}: {}", resource, reason),
            SyncError::ConsensusFailed { reason } =>
                write!(f, "Consensus failed: {}", reason),
            SyncError::DeadlockDetected { resources } =>
                write!(f, "Deadlock detected involving resources: {:?}", resources),
            SyncError::InvalidConfiguration { field, reason } =>
                write!(f, "Invalid configuration for {}: {}", field, reason),
            SyncError::Unknown { message } =>
                write!(f, "Unknown synchronization error: {}", message),
        }
    }
}

impl std::error::Error for SyncError {}

/// Synchronization event types for monitoring
#[derive(Debug, Clone)]
pub enum SyncEvent {
    /// Synchronization started
    SyncStarted { devices: Vec<DeviceId> },
    /// Synchronization completed
    SyncCompleted { duration: Duration, quality: f64 },
    /// Synchronization failed
    SyncFailed { reason: String },
    /// Device added
    DeviceAdded { device_id: DeviceId },
    /// Device removed
    DeviceRemoved { device_id: DeviceId },
    /// Quality degraded
    QualityDegraded { from: f64, to: f64 },
    /// Optimization applied
    OptimizationApplied { config: HashMap<String, f64> },
    /// Alert triggered
    AlertTriggered { alert_type: String, severity: String },
}

/// Synchronization event listener trait
pub trait SyncEventListener: Send + Sync {
    /// Handle synchronization event
    fn handle_event(&self, event: &SyncEvent);
}

/// Default implementation for debugging
impl<F> SyncEventListener for F
where
    F: Fn(&SyncEvent) + Send + Sync,
{
    fn handle_event(&self, event: &SyncEvent) {
        self(event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn test_default_configuration() {
        let config = SynchronizationConfig::default();
        assert_eq!(config.sync_mode, SynchronizationMode::BulkSynchronous);
        assert_eq!(config.global_timeout, Duration::from_secs(30));
        assert!(config.clock_sync.enable);
    }

    #[test]
    fn test_configuration_validation() {
        let mut config = SynchronizationConfig::default();

        // Valid configuration should pass
        assert!(utils::validate_config(&config).is_ok());

        // Invalid configuration should fail
        config.global_timeout = Duration::from_secs(0);
        assert!(utils::validate_config(&config).is_err());
    }

    #[test]
    fn test_configuration_builder() {
        let config = SynchronizationConfigBuilder::new()
            .sync_mode(SynchronizationMode::EventDriven)
            .global_timeout(Duration::from_secs(60))
            .enable_clock_sync(true)
            .clock_protocol(config::ClockSyncProtocol::PTP)
            .consensus_protocol(config::ConsensusProtocol::Raft)
            .enable_optimization(true)
            .build()
            .unwrap();

        assert_eq!(config.sync_mode, SynchronizationMode::EventDriven);
        assert_eq!(config.global_timeout, Duration::from_secs(60));
        assert!(config.clock_sync.enable);
        assert_eq!(config.clock_sync.protocol, config::ClockSyncProtocol::PTP);
        assert_eq!(config.consensus_config.protocol, config::ConsensusProtocol::Raft);
        assert!(config.optimization.enable);
    }

    #[test]
    fn test_synchronization_manager_creation() {
        let config = utils::create_test_config();
        let sync_manager = SynchronizationManager::new(config);
        assert!(sync_manager.is_ok());
    }

    #[test]
    fn test_device_management() {
        let config = utils::create_test_config();
        let mut sync_manager = SynchronizationManager::new(config).unwrap();

        // Add devices
        assert!(sync_manager.add_device(DeviceId(0)).is_ok());
        assert!(sync_manager.add_device(DeviceId(1)).is_ok());

        // Check device count
        assert_eq!(sync_manager.get_global_state().participants.len(), 2);

        // Remove device
        assert!(sync_manager.remove_device(DeviceId(0)).is_ok());
        assert_eq!(sync_manager.get_global_state().participants.len(), 1);
    }

    #[test]
    fn test_global_state_management() {
        let mut global_state = GlobalSynchronizationState::new();

        // Initial state should be not synchronized
        assert!(!global_state.is_synchronized());
        assert_eq!(global_state.get_quality(), 0.0);

        // Update to synchronized state
        global_state.status = GlobalSyncStatus::Synchronized { quality: 0.9 };
        assert!(global_state.is_synchronized());
    }

    #[test]
    fn test_sync_quality_calculation() {
        let metrics = GlobalQualityMetrics {
            overall_quality: 0.0,
            clock_quality: 0.8,
            event_quality: 0.9,
            barrier_quality: 0.7,
            consensus_quality: 0.85,
            deadlock_prevention_quality: 0.95,
            coordination_efficiency: 0.8,
        };

        let quality = utils::calculate_sync_quality(&metrics);
        assert!(quality > 0.0 && quality <= 1.0);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();

        // Add some metrics
        collector.add_metric("latency", 10.0);
        collector.add_metric("latency", 20.0);
        collector.add_metric("throughput", 100.0);

        // Check averages
        assert_eq!(collector.calculate_average("latency"), Some(15.0));
        assert_eq!(collector.calculate_average("throughput"), Some(100.0));
        assert_eq!(collector.calculate_average("nonexistent"), None);
    }

    #[test]
    fn test_duration_conversion() {
        let duration = Duration::from_millis(1500);
        let ms = utils::duration_to_ms(duration);
        assert_eq!(ms, 1500.0);

        let converted_back = utils::ms_to_duration(ms);
        assert_eq!(converted_back, duration);
    }

    #[test]
    fn test_device_set_creation() {
        let device_ids = vec![0, 1, 2, 3];
        let device_set = utils::create_device_set(device_ids);

        assert_eq!(device_set.len(), 4);
        assert!(device_set.contains(&DeviceId(0)));
        assert!(device_set.contains(&DeviceId(3)));
    }

    #[test]
    fn test_operation_id_generation() {
        let id1 = utils::generate_operation_id();
        let id2 = utils::generate_operation_id();

        // IDs should be unique
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_sync_overhead_calculation() {
        let total_time = Duration::from_millis(1000);
        let computation_time = Duration::from_millis(800);

        let overhead = utils::calculate_sync_overhead(total_time, computation_time);

        assert_eq!(overhead.sync_time, Duration::from_millis(200));
        assert_eq!(overhead.overhead_ratio, 0.2);
    }

    #[test]
    fn test_config_merging() {
        let base_config = SynchronizationConfig::default();
        let override_config = SynchronizationConfig {
            sync_mode: SynchronizationMode::EventDriven,
            global_timeout: Duration::from_secs(60),
            ..SynchronizationConfig::default()
        };

        let merged = utils::merge_configs(base_config, override_config);

        assert_eq!(merged.sync_mode, SynchronizationMode::EventDriven);
        assert_eq!(merged.global_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_high_performance_config() {
        let config = utils::create_high_performance_config();

        assert!(matches!(config.sync_mode, SynchronizationMode::Adaptive { .. }));
        assert!(config.optimization.enable);
        assert!(config.consensus_config.optimization.enable);
        assert_eq!(config.clock_sync.protocol, config::ClockSyncProtocol::PTP);
    }

    #[test]
    fn test_sync_error_display() {
        let error = SyncError::Timeout {
            operation: "global_sync".to_string(),
            duration: Duration::from_secs(30),
        };

        let error_msg = format!("{}", error);
        assert!(error_msg.contains("global_sync"));
        assert!(error_msg.contains("30s"));
    }

    #[test]
    fn test_health_summary_creation() {
        let config = utils::create_test_config();
        let sync_manager = SynchronizationManager::new(config).unwrap();

        let health = utils::get_health_summary(&sync_manager);

        assert_eq!(health.device_count, 0);
        assert_eq!(health.synchronized_devices, 0);
        assert!(!matches!(health.overall_status, GlobalSyncStatus::Synchronized { .. }));
    }

    #[test]
    fn test_sync_event_handling() {
        let mut event_received = false;
        let listener = |_event: &SyncEvent| {
            event_received = true;
        };

        let event = SyncEvent::SyncStarted {
            devices: vec![DeviceId(0), DeviceId(1)],
        };

        listener.handle_event(&event);
        // Note: In a real test, we'd need some way to verify the event was handled
        // This is a simplified example
    }

    #[test]
    fn test_bulk_device_operations() {
        let config = utils::create_test_config();
        let mut sync_manager = SynchronizationManager::new(config).unwrap();

        let device_ids = vec![0, 1, 2, 3, 4];
        assert!(utils::bulk_add_devices(&mut sync_manager, &device_ids).is_ok());

        assert_eq!(sync_manager.get_global_state().participants.len(), 5);
    }
}

/// Integration tests for the synchronization module
#[cfg(test)]
mod integration_tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_full_synchronization_workflow() {
        let config = utils::create_test_config();
        let mut sync_manager = SynchronizationManager::new(config).unwrap();

        // Start the synchronization manager
        assert!(sync_manager.start().is_ok());

        // Add multiple devices
        for i in 0..5 {
            assert!(sync_manager.add_device(DeviceId(i)).is_ok());
        }

        // Check that all devices are added
        assert_eq!(sync_manager.get_global_state().participants.len(), 5);

        // Perform global synchronization
        assert!(sync_manager.global_sync().is_ok());

        // Verify synchronization completed
        let global_state = sync_manager.get_global_state();
        assert!(matches!(global_state.status, GlobalSyncStatus::Synchronized { .. }));

        // Stop the synchronization manager
        assert!(sync_manager.stop().is_ok());
    }

    #[test]
    fn test_concurrent_device_operations() {
        let config = utils::create_test_config();
        let sync_manager = Arc::new(Mutex::new(
            SynchronizationManager::new(config).unwrap()
        ));

        let mut handles = vec![];

        // Spawn multiple threads to add devices concurrently
        for i in 0..10 {
            let manager = Arc::clone(&sync_manager);
            let handle = thread::spawn(move || {
                let mut mgr = manager.lock().unwrap();
                mgr.add_device(DeviceId(i)).unwrap();
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Check that all devices were added
        let mgr = sync_manager.lock().unwrap();
        assert_eq!(mgr.get_global_state().participants.len(), 10);
    }

    #[test]
    fn test_synchronization_quality_monitoring() {
        let config = utils::create_high_performance_config();
        let mut sync_manager = SynchronizationManager::new(config).unwrap();

        // Add devices
        for i in 0..3 {
            sync_manager.add_device(DeviceId(i)).unwrap();
        }

        // Start manager
        sync_manager.start().unwrap();

        // Perform synchronization
        sync_manager.global_sync().unwrap();

        // Check quality metrics
        let quality = sync_manager.get_global_state().get_quality();
        assert!(utils::is_sync_quality_acceptable(quality));

        // Get health summary
        let health = utils::get_health_summary(&sync_manager);
        assert_eq!(health.synchronized_devices, 3);

        sync_manager.stop().unwrap();
    }

    #[test]
    fn test_metrics_collection_workflow() {
        let mut collector = MetricsCollector::new();

        // Simulate collecting metrics over time
        for i in 0..100 {
            collector.add_metric("latency", (i as f64) * 0.1);
            collector.add_metric("throughput", 1000.0 - (i as f64) * 2.0);
        }

        // Verify metrics collection
        assert_eq!(collector.get_metric_history("latency").unwrap().len(), 100);
        assert_eq!(collector.get_metric_history("throughput").unwrap().len(), 100);

        // Check averages
        let avg_latency = collector.calculate_average("latency").unwrap();
        let avg_throughput = collector.calculate_average("throughput").unwrap();

        assert!(avg_latency > 0.0);
        assert!(avg_throughput > 0.0);
    }

    #[test]
    fn test_configuration_builder_workflow() {
        let config = SynchronizationConfigBuilder::new()
            .sync_mode(SynchronizationMode::Hybrid {
                modes: vec!["barrier".to_string(), "event".to_string()],
            })
            .global_timeout(Duration::from_secs(45))
            .enable_clock_sync(true)
            .clock_protocol(config::ClockSyncProtocol::PTP)
            .consensus_protocol(config::ConsensusProtocol::PBFT)
            .enable_optimization(true)
            .enable_deadlock_detection(true)
            .build()
            .unwrap();

        // Verify configuration was built correctly
        assert!(matches!(config.sync_mode, SynchronizationMode::Hybrid { .. }));
        assert_eq!(config.global_timeout, Duration::from_secs(45));
        assert!(config.clock_sync.enable);
        assert_eq!(config.clock_sync.protocol, config::ClockSyncProtocol::PTP);
        assert_eq!(config.consensus_config.protocol, config::ConsensusProtocol::PBFT);
        assert!(config.optimization.enable);
        assert!(config.deadlock_config.enable);

        // Create sync manager with built configuration
        let sync_manager = SynchronizationManager::new(config);
        assert!(sync_manager.is_ok());
    }
}

/// Performance benchmarks for synchronization operations
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    #[test]
    fn benchmark_device_addition() {
        let config = utils::create_test_config();
        let mut sync_manager = SynchronizationManager::new(config).unwrap();

        let start = Instant::now();

        // Add 1000 devices
        for i in 0..1000 {
            sync_manager.add_device(DeviceId(i)).unwrap();
        }

        let duration = start.elapsed();

        println!("Added 1000 devices in {:?}", duration);
        println!("Average time per device: {:?}", duration / 1000);

        // Verify all devices were added
        assert_eq!(sync_manager.get_global_state().participants.len(), 1000);
    }

    #[test]
    fn benchmark_synchronization_operations() {
        let config = utils::create_test_config();
        let mut sync_manager = SynchronizationManager::new(config).unwrap();

        // Add devices
        for i in 0..100 {
            sync_manager.add_device(DeviceId(i)).unwrap();
        }

        sync_manager.start().unwrap();

        let start = Instant::now();

        // Perform 10 synchronization cycles
        for _ in 0..10 {
            sync_manager.global_sync().unwrap();
        }

        let duration = start.elapsed();

        println!("Performed 10 sync cycles with 100 devices in {:?}", duration);
        println!("Average sync time: {:?}", duration / 10);

        sync_manager.stop().unwrap();
    }

    #[test]
    fn benchmark_metrics_collection() {
        let mut collector = MetricsCollector::new();

        let start = Instant::now();

        // Collect 10,000 metrics
        for i in 0..10_000 {
            collector.add_metric("test_metric", i as f64);
        }

        let duration = start.elapsed();

        println!("Collected 10,000 metrics in {:?}", duration);
        println!("Average time per metric: {:?}", duration / 10_000);

        // Calculate average
        let avg_start = Instant::now();
        let average = collector.calculate_average("test_metric").unwrap();
        let avg_duration = avg_start.elapsed();

        println!("Calculated average of 10,000 values in {:?}", avg_duration);
        assert!(average > 0.0);
    }
}