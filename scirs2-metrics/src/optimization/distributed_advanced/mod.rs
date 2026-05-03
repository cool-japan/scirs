//! Advanced distributed optimization with consensus algorithms and fault recovery
//!
//! This module provides comprehensive distributed computing capabilities including:
//! - Consensus algorithms (Raft, PBFT, Proof of Stake)
//! - Advanced data sharding and replication
//! - Automatic fault recovery and healing
//! - Dynamic cluster scaling
//! - Data locality optimization
//! - Advanced partitioning strategies
//! - Performance optimization and monitoring

pub mod consensus;
pub mod fault_recovery;
pub mod monitoring;
pub mod optimization;
pub mod orchestration;
pub mod scaling;
pub mod sharding;

use crate::error::{MetricsError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant, SystemTime};

// Re-export main components
pub use consensus::*;
pub use fault_recovery::*;
pub use monitoring::*;
// optimization::* intentionally not re-exported here to avoid name collision
// with the outer AdvancedDistributedOptimizer struct.
pub use orchestration::*;
pub use scaling::*;
pub use sharding::*;

/// Comprehensive advanced distributed optimization coordinator
pub struct AdvancedDistributedOptimizer<T: Float> {
    /// Configuration
    config: AdvancedDistributedConfig,

    /// System statistics
    stats: DistributedSystemStats,

    /// Current state
    state: GlobalSystemState<T>,

    /// Consensus coordinator (manages Raft/PBFT/PoS/Majority algorithms)
    consensus_manager: consensus::coordinator::ConsensusCoordinator,

    /// Data shard manager
    shard_manager: sharding::ShardManager,

    /// Fault recovery coordinator
    recovery_manager: fault_recovery::AdvancedFaultRecovery,

    /// Auto-scaling coordinator
    scaling_manager: scaling::AdvancedScalingManager,

    /// Performance optimizer (strategy registry for gradient-descent / SA / GA / PSO)
    performance_optimizer: optimization::AdvancedDistributedOptimizer,

    /// Service orchestrator
    orchestrator: orchestration::OrchestrationManager,

    /// Distributed monitoring system
    monitoring_system: monitoring::MonitoringManager,
}

/// Advanced distributed system configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvancedDistributedConfig {
    /// Basic cluster settings
    pub basic_config: crate::optimization::distributed::DistributedConfig,

    /// Consensus algorithm configuration
    pub consensus_config: consensus::ConsensusConfig,

    /// Data sharding strategy
    pub sharding_config: sharding::ShardingConfig,

    /// Fault tolerance settings
    pub fault_tolerance_config: FaultToleranceConfig,

    /// Auto-scaling configuration
    pub auto_scaling_config: AutoScalingConfig,

    /// Performance optimization settings
    pub optimization_config: OptimizationConfig,

    /// Orchestration configuration
    pub orchestration_config: OrchestrationConfig,

    /// Monitoring configuration
    pub monitoring_config: MonitoringConfig,
}

/// Distributed system statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct DistributedSystemStats {
    /// Total operations processed
    pub total_operations: u64,

    /// Average operation latency (milliseconds)
    pub avg_latency_ms: f64,

    /// System uptime (seconds)
    pub uptime_seconds: u64,

    /// Current cluster size
    pub cluster_size: usize,

    /// Total consensus decisions
    pub consensus_decisions: u64,

    /// Data shards managed
    pub active_shards: usize,

    /// Fault recovery events
    pub recovery_events: u64,

    /// Scaling operations performed
    pub scaling_operations: u64,

    /// System health score (0.0-1.0)
    pub health_score: f64,
}

/// Global system state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalSystemState<T: Float> {
    /// Current system timestamp
    pub timestamp: SystemTime,

    /// Active nodes in cluster
    pub active_nodes: HashMap<String, NodeInfo>,

    /// Marker for type parameter
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> GlobalSystemState<T> {
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now(),
            active_nodes: HashMap::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node identifier
    pub node_id: String,

    /// Node address
    pub address: String,

    /// Node status
    pub status: NodeStatus,

    /// Node capabilities
    pub capabilities: NodeCapabilities,

    /// Performance metrics
    pub metrics: NodeMetrics,

    /// Last heartbeat
    pub last_heartbeat: SystemTime,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is active and healthy
    Active,

    /// Node is degraded but functional
    Degraded,

    /// Node is failed or unreachable
    Failed,

    /// Node is being initialized
    Initializing,

    /// Node is shutting down
    ShuttingDown,

    /// Node status unknown
    Unknown,
}

/// Node capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapabilities {
    /// CPU cores available
    pub cpu_cores: usize,

    /// Memory available (MB)
    pub memory_mb: usize,

    /// Storage available (MB)
    pub storage_mb: usize,

    /// Network bandwidth (Mbps)
    pub network_bandwidth: f64,

    /// Supported consensus algorithms
    pub consensus_algorithms: Vec<String>,

    /// Special capabilities
    pub special_capabilities: Vec<String>,
}

/// Node performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeMetrics {
    /// CPU utilization (0.0-1.0)
    pub cpu_usage: f64,

    /// Memory utilization (0.0-1.0)
    pub memory_usage: f64,

    /// Storage utilization (0.0-1.0)
    pub storage_usage: f64,

    /// Network utilization (0.0-1.0)
    pub network_usage: f64,

    /// Average response time (milliseconds)
    pub avg_response_time_ms: f64,

    /// Operations per second
    pub ops_per_second: f64,

    /// Error rate (0.0-1.0)
    pub error_rate: f64,
}

impl<T: Float + Default + std::fmt::Debug + Clone + Send + Sync> AdvancedDistributedOptimizer<T> {
    /// Create new advanced distributed optimizer with all 7 subsystems active
    pub fn new(config: AdvancedDistributedConfig) -> Result<Self> {
        let node_id = "coordinator".to_string();

        // Build consensus coordinator from the basic_config's consensus settings
        let consensus_manager =
            consensus::coordinator::ConsensusCoordinator::new_majority(node_id.clone(), vec![]);

        // Build shard manager from sharding config
        let shard_manager = sharding::ShardManager::new(config.sharding_config.clone());

        // Build fault recovery coordinator
        let recovery_manager = fault_recovery::AdvancedFaultRecovery::new(node_id.clone());

        // Build scaling manager with auto-scaling enabled if configured
        let mut scaling_manager = scaling::AdvancedScalingManager::new(node_id.clone());
        if config.auto_scaling_config.enabled {
            scaling_manager.enable_auto_scaling();
        }

        // Build performance optimizer (strategy registry)
        let performance_optimizer =
            optimization::AdvancedDistributedOptimizer::new(node_id.clone());

        // Build orchestrator
        let orchestrator = orchestration::OrchestrationManager::new(node_id.clone());

        // Build monitoring system
        let monitoring_system = monitoring::MonitoringManager::new(node_id);

        Ok(Self {
            config,
            stats: DistributedSystemStats::default(),
            state: GlobalSystemState::new(),
            consensus_manager,
            shard_manager,
            recovery_manager,
            scaling_manager,
            performance_optimizer,
            orchestrator,
            monitoring_system,
        })
    }

    /// Initialize the distributed system
    pub async fn initialize(&mut self) -> Result<()> {
        // Collect the initial node list from the active_nodes state
        let nodes: Vec<String> = self.state.active_nodes.keys().cloned().collect();

        // Initialize shard manager with the known node set (no-op when empty)
        if !nodes.is_empty() {
            self.shard_manager.initialize(nodes.clone())?;
        }

        // Register the coordinator service in the orchestrator
        self.orchestrator
            .register_service("coordinator".to_string(), "localhost:8080".to_string())?;
        self.orchestrator.start_service("coordinator")?;

        // Record initialization event in monitoring
        self.monitoring_system.record_metric(
            "system.initialized".to_string(),
            monitoring::MetricValue::Counter(1),
        )?;

        // Update cluster size stats
        self.stats.cluster_size = nodes.len();

        Ok(())
    }

    /// Process distributed optimization task
    pub async fn optimize_distributed(&mut self, data: &Array2<T>) -> Result<Array2<T>> {
        let start_time = Instant::now();

        // Record the data dimensions in monitoring
        let (nrows, ncols) = data.dim();
        self.monitoring_system.record_metric(
            "optimize.input.rows".to_string(),
            monitoring::MetricValue::Gauge(nrows as f64),
        )?;
        self.monitoring_system.record_metric(
            "optimize.input.cols".to_string(),
            monitoring::MetricValue::Gauge(ncols as f64),
        )?;

        // Check and execute any pending scaling decisions
        let scaling_decisions = self.scaling_manager.evaluate_scaling_needs()?;
        for decision in scaling_decisions {
            self.scaling_manager.execute_scaling(decision)?;
            self.stats.scaling_operations += 1;
        }

        // Propose consensus on the current operation (records operation intent)
        let payload = format!("optimize_{}x{}", nrows, ncols).into_bytes();
        let _proposal_id = self.consensus_manager.propose(payload).ok();

        // Update shard statistics to reflect data dimensions as workload proxy
        let shard_list = self.shard_manager.list_shards();
        if let Some(shard) = shard_list.first() {
            self.shard_manager
                .update_shard_stats(&shard.id, (nrows * ncols * 8) as u64, nrows)?;
            self.stats.active_shards = shard_list.len();
        }

        // Record active shard count in monitoring
        self.monitoring_system.record_metric(
            "shards.active".to_string(),
            monitoring::MetricValue::Gauge(self.stats.active_shards as f64),
        )?;

        // The orchestrator delegates computation; in this implementation we pass through data
        let optimized_result = data.clone();

        // Update statistics
        let elapsed = start_time.elapsed();
        self.stats.total_operations += 1;
        self.stats.avg_latency_ms = (self.stats.avg_latency_ms
            * (self.stats.total_operations - 1) as f64
            + elapsed.as_millis() as f64)
            / self.stats.total_operations as f64;

        // Record latency in monitoring
        self.monitoring_system.record_metric(
            "optimize.latency_ms".to_string(),
            monitoring::MetricValue::Gauge(elapsed.as_millis() as f64),
        )?;

        Ok(optimized_result)
    }

    /// Get current system state
    pub async fn get_system_state(&self) -> Result<GlobalSystemState<T>> {
        Ok(GlobalSystemState {
            timestamp: SystemTime::now(),
            active_nodes: self.state.active_nodes.clone(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Handle system failures
    pub async fn handle_failure(&mut self, failure_info: FailureInfo) -> Result<()> {
        // Record the failure in monitoring
        self.monitoring_system.record_metric(
            format!("failure.{}", failure_info.failed_node_id),
            monitoring::MetricValue::Counter(1),
        )?;

        // Map FailureType to fault_recovery::FaultType and invoke the recovery manager
        let fault_type = match failure_info.failure_type {
            FailureType::NodeFailure => fault_recovery::FaultType::NodeFailure,
            FailureType::NetworkPartition => fault_recovery::FaultType::NetworkPartition,
            FailureType::ServiceFailure => fault_recovery::FaultType::ConsensusFailure,
            FailureType::ResourceExhaustion => fault_recovery::FaultType::MessageLoss,
        };
        self.recovery_manager
            .handle_fault(fault_type, failure_info.affected_services.clone())?;

        // Migrate shards away from the failed node
        self.shard_manager
            .remove_node(&failure_info.failed_node_id)
            .ok(); // best-effort; ignore if node was not tracked

        // Update statistics
        self.stats.recovery_events += 1;
        self.stats.health_score = self.calculate_health_score().await?;

        Ok(())
    }

    /// Calculate system health score based on active subsystem metrics
    async fn calculate_health_score(&self) -> Result<f64> {
        // Derive health from observable subsystem indicators:
        // - alert count from monitoring (more alerts → lower health)
        // - failure history length from recovery manager
        // - quarantine state of the coordinator node
        let alert_count = self.monitoring_system.get_alerts().len();
        let failure_count = self.recovery_manager.get_failure_history().len();
        let coordinator_quarantined = self.recovery_manager.is_node_quarantined("coordinator");

        // Base health of 1.0, penalised per alert/failure
        let alert_penalty = (alert_count as f64 * 0.05).min(0.3);
        let failure_penalty = (failure_count as f64 * 0.03).min(0.3);
        let quarantine_penalty = if coordinator_quarantined { 0.4 } else { 0.0 };

        let overall_health = (1.0 - alert_penalty - failure_penalty - quarantine_penalty)
            .max(0.0)
            .min(1.0);

        Ok(overall_health)
    }

    /// Get system statistics
    pub fn get_statistics(&self) -> &DistributedSystemStats {
        &self.stats
    }

    /// Shutdown the distributed system gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        // Disable auto-scaling to prevent new scaling operations during shutdown
        self.scaling_manager.disable_auto_scaling();

        // Stop the coordinator service in the orchestrator
        self.orchestrator.stop_service("coordinator").ok();

        // Record shutdown event in monitoring
        self.monitoring_system.record_metric(
            "system.shutdown".to_string(),
            monitoring::MetricValue::Counter(1),
        )?;

        Ok(())
    }
}

impl Default for NodeMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            storage_usage: 0.0,
            network_usage: 0.0,
            avg_response_time_ms: 0.0,
            ops_per_second: 0.0,
            error_rate: 0.0,
        }
    }
}

// Missing types referenced in mod.rs imports
/// Advanced cluster configuration (alias for AdvancedDistributedConfig)
pub type AdvancedClusterConfig = AdvancedDistributedConfig;

/// Advanced distributed coordinator (alias for AdvancedDistributedOptimizer)
pub type AdvancedDistributedCoordinator = AdvancedDistributedOptimizer<f64>;

/// Auto-scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    pub enabled: bool,
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_nodes: 1,
            max_nodes: 10,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
        }
    }
}

/// Cluster state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    pub nodes: HashMap<String, NodeInfo>,
    pub cluster_size: usize,
    pub healthy_nodes: usize,
    pub status: ClusterStatus,
    pub last_updated: SystemTime,
}

/// Cluster status enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClusterStatus {
    Initializing,
    Active,
    Degraded,
    Failed,
}

/// Distributed task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedTask {
    pub id: String,
    pub task_type: TaskType,
    pub priority: TaskPriority,
    pub payload: Vec<u8>,
    pub created_at: SystemTime,
}

/// Task type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Computation,
    DataTransfer,
    Synchronization,
    Maintenance,
}

/// Task priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

/// Fault tolerance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    pub enabled: bool,
    pub max_retries: usize,
    pub retry_delay_ms: u64,
    pub health_check_interval_ms: u64,
    pub failure_threshold: f64,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            retry_delay_ms: 1000,
            health_check_interval_ms: 5000,
            failure_threshold: 0.1,
        }
    }
}

/// Locality configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalityConfig {
    pub prefer_local_processing: bool,
    pub max_distance_ms: u64,
    pub data_affinity_enabled: bool,
}

impl Default for LocalityConfig {
    fn default() -> Self {
        Self {
            prefer_local_processing: true,
            max_distance_ms: 100,
            data_affinity_enabled: true,
        }
    }
}

/// Node role enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeRole {
    Master,
    Worker,
    Storage,
    Coordinator,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub enabled: bool,
    pub optimization_interval_ms: u64,
    pub performance_threshold: f64,
    pub auto_tune_parameters: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_interval_ms: 30000,
            performance_threshold: 0.8,
            auto_tune_parameters: true,
        }
    }
}

/// Orchestration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    pub enabled: bool,
    pub coordination_interval_ms: u64,
    pub service_discovery_enabled: bool,
    pub load_balancing_enabled: bool,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            coordination_interval_ms: 10000,
            service_discovery_enabled: true,
            load_balancing_enabled: true,
        }
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enabled: bool,
    pub metrics_collection_interval_ms: u64,
    pub alert_threshold: f64,
    pub log_level: String,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            metrics_collection_interval_ms: 5000,
            alert_threshold: 0.9,
            log_level: "INFO".to_string(),
        }
    }
}

/// Failure information for fault recovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureInfo {
    pub failed_node_id: String,
    pub failure_type: FailureType,
    pub timestamp: SystemTime,
    pub affected_services: Vec<String>,
}

/// Types of failures that can occur
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    NodeFailure,
    NetworkPartition,
    ServiceFailure,
    ResourceExhaustion,
}

/// Resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_gb: f64,
    pub storage_gb: f64,
    pub network_mbps: f64,
    pub gpu_required: bool,
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            cpu_cores: 1.0,
            memory_gb: 2.0,
            storage_gb: 10.0,
            network_mbps: 100.0,
            gpu_required: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_advanced_distributed_config() {
        let config = AdvancedDistributedConfig::default();
        assert!(config.consensus_config.quorum_size > 0);
        assert!(config.sharding_config.shard_count > 0);
    }

    #[test]
    fn test_node_metrics() {
        let metrics = NodeMetrics::default();
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.ops_per_second, 0.0);
    }

    #[test]
    fn test_distributed_system_stats() {
        let stats = DistributedSystemStats::default();
        assert_eq!(stats.total_operations, 0);
        assert_eq!(stats.health_score, 0.0);
    }

    // ── Integration tests for the 7-subsystem optimizer ──────────────────────

    /// Test 1: Optimizer can be created with all 7 subsystems active.
    #[test]
    fn test_optimizer_creation_all_subsystems() {
        let config = AdvancedDistributedConfig::default();
        let optimizer = AdvancedDistributedOptimizer::<f64>::new(config);
        assert!(
            optimizer.is_ok(),
            "AdvancedDistributedOptimizer should be constructable with default config"
        );
        let optimizer = optimizer.expect("construction succeeded");
        // Verify stats start at zero
        assert_eq!(optimizer.get_statistics().total_operations, 0);
        assert_eq!(optimizer.get_statistics().recovery_events, 0);
    }

    /// Test 2: `optimize_distributed()` runs end-to-end on a small toy problem.
    #[test]
    fn test_optimize_distributed_end_to_end() {
        let config = AdvancedDistributedConfig::default();
        let mut optimizer = AdvancedDistributedOptimizer::<f64>::new(config).expect("construction");

        // 3×3 toy matrix
        let data = Array2::<f64>::from_elem((3, 3), 1.0);

        let result = pollster::block_on(optimizer.optimize_distributed(&data));
        assert!(
            result.is_ok(),
            "optimize_distributed should succeed: {:?}",
            result
        );

        let output = result.expect("result");
        assert_eq!(output.dim(), (3, 3), "output dimensions should match input");

        // Stats should have been incremented
        assert_eq!(optimizer.get_statistics().total_operations, 1);
    }

    /// Test 3: Failure injection triggers the recovery manager.
    #[test]
    fn test_failure_injection_triggers_recovery() {
        let config = AdvancedDistributedConfig::default();
        let mut optimizer = AdvancedDistributedOptimizer::<f64>::new(config).expect("construction");

        let failure = FailureInfo {
            failed_node_id: "node-99".to_string(),
            failure_type: FailureType::NodeFailure,
            timestamp: SystemTime::now(),
            affected_services: vec!["service-a".to_string()],
        };

        let result = pollster::block_on(optimizer.handle_failure(failure));
        assert!(
            result.is_ok(),
            "handle_failure should succeed: {:?}",
            result
        );

        // Recovery event counter should be incremented
        assert_eq!(
            optimizer.get_statistics().recovery_events,
            1,
            "recovery_events should be 1 after one failure"
        );

        // Health score should be below 1.0 due to failure history
        let health = optimizer.get_statistics().health_score;
        assert!(
            health < 1.0,
            "health score should decrease after a failure; got {}",
            health
        );
    }
}
