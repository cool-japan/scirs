//! Advanced distributed computing with modular architecture
//!
//! This module provides comprehensive distributed computing capabilities organized
//! into focused submodules:
//!
//! - `config`: Configuration structures for all distributed components
//! - `consensus`: Consensus algorithms (Raft, PBFT, etc.)
//! - `sharding`: Data sharding and distribution management
//! - `fault_tolerance`: Fault tolerance and recovery management
//! - `scaling`: Auto-scaling and cluster management
//! - `locality`: Data locality optimization
//! - `optimization`: Performance optimization
//! - `scheduling`: Task scheduling and coordination
//! - `monitoring`: Performance monitoring and metrics collection

pub mod config;
pub mod consensus;
pub mod fault_tolerance;
pub mod sharding;
pub mod transport;

// Re-export main types for easier access
pub use config::*;
pub use consensus::{
    ConsensusFactory, ConsensusManager, PbftConsensus, RaftConsensus, SimpleMajorityConsensus,
};
pub use fault_tolerance::{
    FaultRecoveryManager, HealthMonitor, HealthSummary, NodeMetrics, RecoveryAction,
};
pub use sharding::{DataShard, ShardManager, ShardMigration, ShardingStats};

// Missing types referenced in mod.rs
/// Basic distributed configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub cluster_size: usize,
    pub node_id: String,
    pub timeout_ms: u64,
}

/// Distributed metrics builder
#[derive(Debug, Clone)]
pub struct DistributedMetricsBuilder {
    config: DistributedConfig,
}

impl DistributedMetricsBuilder {
    pub fn new(config: DistributedConfig) -> Self {
        Self { config }
    }
}

/// Distributed metrics coordinator
#[derive(Debug)]
pub struct DistributedMetricsCoordinator {
    config: DistributedConfig,
}

impl DistributedMetricsCoordinator {
    pub fn new(config: DistributedConfig) -> Self {
        Self { config }
    }
}

use crate::error::{MetricsError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Advanced distributed coordinator that integrates all distributed components
pub struct AdvancedDistributedCoordinator {
    /// Configuration
    config: AdvancedClusterConfig,
    /// Consensus manager
    #[allow(dead_code)]
    consensus: Option<Box<dyn ConsensusManager>>,
    /// Shard manager
    shard_manager: ShardManager,
    /// Fault recovery manager
    fault_manager: FaultRecoveryManager,
    /// Cluster state
    cluster_state: Arc<RwLock<ClusterState>>,
    /// Performance metrics
    performance_metrics: Arc<RwLock<ClusterPerformanceMetrics>>,
    /// Coordinator status
    status: CoordinatorStatus,
}

impl AdvancedDistributedCoordinator {
    /// Create a new advanced distributed coordinator
    pub fn new(config: AdvancedClusterConfig) -> Result<Self> {
        // Initialize shard manager
        let shard_manager = ShardManager::new(config.sharding_config.clone());

        // Initialize fault recovery manager
        let fault_manager = FaultRecoveryManager::new(config.fault_tolerance.clone());

        // Initialize cluster state
        let cluster_state = Arc::new(RwLock::new(ClusterState::new()));

        // Initialize performance metrics
        let performance_metrics = Arc::new(RwLock::new(ClusterPerformanceMetrics::new()));

        Ok(Self {
            config,
            consensus: None,
            shard_manager,
            fault_manager,
            cluster_state,
            performance_metrics,
            status: CoordinatorStatus::Stopped,
        })
    }

    /// Start the distributed coordinator
    pub fn start(&mut self, node_id: String, peers: Vec<String>) -> Result<()> {
        // Initialize consensus if configured
        if self.config.consensus_config.algorithm != ConsensusAlgorithm::None {
            let consensus = ConsensusFactory::create_consensus(
                self.config.consensus_config.algorithm.clone(),
                node_id.clone(),
                peers.clone(),
                self.config.consensus_config.clone(),
            )?;
            self.consensus = Some(consensus);

            if let Some(ref mut consensus) = self.consensus {
                consensus.start()?;
            }
        }

        // Initialize sharding
        self.shard_manager.initialize(peers.clone())?;

        // Start fault monitoring
        self.fault_manager.start()?;

        // Update cluster state
        {
            let mut state = self.cluster_state.write().expect("Operation failed");
            state.local_node_id = node_id;
            state.cluster_size = peers.len() + 1; // +1 for local node
            state.status = ClusterStatus::Active;
            state.last_updated = SystemTime::now();
        }

        self.status = CoordinatorStatus::Running;

        Ok(())
    }

    /// Stop the distributed coordinator
    pub fn stop(&mut self) -> Result<()> {
        // Stop fault monitoring
        self.fault_manager.stop()?;

        // Update status
        self.status = CoordinatorStatus::Stopped;

        // Update cluster state
        {
            let mut state = self.cluster_state.write().expect("Operation failed");
            state.status = ClusterStatus::Stopped;
            state.last_updated = SystemTime::now();
        }

        Ok(())
    }

    /// Submit data for consensus
    pub fn submit_consensus(&mut self, data: Vec<u8>) -> Result<String> {
        if let Some(ref mut consensus) = self.consensus {
            consensus.propose(data)
        } else {
            Err(MetricsError::ConsensusError(
                "Consensus not initialized".to_string(),
            ))
        }
    }

    /// Get consensus state
    pub fn get_consensus_state(&self) -> Option<consensus::ConsensusState> {
        self.consensus.as_ref().map(|c| c.get_state())
    }

    /// Find shard for a key
    pub fn find_shard(&self, key: &str) -> Result<String> {
        self.shard_manager.find_shard(key)
    }

    /// Get node responsible for a key
    pub fn get_node_for_key(&self, key: &str) -> Result<String> {
        self.shard_manager.get_node_for_key(key)
    }

    /// Add a new node to the cluster
    pub fn add_node(&mut self, node_id: String) -> Result<()> {
        // Add to sharding
        self.shard_manager.add_node(node_id.clone())?;

        // Register for health monitoring
        let metrics = NodeMetrics::healthy();
        self.fault_manager.register_node(node_id.clone(), metrics)?;

        // Update cluster state
        {
            let mut state = self.cluster_state.write().expect("Operation failed");
            state.cluster_size += 1;
            state.last_updated = SystemTime::now();
        }

        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: &str) -> Result<()> {
        // Remove from sharding
        self.shard_manager.remove_node(node_id)?;

        // Unregister from health monitoring
        self.fault_manager.unregister_node(node_id)?;

        // Update cluster state
        {
            let mut state = self.cluster_state.write().expect("Operation failed");
            state.cluster_size = state.cluster_size.saturating_sub(1);
            state.last_updated = SystemTime::now();
        }

        Ok(())
    }

    /// Update node metrics
    pub fn update_node_metrics(&mut self, node_id: &str, metrics: NodeMetrics) -> Result<()> {
        self.fault_manager.update_node_metrics(node_id, metrics)
    }

    /// Get cluster health summary
    pub fn get_health_summary(&self) -> HealthSummary {
        self.fault_manager.get_health_summary()
    }

    /// Get sharding statistics
    pub fn get_sharding_stats(&self) -> ShardingStats {
        self.shard_manager.get_stats()
    }

    /// Get cluster state
    pub fn get_cluster_state(&self) -> ClusterState {
        let state = self.cluster_state.read().expect("Operation failed");
        state.clone()
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> ClusterPerformanceMetrics {
        let metrics = self.performance_metrics.read().expect("Operation failed");
        metrics.clone()
    }

    /// Get coordinator status
    pub fn get_status(&self) -> CoordinatorStatus {
        self.status.clone()
    }

    /// Migrate shard to different node
    pub fn migrate_shard(&mut self, shard_id: &str, target_node: Option<String>) -> Result<String> {
        self.shard_manager.migrate_shard(shard_id, target_node)
    }

    /// Process recovery actions
    pub fn process_recovery_actions(&mut self) -> Result<Vec<RecoveryAction>> {
        Ok(self.fault_manager.get_recovery_history())
    }

    /// Update cluster performance metrics
    pub fn update_performance_metrics(&mut self, metrics: ClusterPerformanceMetrics) {
        let mut perf_metrics = self.performance_metrics.write().expect("Operation failed");
        *perf_metrics = metrics;
    }

    /// Get active recovery operations
    pub fn get_active_recoveries(&self) -> Vec<fault_tolerance::RecoveryOperation> {
        self.fault_manager.get_active_recoveries()
    }

    /// List all shards
    pub fn list_shards(&self) -> Vec<DataShard> {
        self.shard_manager.list_shards()
    }

    /// Get shard by ID
    pub fn get_shard(&self, shard_id: &str) -> Option<DataShard> {
        self.shard_manager.get_shard(shard_id)
    }

    /// Update shard statistics
    pub fn update_shard_stats(
        &mut self,
        shard_id: &str,
        size_bytes: u64,
        key_count: usize,
    ) -> Result<()> {
        self.shard_manager
            .update_shard_stats(shard_id, size_bytes, key_count)
    }
}

/// Coordinator status
#[derive(Debug, Clone, PartialEq)]
pub enum CoordinatorStatus {
    /// Coordinator is stopped
    Stopped,
    /// Coordinator is starting up
    Starting,
    /// Coordinator is running normally
    Running,
    /// Coordinator is stopping
    Stopping,
    /// Coordinator is in error state
    Error(String),
}

/// Cluster state information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    /// Local node ID
    pub local_node_id: String,
    /// Total cluster size
    pub cluster_size: usize,
    /// Cluster status
    pub status: ClusterStatus,
    /// Last state update
    pub last_updated: SystemTime,
    /// Node information
    pub nodes: HashMap<String, NodeInfo>,
    /// Active tasks
    pub active_tasks: usize,
    /// Configuration version
    pub config_version: u64,
}

impl ClusterState {
    /// Create new cluster state
    pub fn new() -> Self {
        Self {
            local_node_id: String::new(),
            cluster_size: 0,
            status: ClusterStatus::Stopped,
            last_updated: SystemTime::now(),
            nodes: HashMap::new(),
            active_tasks: 0,
            config_version: 1,
        }
    }

    /// Add node information
    pub fn add_node(&mut self, node_id: String, info: NodeInfo) {
        self.nodes.insert(node_id, info);
        self.last_updated = SystemTime::now();
    }

    /// Remove node information
    pub fn remove_node(&mut self, node_id: &str) {
        self.nodes.remove(node_id);
        self.last_updated = SystemTime::now();
    }

    /// Update node information
    pub fn update_node(&mut self, node_id: &str, info: NodeInfo) {
        if self.nodes.contains_key(node_id) {
            self.nodes.insert(node_id.to_string(), info);
            self.last_updated = SystemTime::now();
        }
    }

    /// Get healthy node count
    pub fn healthy_node_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|node| node.status == NodeStatus::Healthy)
            .count()
    }
}

impl Default for ClusterState {
    fn default() -> Self {
        Self::new()
    }
}

/// Cluster status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ClusterStatus {
    /// Cluster is stopped
    Stopped,
    /// Cluster is starting
    Starting,
    /// Cluster is active and healthy
    Active,
    /// Cluster is degraded but functional
    Degraded,
    /// Cluster has failed
    Failed,
    /// Cluster is in maintenance mode
    Maintenance,
}

/// Node information in cluster state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node ID
    pub id: String,
    /// Node address
    pub address: Option<String>,
    /// Node status
    pub status: NodeStatus,
    /// Node role
    pub role: NodeRole,
    /// Resource information
    pub resources: ResourceInfo,
    /// Last seen timestamp
    pub last_seen: SystemTime,
    /// Node metadata
    pub metadata: HashMap<String, String>,
}

/// Node status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeStatus {
    /// Node is healthy
    Healthy,
    /// Node is degraded
    Degraded,
    /// Node has failed
    Failed,
    /// Node is unknown
    Unknown,
    /// Node is in maintenance
    Maintenance,
}

/// Node role in the cluster
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NodeRole {
    /// Master/leader node
    Master,
    /// Worker node
    Worker,
    /// Standby node
    Standby,
    /// Storage node
    Storage,
    /// Compute node
    Compute,
    /// Gateway node
    Gateway,
    /// Mixed role node
    Mixed(Vec<String>),
}

/// Resource information for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceInfo {
    /// CPU cores available
    pub cpu_cores: f64,
    /// Memory in GB
    pub memory_gb: f64,
    /// Storage in GB
    pub storage_gb: f64,
    /// Network bandwidth in Gbps
    pub network_gbps: f64,
    /// GPU information
    pub gpu_info: Option<GpuInfo>,
    /// Custom resources
    pub custom_resources: HashMap<String, f64>,
}

/// GPU resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU model
    pub model: String,
    /// GPU memory in GB
    pub memory_gb: f64,
    /// Number of GPU cores
    pub cores: usize,
    /// GPU utilization percentage
    pub utilization: f64,
}

/// Cluster performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterPerformanceMetrics {
    /// Total throughput (operations/second)
    pub throughput: f64,
    /// Average latency (milliseconds)
    pub latency_ms: f64,
    /// Error rate (0-1)
    pub error_rate: f64,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
    /// Network statistics
    pub network_stats: NetworkStats,
    /// Storage statistics
    pub storage_stats: StorageStats,
    /// Last updated timestamp
    pub last_updated: SystemTime,
}

impl ClusterPerformanceMetrics {
    /// Create new performance metrics
    pub fn new() -> Self {
        Self {
            throughput: 0.0,
            latency_ms: 0.0,
            error_rate: 0.0,
            resource_utilization: ResourceUtilization::default(),
            network_stats: NetworkStats::default(),
            storage_stats: StorageStats::default(),
            last_updated: SystemTime::now(),
        }
    }
}

impl Default for ClusterPerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Resource utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Average CPU utilization across cluster
    pub cpu_percent: f64,
    /// Average memory utilization across cluster
    pub memory_percent: f64,
    /// Average storage utilization across cluster
    pub storage_percent: f64,
    /// Average network utilization across cluster
    pub network_percent: f64,
    /// GPU utilization (if available)
    pub gpu_percent: Option<f64>,
}

impl Default for ResourceUtilization {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,
            memory_percent: 0.0,
            storage_percent: 0.0,
            network_percent: 0.0,
            gpu_percent: None,
        }
    }
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total bytes sent
    pub bytes_sent: u64,
    /// Total bytes received
    pub bytes_received: u64,
    /// Packets sent
    pub packets_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Network errors
    pub errors: u64,
    /// Average bandwidth utilization (Mbps)
    pub bandwidth_mbps: f64,
}

impl Default for NetworkStats {
    fn default() -> Self {
        Self {
            bytes_sent: 0,
            bytes_received: 0,
            packets_sent: 0,
            packets_received: 0,
            errors: 0,
            bandwidth_mbps: 0.0,
        }
    }
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageStats {
    /// Total reads
    pub reads: u64,
    /// Total writes
    pub writes: u64,
    /// Bytes read
    pub bytes_read: u64,
    /// Bytes written
    pub bytes_written: u64,
    /// Storage errors
    pub errors: u64,
    /// Average IOPS
    pub iops: f64,
}

impl Default for StorageStats {
    fn default() -> Self {
        Self {
            reads: 0,
            writes: 0,
            bytes_read: 0,
            bytes_written: 0,
            errors: 0,
            iops: 0.0,
        }
    }
}

/// Serde module for Duration serialization
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_u64(duration.as_millis() as u64)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let millis = u64::deserialize(deserializer)?;
        Ok(Duration::from_millis(millis))
    }
}

/// Metrics collector placeholder for compatibility
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    // Implementation details
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_cluster_config_creation() {
        let config = AdvancedClusterConfig::default();
        assert!(config.consensus_config.quorum_size > 0);
        assert!(config.sharding_config.shard_count > 0);
        assert!(config.optimization_config.enabled);
    }

    #[test]
    fn test_distributed_coordinator_creation() {
        let config = AdvancedClusterConfig::default();
        let coordinator = AdvancedDistributedCoordinator::new(config);
        assert!(coordinator.is_ok());

        let coordinator = coordinator.expect("Operation failed");
        assert_eq!(coordinator.get_status(), CoordinatorStatus::Stopped);
    }

    #[test]
    fn test_cluster_state_operations() {
        let mut state = ClusterState::new();
        assert_eq!(state.cluster_size, 0);

        let node_info = NodeInfo {
            id: "node1".to_string(),
            address: Some("localhost:8080".to_string()),
            status: NodeStatus::Healthy,
            role: NodeRole::Worker,
            resources: ResourceInfo {
                cpu_cores: 4.0,
                memory_gb: 16.0,
                storage_gb: 100.0,
                network_gbps: 1.0,
                gpu_info: None,
                custom_resources: HashMap::new(),
            },
            last_seen: SystemTime::now(),
            metadata: HashMap::new(),
        };

        state.add_node("node1".to_string(), node_info);
        assert_eq!(state.nodes.len(), 1);
        assert_eq!(state.healthy_node_count(), 1);
    }

    #[test]
    fn test_cluster_performance_metrics() {
        let metrics = ClusterPerformanceMetrics::new();
        assert_eq!(metrics.throughput, 0.0);
        assert_eq!(metrics.error_rate, 0.0);
    }

    #[test]
    fn test_coordinator_start_stop() {
        let config = AdvancedClusterConfig::default();
        let mut coordinator =
            AdvancedDistributedCoordinator::new(config).expect("Operation failed");

        let nodes = vec!["node1".to_string(), "node2".to_string()];
        coordinator
            .start("node0".to_string(), nodes)
            .expect("Operation failed");
        assert_eq!(coordinator.get_status(), CoordinatorStatus::Running);

        coordinator.stop().expect("Operation failed");
        assert_eq!(coordinator.get_status(), CoordinatorStatus::Stopped);
    }
}
