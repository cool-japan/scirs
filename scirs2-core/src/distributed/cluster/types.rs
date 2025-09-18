//! Type definitions for cluster management
//!
//! This module contains all the core types, enums, and data structures
//! used throughout the cluster management system.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

// ================================================================================================
// Configuration Types
// ================================================================================================

#[derive(Debug, Clone)]
pub struct ClusterConfiguration {
    pub auto_discovery_enabled: bool,
    pub discovery_methods: Vec<NodeDiscoveryMethod>,
    pub health_check_interval: Duration,
    pub leadership_timeout: Duration,
    pub resource_allocation_strategy: AllocationStrategy,
    pub max_nodes: Option<usize>,
}

impl Default for ClusterConfiguration {
    fn default() -> Self {
        Self {
            auto_discovery_enabled: true,
            discovery_methods: vec![NodeDiscoveryMethod::Static(vec![])],
            health_check_interval: Duration::from_secs(30),
            leadership_timeout: Duration::from_secs(300),
            resource_allocation_strategy: AllocationStrategy::FirstFit,
            max_nodes: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum NodeDiscoveryMethod {
    Static(Vec<SocketAddr>),
    Multicast { group: IpAddr, port: u16 },
    DnsService { service_name: String },
    Consul { endpoint: String },
}

// ================================================================================================
// Node Types
// ================================================================================================

#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub id: String,
    pub address: SocketAddr,
    pub node_type: NodeType,
    pub capabilities: NodeCapabilities,
    pub status: NodeStatus,
    pub last_seen: Instant,
    pub metadata: NodeMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    Master,
    Worker,
    Storage,
    Compute,
    ComputeOptimized,
    MemoryOptimized,
    StorageOptimized,
    General,
}

#[derive(Debug, Clone)]
pub struct NodeCapabilities {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub gpu_count: usize,
    pub disk_space_gb: usize,
    pub networkbandwidth_gbps: f64,
    pub specialized_units: Vec<SpecializedUnit>,
}

impl Default for NodeCapabilities {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_gb: 8,
            gpu_count: 0,
            disk_space_gb: 100,
            networkbandwidth_gbps: 1.0f64,
            specialized_units: Vec::new(),
        }
    }
}

/// Specialized computing units available on a node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpecializedUnit {
    TensorCore,
    QuantumProcessor,
    VectorUnit,
    CryptoAccelerator,
    NeuralProcessingUnit,
    Fpga,
    Asic,
    CustomAsic(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    Unknown,
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
    Draining,
}

#[derive(Debug, Clone)]
pub struct NodeMetadata {
    pub hostname: String,
    pub operating_system: String,
    pub kernel_version: String,
    pub container_runtime: Option<String>,
    pub labels: HashMap<String, String>,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            hostname: "unknown".to_string(),
            operating_system: "unknown".to_string(),
            kernel_version: "unknown".to_string(),
            container_runtime: None,
            labels: HashMap::new(),
        }
    }
}

// ================================================================================================
// Cluster Topology Types
// ================================================================================================

#[derive(Debug, Clone)]
pub struct ClusterTopology {
    pub zones: BTreeMap<String, Zone>,
    pub network_topology: NetworkTopology,
}

impl Default for ClusterTopology {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterTopology {
    pub fn new() -> Self {
        Self {
            zones: BTreeMap::new(),
            network_topology: NetworkTopology::Flat,
        }
    }

    pub fn update(&mut self, nodes: &[NodeInfo]) {
        // Simple topology update - group nodes by network zone
        self.zones.clear();

        for node in nodes {
            let zone_name = self.determine_zone(&node.address);
            let zone = self.zones.entry(zone_name).or_default();
            zone.add_node(node.clone());
        }
    }

    fn determine_zone(&self, address: &SocketAddr) -> String {
        // Simple zone determination based on IP address
        // In a real implementation, this would use proper network topology discovery
        format!(
            "zone_{}",
            address.ip().to_string().split('.').next().unwrap_or("0")
        )
    }

    /// Update the topology model with new node information
    pub fn update_model(&mut self, nodes: &[NodeInfo]) {
        // Update the topology model based on new node information
        self.update(nodes);

        // Additional model updates can be added here
        // For example, network latency measurements, bandwidth tests, etc.
    }
}

#[derive(Debug, Clone)]
pub struct Zone {
    pub nodes: Vec<NodeInfo>,
    pub capacity: ComputeCapacity,
}

impl Default for Zone {
    fn default() -> Self {
        Self::new()
    }
}

impl Zone {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            capacity: ComputeCapacity::default(),
        }
    }

    pub fn add_node(&mut self, node: NodeInfo) {
        self.capacity.cpu_cores += node.capabilities.cpu_cores;
        self.capacity.memory_gb += node.capabilities.memory_gb;
        self.capacity.gpu_count += node.capabilities.gpu_count;
        self.capacity.disk_space_gb += node.capabilities.disk_space_gb;

        self.nodes.push(node);
    }
}

#[derive(Debug, Clone)]
pub enum NetworkTopology {
    Flat,
    Hierarchical,
    Mesh,
    Ring,
}

// ================================================================================================
// Health Monitoring Types
// ================================================================================================

#[derive(Debug, Clone)]
pub struct NodeHealthStatus {
    pub status: NodeStatus,
    pub health_score: f64,
    pub failing_checks: Vec<HealthCheck>,
    pub last_checked: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthCheck {
    Ping,
    CpuLoad,
    MemoryUsage,
    DiskSpace,
    NetworkConnectivity,
}

#[derive(Debug)]
pub struct HealthCheckResult {
    pub is_healthy: bool,
    pub impact_score: f64,
    pub details: String,
}

#[derive(Debug, Clone)]
pub struct ClusterHealth {
    pub status: ClusterHealthStatus,
    pub healthy_nodes: usize,
    pub total_nodes: usize,
    pub health_percentage: f64,
    pub last_updated: Instant,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

// ================================================================================================
// Resource Management Types
// ================================================================================================

#[derive(Debug, Clone, Default)]
pub struct ComputeCapacity {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub gpu_count: usize,
    pub disk_space_gb: usize,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub gpu_count: usize,
    pub disk_space_gb: usize,
    pub specialized_requirements: Vec<SpecializedRequirement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpecializedRequirement {
    pub unit_type: SpecializedUnit,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub allocation_id: AllocationId,
    pub allocated_resources: ComputeCapacity,
    pub assigned_nodes: Vec<String>,
    pub created_at: Instant,
    pub expires_at: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AllocationId(String);

impl AllocationId {
    pub fn generate() -> Self {
        Self(format!(
            "alloc_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    LoadBalanced,
}

// ================================================================================================
// Task Management Types
// ================================================================================================

#[derive(Debug, Clone)]
pub struct DistributedTask {
    pub taskid: TaskId,
    pub task_type: TaskType,
    pub resource_requirements: ResourceRequirements,
    pub data_dependencies: Vec<DataDependency>,
    pub execution_parameters: TaskParameters,
    pub priority: TaskPriority,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TaskId(String);

impl TaskId {
    pub fn generate() -> Self {
        Self(format!(
            "task_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()
        ))
    }
}

#[derive(Debug, Clone)]
pub enum TaskType {
    Computation,
    DataProcessing,
    MachineLearning,
    Simulation,
    Analysis,
}

#[derive(Debug, Clone)]
pub struct DataDependency {
    pub data_id: String,
    pub access_type: DataAccessType,
    pub size_hint: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataAccessType {
    Read,
    Write,
    ReadWrite,
}

#[derive(Debug, Clone)]
pub struct TaskParameters {
    pub environment_variables: HashMap<String, String>,
    pub command_arguments: Vec<String>,
    pub timeout: Option<Duration>,
    pub retrypolicy: RetryPolicy,
}

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: usize,
    pub backoff_strategy: BackoffStrategy,
}

#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    Fixed(Duration),
    Linear(Duration),
    Exponential { base: Duration, multiplier: f64 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub taskid: TaskId,
    pub task: DistributedTask,
    pub node_allocation: ResourceAllocation,
    pub created_at: Instant,
    pub status: ExecutionStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Pending,
    Scheduled,
    Running,
    Completed,
    Failed,
    Cancelled,
}

// ================================================================================================
// Event Types
// ================================================================================================

#[derive(Debug, Clone)]
pub enum ClusterEvent {
    NodeDiscovered {
        nodeid: String,
        address: SocketAddr,
        timestamp: Instant,
    },
    NodeStatusChanged {
        nodeid: String,
        old_status: NodeStatus,
        new_status: NodeStatus,
        timestamp: Instant,
    },
    LeaderElected {
        nodeid: String,
        timestamp: Instant,
    },
    TaskScheduled {
        taskid: TaskId,
        nodeid: String,
        timestamp: Instant,
    },
    TaskCompleted {
        taskid: TaskId,
        nodeid: String,
        execution_time: Duration,
        timestamp: Instant,
    },
    ResourceAllocation {
        allocation_id: AllocationId,
        resources: ComputeCapacity,
        timestamp: Instant,
    },
}

// ================================================================================================
// Statistics Types
// ================================================================================================

#[derive(Debug, Clone)]
pub struct ClusterStatistics {
    pub total_nodes: usize,
    pub healthy_nodes: usize,
    pub total_capacity: ComputeCapacity,
    pub available_capacity: ComputeCapacity,
    pub resource_utilization: ResourceUtilization,
}

#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
}
