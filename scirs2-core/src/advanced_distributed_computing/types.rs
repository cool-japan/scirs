//! Core types and configurations for distributed computing
//!
//! This module contains shared types, enums, and configuration structures
//! used throughout the distributed computing framework.

use crate::distributed::NodeType;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// Helper function for serde default
#[allow(dead_code)]
pub fn default_instant() -> Instant {
    Instant::now()
}

/// Configuration for distributed computing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedComputingConfig {
    /// Enable automatic node discovery
    pub enable_auto_discovery: bool,
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Enable fault tolerance
    pub enable_fault_tolerance: bool,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// Heartbeat interval (milliseconds)
    pub heartbeat_interval_ms: u64,
    /// Task timeout (seconds)
    pub task_timeout_seconds: u64,
    /// Communication timeout (milliseconds)
    pub communication_timeout_ms: u64,
    /// Enable encryption
    pub enable_encryption: bool,
    /// Enable compression
    pub enable_compression: bool,
    /// Cluster discovery port
    pub discovery_port: u16,
    /// Communication port range
    pub communication_port_range: (u16, u16),
    /// Node failure detection threshold
    pub failure_detection_threshold: u32,
    /// Enable elastic scaling
    pub enable_elastic_scaling: bool,
}

impl Default for DistributedComputingConfig {
    fn default() -> Self {
        Self {
            enable_auto_discovery: true,
            enable_load_balancing: true,
            enable_fault_tolerance: true,
            max_nodes: 256,
            heartbeat_interval_ms: 5000,
            task_timeout_seconds: 300,
            communication_timeout_ms: 10000,
            enable_encryption: true,
            enable_compression: true,
            discovery_port: 9090,
            communication_port_range: (9100, 9200),
            failure_detection_threshold: 3,
            enable_elastic_scaling: true,
        }
    }
}

/// Configuration for fault tolerance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaultToleranceConfig {
    /// Enable predictive failure detection
    pub enable_predictive_detection: bool,
    /// Enable automatic recovery
    pub enable_automatic_recovery: bool,
    /// Recovery timeout in seconds
    pub recoverytimeout_seconds: u64,
    /// Checkpoint frequency in seconds
    pub checkpoint_frequency_seconds: u64,
    /// Maximum retries for failed tasks
    pub maxretries: u32,
    /// Fault tolerance level
    pub level: FaultToleranceLevel,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
}

impl Default for FaultToleranceConfig {
    fn default() -> Self {
        Self {
            enable_predictive_detection: true,
            enable_automatic_recovery: true,
            recoverytimeout_seconds: 300,
            checkpoint_frequency_seconds: 60,
            maxretries: 3,
            level: FaultToleranceLevel::default(),
            checkpoint_interval: Duration::from_secs(60),
        }
    }
}

/// Requirements specification for distributed tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequirements {
    /// Minimum CPU cores required
    pub min_cpu_cores: u32,
    /// Minimum memory in GB
    pub min_memory_gb: f64,
    /// Minimum GPU memory in GB (if GPU required)
    pub min_gpu_memory_gb: Option<f64>,
    /// Required node type
    pub required_node_type: Option<NodeType>,
    /// Network bandwidth requirements in Mbps
    pub min_networkbandwidth_mbps: f64,
    /// Storage requirements in GB
    pub min_storage_gb: f64,
    /// Geographic constraints
    pub geographic_constraints: Vec<String>,
    /// Compute complexity level
    pub compute_complexity: f64,
    /// Memory intensity level
    pub memory_intensity: f64,
    /// I/O requirements
    pub io_requirements: f64,
}

impl Default for TaskRequirements {
    fn default() -> Self {
        Self {
            min_cpu_cores: 1,
            min_memory_gb: 1.0,
            min_gpu_memory_gb: None,
            required_node_type: None,
            min_networkbandwidth_mbps: 100.0,
            min_storage_gb: 10.0,
            geographic_constraints: Vec::new(),
            compute_complexity: 0.5,
            memory_intensity: 0.5,
            io_requirements: 0.5,
        }
    }
}

/// Distribution strategy for distributed tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionStrategy {
    DataParallel,
    ModelParallel,
    PipelineParallel,
    Independent,
}

impl Default for DistributionStrategy {
    fn default() -> Self {
        Self::DataParallel
    }
}

/// Fault tolerance level for tasks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FaultToleranceLevel {
    None,
    Basic,
    Standard,
    High,
    Critical,
}

impl Default for FaultToleranceLevel {
    fn default() -> Self {
        Self::Standard
    }
}

/// Resource analysis for determining optimal resource profile
#[derive(Debug, Clone)]
pub struct ResourceAnalysis {
    pub cpu_cores: usize,
    pub memory_gb: usize,
    pub gpu_required: bool,
    pub network_intensive: bool,
    pub storage_intensive: bool,
}

/// Resource profile for grouping tasks by requirements
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceProfile {
    LowMemoryLowCpu,
    LowMemoryHighCpu,
    HighMemoryLowCpu,
    HighMemoryHighCpu,
    GpuAccelerated,
    NetworkIntensive,
    StorageIntensive,
}

impl Default for ResourceProfile {
    fn default() -> Self {
        Self::LowMemoryLowCpu
    }
}

impl ResourceProfile {
    pub fn from_analysis(analysis: &ResourceAnalysis) -> Self {
        // Determine resource profile based on _analysis
        if analysis.gpu_required {
            Self::GpuAccelerated
        } else if analysis.network_intensive {
            Self::NetworkIntensive
        } else if analysis.storage_intensive {
            Self::StorageIntensive
        } else if analysis.memory_gb > 16 && analysis.cpu_cores > 8 {
            Self::HighMemoryHighCpu
        } else if analysis.memory_gb > 16 {
            Self::HighMemoryLowCpu
        } else if analysis.cpu_cores > 8 {
            Self::LowMemoryHighCpu
        } else {
            Self::LowMemoryLowCpu
        }
    }
}
