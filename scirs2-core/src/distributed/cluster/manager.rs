//! Main cluster manager implementation
//!
//! This module provides the core ClusterManager implementation that
//! orchestrates all cluster management functionality.

use crate::error::{CoreError, CoreResult, ErrorContext, ErrorLocation};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use super::allocator::ResourceAllocator;
use super::coordination::ClusterCoordination;
use super::events::ClusterEventLog;
use super::health::HealthMonitor;
use super::registry::NodeRegistry;
use super::state::ClusterState;
use super::types::{
    ClusterConfiguration, ClusterHealth, ClusterHealthStatus, ClusterStatistics, ComputeCapacity,
    DistributedTask, ExecutionPlan, ExecutionStatus, NodeInfo, NodeStatus, ResourceUtilization,
    TaskId,
};

/// Global cluster manager instance
static GLOBAL_CLUSTER_MANAGER: std::sync::OnceLock<Arc<ClusterManager>> =
    std::sync::OnceLock::new();

/// Comprehensive cluster management system
#[derive(Debug)]
pub struct ClusterManager {
    cluster_state: Arc<RwLock<ClusterState>>,
    node_registry: Arc<RwLock<NodeRegistry>>,
    healthmonitor: Arc<Mutex<HealthMonitor>>,
    resource_allocator: Arc<RwLock<ResourceAllocator>>,
    configuration: Arc<RwLock<ClusterConfiguration>>,
    eventlog: Arc<Mutex<ClusterEventLog>>,
}

#[allow(dead_code)]
impl ClusterManager {
    /// Create new cluster manager
    pub fn new(config: ClusterConfiguration) -> CoreResult<Self> {
        Ok(Self {
            cluster_state: Arc::new(RwLock::new(ClusterState::new())),
            node_registry: Arc::new(RwLock::new(NodeRegistry::new())),
            healthmonitor: Arc::new(Mutex::new(HealthMonitor::new()?)),
            resource_allocator: Arc::new(RwLock::new(ResourceAllocator::new())),
            configuration: Arc::new(RwLock::new(config)),
            eventlog: Arc::new(Mutex::new(ClusterEventLog::new())),
        })
    }

    /// Get global cluster manager instance
    pub fn global() -> CoreResult<Arc<Self>> {
        Ok(GLOBAL_CLUSTER_MANAGER
            .get_or_init(|| Arc::new(Self::new(ClusterConfiguration::default()).unwrap()))
            .clone())
    }

    /// Start cluster management services
    pub fn start(&self) -> CoreResult<()> {
        // Start node discovery
        self.start_node_discovery()?;

        // Start health monitoring
        self.start_health_monitoring()?;

        // Start resource management
        self.start_resource_management()?;

        // Start cluster coordination
        self.start_cluster_coordination()?;

        Ok(())
    }

    /// Start the node discovery background thread
    fn start_node_discovery(&self) -> CoreResult<()> {
        let registry = self.node_registry.clone();
        let config = self.configuration.clone();
        let eventlog = self.eventlog.clone();

        thread::spawn(move || loop {
            if let Err(e) = ClusterCoordination::node_discovery_loop(&registry, &config, &eventlog)
            {
                eprintln!("Node discovery error: {e:?}");
            }
            thread::sleep(Duration::from_secs(30));
        });

        Ok(())
    }

    /// Start the health monitoring background thread
    fn start_health_monitoring(&self) -> CoreResult<()> {
        let healthmonitor = self.healthmonitor.clone();
        let registry = self.node_registry.clone();
        let eventlog = self.eventlog.clone();

        thread::spawn(move || loop {
            if let Err(e) =
                ClusterCoordination::health_monitoring_loop(&healthmonitor, &registry, &eventlog)
            {
                eprintln!("Health monitoring error: {e:?}");
            }
            thread::sleep(Duration::from_secs(10));
        });

        Ok(())
    }

    /// Start the resource management background thread
    fn start_resource_management(&self) -> CoreResult<()> {
        let allocator = self.resource_allocator.clone();
        let registry = self.node_registry.clone();

        thread::spawn(move || loop {
            if let Err(e) = ClusterCoordination::resource_management_loop(&allocator, &registry) {
                eprintln!("Resource management error: {e:?}");
            }
            thread::sleep(Duration::from_secs(15));
        });

        Ok(())
    }

    /// Start the cluster coordination background thread
    fn start_cluster_coordination(&self) -> CoreResult<()> {
        let cluster_state = self.cluster_state.clone();
        let registry = self.node_registry.clone();
        let eventlog = self.eventlog.clone();

        thread::spawn(move || loop {
            if let Err(e) =
                ClusterCoordination::cluster_coordination_loop(&cluster_state, &registry, &eventlog)
            {
                eprintln!("Cluster coordination error: {e:?}");
            }
            thread::sleep(Duration::from_secs(5));
        });

        Ok(())
    }

    /// Register a new node in the cluster
    pub fn register_node(&self, nodeinfo: NodeInfo) -> CoreResult<()> {
        let mut registry = self.node_registry.write().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        registry.register_node(nodeinfo)?;
        Ok(())
    }

    /// Get cluster health status
    pub fn get_health(&self) -> CoreResult<ClusterHealth> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let all_nodes = registry.get_all_nodes();
        let healthy_nodes = all_nodes
            .iter()
            .filter(|n| n.status == NodeStatus::Healthy)
            .count();
        let total_nodes = all_nodes.len();

        let health_percentage = if total_nodes == 0 {
            100.0
        } else {
            (healthy_nodes as f64 / total_nodes as f64) * 100.0
        };

        let status = if health_percentage >= 80.0 {
            ClusterHealthStatus::Healthy
        } else if health_percentage >= 50.0 {
            ClusterHealthStatus::Degraded
        } else {
            ClusterHealthStatus::Unhealthy
        };

        Ok(ClusterHealth {
            status,
            healthy_nodes,
            total_nodes,
            health_percentage,
            last_updated: Instant::now(),
        })
    }

    /// Get list of active nodes
    pub fn get_active_nodes(&self) -> CoreResult<Vec<NodeInfo>> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        Ok(registry.get_healthy_nodes())
    }

    /// Get available nodes (returns nodeid -> nodeinfo mapping)
    pub fn get_available_nodes(&self) -> CoreResult<HashMap<String, NodeInfo>> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let nodes = registry.get_healthy_nodes();
        let mut node_map = HashMap::new();
        for node in nodes {
            node_map.insert(node.id.clone(), node);
        }
        Ok(node_map)
    }

    /// Get total cluster compute capacity
    pub fn get_total_capacity(&self) -> CoreResult<ComputeCapacity> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let nodes = registry.get_healthy_nodes();
        let mut total_capacity = ComputeCapacity::default();

        for node in nodes {
            total_capacity.cpu_cores += node.capabilities.cpu_cores;
            total_capacity.memory_gb += node.capabilities.memory_gb;
            total_capacity.gpu_count += node.capabilities.gpu_count;
            total_capacity.disk_space_gb += node.capabilities.disk_space_gb;
        }

        Ok(total_capacity)
    }

    /// Submit a distributed task to the cluster
    pub fn submit_task(&self, task: DistributedTask) -> CoreResult<TaskId> {
        let allocator = self.resource_allocator.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire allocator lock"))
        })?;

        let allocation = allocator.allocate_resources(&task.resource_requirements)?;

        // Create task execution plan
        let taskid = TaskId::generate();
        let _execution_plan = ExecutionPlan {
            taskid: taskid.clone(),
            task,
            node_allocation: allocation,
            created_at: Instant::now(),
            status: ExecutionStatus::Pending,
        };

        // Submit to scheduler (placeholder)
        // In a real implementation, this would go to the distributed scheduler
        Ok(taskid)
    }

    /// Get cluster statistics
    pub fn get_cluster_statistics(&self) -> CoreResult<ClusterStatistics> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(
                ErrorContext::new("Failed to acquire registry lock")
                    .with_location(ErrorLocation::new(file!(), line!())),
            )
        })?;

        let allocator = self.resource_allocator.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire allocator lock"))
        })?;

        let nodes = registry.get_all_nodes();
        let total_capacity = self.get_total_capacity()?;
        let available_capacity = (*allocator).available_capacity();

        Ok(ClusterStatistics {
            total_nodes: nodes.len(),
            healthy_nodes: nodes
                .iter()
                .filter(|n| n.status == NodeStatus::Healthy)
                .count(),
            total_capacity: total_capacity.clone(),
            available_capacity: available_capacity.clone(),
            resource_utilization: ResourceUtilization {
                cpu_utilization: 1.0
                    - (available_capacity.cpu_cores as f64 / total_capacity.cpu_cores as f64),
                memory_utilization: 1.0
                    - (available_capacity.memory_gb as f64 / total_capacity.memory_gb as f64),
                gpu_utilization: if total_capacity.gpu_count > 0 {
                    1.0 - (available_capacity.gpu_count as f64 / total_capacity.gpu_count as f64)
                } else {
                    0.0
                },
            },
        })
    }

    /// Stop the cluster manager and all background threads
    pub fn stop(&self) -> CoreResult<()> {
        // In a real implementation, this would signal the background threads to stop
        // For now, it's a placeholder
        Ok(())
    }

    /// Get cluster configuration
    pub fn get_configuration(&self) -> CoreResult<ClusterConfiguration> {
        let config = self.configuration.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire config lock"))
        })?;
        Ok(config.clone())
    }

    /// Update cluster configuration
    pub fn update_configuration(&self, new_config: ClusterConfiguration) -> CoreResult<()> {
        let mut config = self.configuration.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire config lock"))
        })?;
        *config = new_config;
        Ok(())
    }

    /// Get cluster state
    pub fn get_cluster_state(&self) -> CoreResult<String> {
        let state = self.cluster_state.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire cluster state lock"))
        })?;

        if let Some(leader) = state.get_leader() {
            Ok(format!(
                "Leader: {leader}, Last updated: {:?}",
                state.last_updated()
            ))
        } else {
            Ok("No leader elected".to_string())
        }
    }

    /// Force leader election
    pub fn force_leader_election(&self) -> CoreResult<Option<String>> {
        let registry = self.node_registry.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
        })?;

        let healthy_nodes = registry.get_healthy_nodes();
        ClusterCoordination::elect_leader(&healthy_nodes)
    }

    /// Remove a node from the cluster
    pub fn remove_node(&self, node_id: &str) -> CoreResult<()> {
        ClusterCoordination::handle_node_removal(node_id, &self.node_registry, &self.eventlog)
    }

    /// Gracefully shutdown a node
    pub fn shutdown_node(&self, node_id: &str) -> CoreResult<()> {
        ClusterCoordination::handle_node_shutdown(node_id, &self.node_registry, &self.eventlog)
    }
}

/// Initialize cluster manager with default configuration
#[allow(dead_code)]
pub fn initialize_cluster_manager() -> CoreResult<()> {
    let cluster_manager = ClusterManager::global()?;
    cluster_manager.start()?;
    Ok(())
}
