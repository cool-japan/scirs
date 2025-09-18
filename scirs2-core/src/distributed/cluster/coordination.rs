//! Cluster coordination and background processing loops
//!
//! This module provides the coordination logic for cluster management
//! including discovery loops, health monitoring loops, resource management
//! loops, and leader election.

use crate::error::{CoreError, CoreResult, ErrorContext};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use super::allocator::ResourceAllocator;
use super::discovery::NodeDiscovery;
use super::events::ClusterEventLog;
use super::health::HealthMonitor;
use super::registry::NodeRegistry;
use super::state::ClusterState;
use super::types::{
    ClusterConfiguration, ClusterEvent, NodeCapabilities, NodeDiscoveryMethod, NodeInfo,
    NodeMetadata, NodeStatus, NodeType,
};

/// Cluster coordination utilities
pub struct ClusterCoordination;

impl ClusterCoordination {
    /// Run the node discovery loop
    pub fn node_discovery_loop(
        registry: &Arc<RwLock<NodeRegistry>>,
        config: &Arc<RwLock<ClusterConfiguration>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let config_read = config.read().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire config lock"))
        })?;

        if !config_read.auto_discovery_enabled {
            return Ok(());
        }

        // Discover nodes using configured methods
        for discovery_method in &config_read.discovery_methods {
            let discovered_nodes = NodeDiscovery::discover_nodes(discovery_method)?;

            let mut registry_write = registry.write().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;

            for nodeinfo in discovered_nodes {
                if registry_write.register_node(nodeinfo.clone())? {
                    // New node discovered
                    let mut log = eventlog.lock().map_err(|_| {
                        CoreError::InvalidState(ErrorContext::new(
                            "Failed to acquire event log lock",
                        ))
                    })?;
                    log.log_event(ClusterEvent::NodeDiscovered {
                        nodeid: nodeinfo.id.clone(),
                        address: nodeinfo.address,
                        timestamp: Instant::now(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Run the health monitoring loop
    pub fn health_monitoring_loop(
        healthmonitor: &Arc<Mutex<HealthMonitor>>,
        registry: &Arc<RwLock<NodeRegistry>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let nodes = {
            let registry_read = registry.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;
            registry_read.get_all_nodes()
        };

        let mut monitor = healthmonitor.lock().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire health monitor lock"))
        })?;

        for nodeinfo in nodes {
            let health_status = monitor.check_node_health(&nodeinfo)?;

            // Update node status
            let mut registry_write = registry.write().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;

            let previous_status = registry_write.get_node_status(&nodeinfo.id);
            registry_write.update_node_status(&nodeinfo.id, health_status.status)?;

            // Log status changes
            if let Some(prev_status) = previous_status {
                if prev_status != health_status.status {
                    let mut log = eventlog.lock().map_err(|_| {
                        CoreError::InvalidState(ErrorContext::new(
                            "Failed to acquire event log lock",
                        ))
                    })?;
                    log.log_event(ClusterEvent::NodeStatusChanged {
                        nodeid: nodeinfo.id.clone(),
                        old_status: prev_status,
                        new_status: health_status.status,
                        timestamp: Instant::now(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Run the resource management loop
    pub fn resource_management_loop(
        allocator: &Arc<RwLock<ResourceAllocator>>,
        registry: &Arc<RwLock<NodeRegistry>>,
    ) -> CoreResult<()> {
        let nodes = {
            let registry_read = registry.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;
            registry_read.get_healthy_nodes()
        };

        let mut allocator_write = allocator.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire allocator lock"))
        })?;

        allocator_write.update_available_resources(&nodes)?;
        allocator_write.optimize_resource_allocation()?;

        Ok(())
    }

    /// Run the cluster coordination loop
    pub fn cluster_coordination_loop(
        cluster_state: &Arc<RwLock<ClusterState>>,
        registry: &Arc<RwLock<NodeRegistry>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let healthy_nodes = {
            let registry_read = registry.read().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
            })?;
            registry_read.get_healthy_nodes()
        };

        let mut state_write = cluster_state.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire cluster state lock"))
        })?;

        // Update cluster topology
        state_write.update_topology(&healthy_nodes)?;

        // Check for leadership changes
        if state_write.needs_leader_election() {
            // Implement leader election logic: select node with smallest ID (deterministic)
            let new_leader = if healthy_nodes.is_empty() {
                None
            } else {
                healthy_nodes
                    .iter()
                    .filter(|node| node.status == NodeStatus::Healthy)
                    .min_by(|a, b| a.id.cmp(&b.id))
                    .map(|node| node.id.clone())
            };
            if let Some(leader) = new_leader {
                state_write.set_leader(leader.clone());

                let mut log = eventlog.lock().map_err(|_| {
                    CoreError::InvalidState(ErrorContext::new("Failed to acquire event log lock"))
                })?;
                log.log_event(ClusterEvent::LeaderElected {
                    nodeid: leader,
                    timestamp: Instant::now(),
                });
            }
        }

        Ok(())
    }

    /// Perform leader election
    pub fn elect_leader(nodes: &[NodeInfo]) -> CoreResult<Option<String>> {
        // Simple leader election based on node ID
        if nodes.is_empty() {
            return Ok(None);
        }

        // Select node with smallest ID (deterministic)
        let leader = nodes
            .iter()
            .filter(|node| node.status == NodeStatus::Healthy)
            .min_by(|a, b| a.id.cmp(&b.id));

        Ok(leader.map(|node| node.id.clone()))
    }

    /// Check if a node is reachable for discovery
    pub fn is_node_reachable(address: std::net::SocketAddr) -> CoreResult<bool> {
        // Simple reachability check
        // In a real implementation, this would do proper health checking
        Ok(true) // Placeholder
    }

    /// Create a node info from discovery
    pub fn create_discovered_node(
        address: std::net::SocketAddr,
        node_type: Option<NodeType>,
    ) -> NodeInfo {
        NodeInfo {
            id: format!("node_{address}"),
            address,
            node_type: node_type.unwrap_or(NodeType::Worker),
            capabilities: NodeCapabilities::default(),
            status: NodeStatus::Unknown,
            last_seen: Instant::now(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Validate cluster configuration
    pub fn validate_configuration(config: &ClusterConfiguration) -> CoreResult<()> {
        if config.discovery_methods.is_empty() {
            return Err(CoreError::ValidationError(ErrorContext::new(
                "At least one discovery method must be configured",
            )));
        }

        if config.health_check_interval.as_secs() == 0 {
            return Err(CoreError::ValidationError(ErrorContext::new(
                "Health check interval must be greater than 0",
            )));
        }

        if config.leadership_timeout.as_secs() == 0 {
            return Err(CoreError::ValidationError(ErrorContext::new(
                "Leadership timeout must be greater than 0",
            )));
        }

        if let Some(max_nodes) = config.max_nodes {
            if max_nodes == 0 {
                return Err(CoreError::ValidationError(ErrorContext::new(
                    "Max nodes must be greater than 0 if specified",
                )));
            }
        }

        Ok(())
    }

    /// Handle node removal from cluster
    pub fn handle_node_removal(
        node_id: &str,
        registry: &Arc<RwLock<NodeRegistry>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let mut registry_write = registry.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
        })?;

        if let Some(node) = registry_write.remove_node(node_id) {
            let mut log = eventlog.lock().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire event log lock"))
            })?;

            log.log_event(ClusterEvent::NodeStatusChanged {
                nodeid: node.id.clone(),
                old_status: node.status,
                new_status: NodeStatus::Offline,
                timestamp: Instant::now(),
            });
        }

        Ok(())
    }

    /// Handle graceful node shutdown
    pub fn handle_node_shutdown(
        node_id: &str,
        registry: &Arc<RwLock<NodeRegistry>>,
        eventlog: &Arc<Mutex<ClusterEventLog>>,
    ) -> CoreResult<()> {
        let mut registry_write = registry.write().map_err(|_| {
            CoreError::InvalidState(ErrorContext::new("Failed to acquire registry lock"))
        })?;

        if let Some(prev_status) = registry_write.get_node_status(node_id) {
            registry_write.update_node_status(node_id, NodeStatus::Draining)?;

            let mut log = eventlog.lock().map_err(|_| {
                CoreError::InvalidState(ErrorContext::new("Failed to acquire event log lock"))
            })?;

            log.log_event(ClusterEvent::NodeStatusChanged {
                nodeid: node_id.to_string(),
                old_status: prev_status,
                new_status: NodeStatus::Draining,
                timestamp: Instant::now(),
            });
        }

        Ok(())
    }
}
