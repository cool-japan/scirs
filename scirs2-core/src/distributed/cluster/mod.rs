//! Cluster management for distributed computing
//!
//! This module provides comprehensive cluster management capabilities
//! including node discovery, health monitoring, resource allocation,
//! and fault-tolerant cluster coordination.

// Module declarations
pub mod allocator;
pub mod coordination;
pub mod discovery;
pub mod events;
pub mod health;
pub mod manager;
pub mod registry;
pub mod state;
pub mod types;

// Re-export main types and functionality
pub use manager::{initialize_cluster_manager, ClusterManager};

pub use types::{
    AllocationId,
    AllocationStrategy,

    BackoffStrategy,
    // Configuration types
    ClusterConfiguration,
    // Events
    ClusterEvent,

    ClusterHealth,
    ClusterHealthStatus,

    // Statistics
    ClusterStatistics,
    // Cluster topology
    ClusterTopology,
    // Resource management
    ComputeCapacity,
    DataAccessType,
    DataDependency,
    // Task management
    DistributedTask,
    ExecutionPlan,
    ExecutionStatus,

    HealthCheck,
    HealthCheckResult,
    NetworkTopology,

    NodeCapabilities,
    NodeDiscoveryMethod,
    // Health monitoring
    NodeHealthStatus,
    // Node types
    NodeInfo,
    NodeMetadata,
    NodeStatus,
    NodeType,
    ResourceAllocation,
    ResourceRequirements,
    ResourceUtilization,
    RetryPolicy,
    SpecializedRequirement,

    SpecializedUnit,

    TaskId,
    TaskParameters,
    TaskPriority,
    TaskType,
    Zone,
};

pub use allocator::ResourceAllocator;
pub use coordination::ClusterCoordination;
pub use discovery::NodeDiscovery;
pub use events::ClusterEventLog;
pub use health::HealthMonitor;
pub use registry::NodeRegistry;
pub use state::ClusterState;

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::time::Instant;

    #[test]
    fn test_cluster_manager_creation() {
        let config = ClusterConfiguration::default();
        let manager = ClusterManager::new(config).unwrap();
        // Basic functionality test
    }

    #[test]
    fn test_node_registry() {
        let mut registry = NodeRegistry::new();

        let node = NodeInfo {
            id: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            node_type: NodeType::Worker,
            capabilities: NodeCapabilities::default(),
            status: NodeStatus::Healthy,
            last_seen: Instant::now(),
            metadata: NodeMetadata::default(),
        };

        let is_new = registry.register_node(node.clone()).unwrap();
        assert!(is_new);

        let healthy_nodes = registry.get_healthy_nodes();
        assert_eq!(healthy_nodes.len(), 1);
        assert_eq!(healthy_nodes[0usize].id, "test_node");
    }

    #[test]
    fn test_resource_allocator() {
        let mut allocator = ResourceAllocator::new();

        // Set some available resources
        let nodes = vec![NodeInfo {
            id: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            node_type: NodeType::Worker,
            capabilities: NodeCapabilities {
                cpu_cores: 8,
                memory_gb: 16,
                gpu_count: 1,
                disk_space_gb: 100,
                networkbandwidth_gbps: 1.0,
                specialized_units: Vec::new(),
            },
            status: NodeStatus::Healthy,
            last_seen: Instant::now(),
            metadata: NodeMetadata::default(),
        }];

        allocator.update_available_resources(&nodes).unwrap();

        let requirements = ResourceRequirements {
            cpu_cores: 4,
            memory_gb: 8,
            gpu_count: 0,
            disk_space_gb: 50,
            specialized_requirements: Vec::new(),
        };

        let allocation = allocator.allocate_resources(&requirements).unwrap();
        assert_eq!(allocation.allocated_resources.cpu_cores, 4);
        assert_eq!(allocation.allocated_resources.memory_gb, 8);
    }

    #[test]
    fn test_health_monitor() {
        let mut monitor = HealthMonitor::new().unwrap();

        let node = NodeInfo {
            id: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            node_type: NodeType::Worker,
            capabilities: NodeCapabilities::default(),
            status: NodeStatus::Unknown,
            last_seen: Instant::now(),
            metadata: NodeMetadata::default(),
        };

        let health_status = monitor.check_node_health(&node).unwrap();
        assert!(health_status.health_score >= 0.0 && health_status.health_score <= 100.0f64);
    }

    #[test]
    fn test_cluster_topology() {
        let mut topology = ClusterTopology::new();

        let nodes = vec![
            NodeInfo {
                id: "node1".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 8080),
                node_type: NodeType::Worker,
                capabilities: NodeCapabilities::default(),
                status: NodeStatus::Healthy,
                last_seen: Instant::now(),
                metadata: NodeMetadata::default(),
            },
            NodeInfo {
                id: "node2".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)), 8080),
                node_type: NodeType::Worker,
                capabilities: NodeCapabilities::default(),
                status: NodeStatus::Healthy,
                last_seen: Instant::now(),
                metadata: NodeMetadata::default(),
            },
        ];

        topology.update_model(&nodes);
        assert_eq!(topology.zones.len(), 2); // Two different zones based on IP
    }

    #[test]
    fn test_cluster_state() {
        let mut state = ClusterState::new();

        assert!(state.needs_leader_election());
        assert!(state.get_leader().is_none());

        state.set_leader("node1".to_string());
        assert_eq!(state.get_leader(), Some(&"node1".to_string()));
        assert!(!state.needs_leader_election());
    }

    #[test]
    fn test_cluster_event_log() {
        let mut log = ClusterEventLog::new();

        let event = ClusterEvent::NodeDiscovered {
            nodeid: "test_node".to_string(),
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
            timestamp: Instant::now(),
        };

        log.log_event(event);
        assert_eq!(log.event_count(), 1);

        let recent_events = log.get_recent_events(10);
        assert_eq!(recent_events.len(), 1);
    }

    #[test]
    fn test_coordination_leader_election() {
        let nodes = vec![
            NodeInfo {
                id: "node_z".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
                node_type: NodeType::Worker,
                capabilities: NodeCapabilities::default(),
                status: NodeStatus::Healthy,
                last_seen: Instant::now(),
                metadata: NodeMetadata::default(),
            },
            NodeInfo {
                id: "node_a".to_string(),
                address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2)), 8080),
                node_type: NodeType::Worker,
                capabilities: NodeCapabilities::default(),
                status: NodeStatus::Healthy,
                last_seen: Instant::now(),
                metadata: NodeMetadata::default(),
            },
        ];

        let leader = ClusterCoordination::elect_leader(&nodes).unwrap();
        assert_eq!(leader, Some("node_a".to_string())); // Should elect lexicographically first
    }
}
