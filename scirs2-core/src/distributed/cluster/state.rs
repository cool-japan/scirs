//! Cluster state management
//!
//! This module handles cluster state tracking, including leadership,
//! topology updates, and cluster health status.

use crate::error::CoreResult;
use std::time::{Duration, Instant};

use super::types::{ClusterTopology, NodeInfo};

/// Cluster state management
#[derive(Debug)]
pub struct ClusterState {
    leader_node: Option<String>,
    topology: ClusterTopology,
    last_updated: Instant,
}

impl Default for ClusterState {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusterState {
    /// Create a new cluster state
    pub fn new() -> Self {
        Self {
            leader_node: None,
            topology: ClusterTopology::new(),
            last_updated: Instant::now(),
        }
    }

    /// Update the cluster topology with new node information
    pub fn update_topology(&mut self, nodes: &[NodeInfo]) -> CoreResult<()> {
        self.topology.update(nodes);
        self.last_updated = Instant::now();
        Ok(())
    }

    /// Check if leader election is needed
    pub fn needs_leader_election(&self) -> bool {
        self.leader_node.is_none() || self.last_updated.elapsed() > Duration::from_secs(300)
        // Re-elect every 5 minutes
    }

    /// Set the cluster leader
    pub fn set_leader(&mut self, nodeid: String) {
        self.leader_node = Some(nodeid);
        self.last_updated = Instant::now();
    }

    /// Get the current cluster leader
    pub fn get_leader(&self) -> Option<&String> {
        self.leader_node.as_ref()
    }

    /// Get the cluster topology
    pub fn get_topology(&self) -> &ClusterTopology {
        &self.topology
    }

    /// Get the last update timestamp
    pub fn last_updated(&self) -> Instant {
        self.last_updated
    }

    /// Check if the cluster state is stale
    pub fn is_stale(&self, max_age: Duration) -> bool {
        self.last_updated.elapsed() > max_age
    }

    /// Reset cluster state
    pub fn reset(&mut self) {
        self.leader_node = None;
        self.topology = ClusterTopology::new();
        self.last_updated = Instant::now();
    }
}
