//! Node registry for tracking cluster members
//!
//! This module provides the NodeRegistry for managing cluster node
//! information, status tracking, and node health monitoring.

use crate::error::CoreResult;
use std::collections::HashMap;

use super::types::{NodeInfo, NodeStatus};

/// Node registry for tracking cluster members
#[derive(Debug)]
pub struct NodeRegistry {
    nodes: HashMap<String, NodeInfo>,
    node_status: HashMap<String, NodeStatus>,
}

impl Default for NodeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeRegistry {
    /// Create a new node registry
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            node_status: HashMap::new(),
        }
    }

    /// Register a new node in the cluster
    /// Returns true if the node is new, false if it was already registered
    pub fn register_node(&mut self, nodeinfo: NodeInfo) -> CoreResult<bool> {
        let is_new = !self.nodes.contains_key(&nodeinfo.id);
        self.nodes.insert(nodeinfo.id.clone(), nodeinfo.clone());
        self.node_status
            .insert(nodeinfo.id.clone(), nodeinfo.status);
        Ok(is_new)
    }

    /// Get all registered nodes
    pub fn get_all_nodes(&self) -> Vec<NodeInfo> {
        self.nodes.values().cloned().collect()
    }

    /// Get only healthy nodes
    pub fn get_healthy_nodes(&self) -> Vec<NodeInfo> {
        self.nodes
            .values()
            .filter(|node| self.node_status.get(&node.id) == Some(&NodeStatus::Healthy))
            .cloned()
            .collect()
    }

    /// Get nodes with a specific status
    pub fn get_nodes_by_status(&self, status: NodeStatus) -> Vec<NodeInfo> {
        self.nodes
            .values()
            .filter(|node| self.node_status.get(&node.id) == Some(&status))
            .cloned()
            .collect()
    }

    /// Get a specific node by ID
    pub fn get_node(&self, nodeid: &str) -> Option<&NodeInfo> {
        self.nodes.get(nodeid)
    }

    /// Get the status of a specific node
    pub fn get_node_status(&self, nodeid: &str) -> Option<NodeStatus> {
        self.node_status.get(nodeid).copied()
    }

    /// Update the status of a specific node
    pub fn update_node_status(&mut self, nodeid: &str, status: NodeStatus) -> CoreResult<()> {
        if let Some(node) = self.nodes.get_mut(nodeid) {
            node.status = status;
            self.node_status.insert(nodeid.to_string(), status);
        }
        Ok(())
    }

    /// Remove a node from the registry
    pub fn remove_node(&mut self, nodeid: &str) -> Option<NodeInfo> {
        self.node_status.remove(nodeid);
        self.nodes.remove(nodeid)
    }

    /// Get the total number of registered nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get the number of healthy nodes
    pub fn healthy_node_count(&self) -> usize {
        self.get_healthy_nodes().len()
    }

    /// Check if a node is registered
    pub fn contains_node(&self, nodeid: &str) -> bool {
        self.nodes.contains_key(nodeid)
    }

    /// Clear all nodes from the registry
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.node_status.clear();
    }

    /// Get nodes that haven't been seen recently
    pub fn get_stale_nodes(&self, max_age: std::time::Duration) -> Vec<NodeInfo> {
        let now = std::time::Instant::now();
        self.nodes
            .values()
            .filter(|node| now.duration_since(node.last_seen) > max_age)
            .cloned()
            .collect()
    }

    /// Update a node's last seen timestamp
    pub fn update_node_last_seen(&mut self, nodeid: &str) {
        if let Some(node) = self.nodes.get_mut(nodeid) {
            node.last_seen = std::time::Instant::now();
        }
    }
}
