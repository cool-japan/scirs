//! Evolving graph model: graphs that mutate through explicit, time-stamped operations.
//!
//! An `EvolvingGraph` records every structural change in a history log and can
//! reconstruct its exact state at any past time by replaying the log.

use std::collections::{HashMap, HashSet};

/// An atomic structural change that can be applied to a graph.
#[derive(Debug, Clone)]
pub enum GraphChange {
    /// Insert a new node.
    AddNode(usize),
    /// Delete a node (and any incident edges).
    RemoveNode(usize),
    /// Insert an undirected edge with the given weight.
    AddEdge(usize, usize, f64),
    /// Delete an undirected edge.
    RemoveEdge(usize, usize),
    /// Overwrite the weight of an existing (or new) edge.
    UpdateEdgeWeight(usize, usize, f64),
    /// Set a floating-point attribute on a node.
    UpdateNodeAttribute(usize, String, f64),
}

/// A graph that evolves over time through a sequence of explicit changes.
///
/// Internally the current state is maintained for O(1) queries; the full history
/// is kept so that `state_at` can reconstruct any past configuration.
pub struct EvolvingGraph {
    /// Chronological log of (timestamp, change) pairs.
    pub history: Vec<(f64, GraphChange)>,
    /// Timestamp of the most recent change applied.
    pub current_time: f64,
    nodes: HashSet<usize>,
    /// Canonical edge key: (min(u,v), max(u,v)) → weight.
    edges: HashMap<(usize, usize), f64>,
    /// Per-node attribute maps.
    node_attrs: HashMap<usize, HashMap<String, f64>>,
}

impl EvolvingGraph {
    /// Create an empty evolving graph.
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            current_time: 0.0,
            nodes: HashSet::new(),
            edges: HashMap::new(),
            node_attrs: HashMap::new(),
        }
    }

    /// Apply a change at the given timestamp, updating internal state and history.
    pub fn apply_change(&mut self, time: f64, change: GraphChange) {
        self.current_time = time;
        match &change {
            GraphChange::AddNode(n) => {
                self.nodes.insert(*n);
            }
            GraphChange::RemoveNode(n) => {
                self.nodes.remove(n);
                // Remove incident edges.
                self.edges.retain(|&(a, b), _| a != *n && b != *n);
                self.node_attrs.remove(n);
            }
            GraphChange::AddEdge(u, v, w) => {
                self.nodes.insert(*u);
                self.nodes.insert(*v);
                let key = edge_key(*u, *v);
                self.edges.insert(key, *w);
            }
            GraphChange::RemoveEdge(u, v) => {
                let key = edge_key(*u, *v);
                self.edges.remove(&key);
            }
            GraphChange::UpdateEdgeWeight(u, v, w) => {
                let key = edge_key(*u, *v);
                self.edges.insert(key, *w);
            }
            GraphChange::UpdateNodeAttribute(n, attr, val) => {
                self.node_attrs
                    .entry(*n)
                    .or_default()
                    .insert(attr.clone(), *val);
            }
        }
        self.history.push((time, change));
    }

    /// Number of nodes in the current state.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges in the current state.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Whether node `n` exists in the current state.
    pub fn has_node(&self, n: usize) -> bool {
        self.nodes.contains(&n)
    }

    /// Whether an edge between `u` and `v` exists in the current state.
    pub fn has_edge(&self, u: usize, v: usize) -> bool {
        self.edges.contains_key(&edge_key(u, v))
    }

    /// Weight of the edge between `u` and `v`, if it exists.
    pub fn edge_weight(&self, u: usize, v: usize) -> Option<f64> {
        self.edges.get(&edge_key(u, v)).copied()
    }

    /// Get the value of attribute `attr` on node `n`, if set.
    pub fn node_attribute(&self, n: usize, attr: &str) -> Option<f64> {
        self.node_attrs.get(&n).and_then(|m| m.get(attr)).copied()
    }

    /// Reconstruct the graph state at `target_time` by replaying history.
    ///
    /// Only changes with timestamp `<= target_time` are applied.
    pub fn state_at(&self, target_time: f64) -> Self {
        let mut g = Self::new();
        for (t, change) in &self.history {
            if *t <= target_time {
                g.apply_change(*t, change.clone());
            }
        }
        g
    }

    /// Iterator over all current node identifiers (unordered).
    pub fn nodes(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes.iter().copied()
    }

    /// Iterator over all current edges: `(u, v, weight)`.
    pub fn edges_iter(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        self.edges.iter().map(|(&(u, v), &w)| (u, v, w))
    }
}

impl Default for EvolvingGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Canonical undirected edge key.
#[inline]
fn edge_key(u: usize, v: usize) -> (usize, usize) {
    (u.min(v), u.max(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_evolving() -> EvolvingGraph {
        let mut g = EvolvingGraph::new();
        g.apply_change(1.0, GraphChange::AddNode(0));
        g.apply_change(2.0, GraphChange::AddNode(1));
        g.apply_change(3.0, GraphChange::AddEdge(0, 1, 1.5));
        g.apply_change(4.0, GraphChange::AddNode(2));
        g.apply_change(5.0, GraphChange::AddEdge(1, 2, 2.0));
        g.apply_change(6.0, GraphChange::RemoveNode(0));
        g
    }

    #[test]
    fn test_state_at_replays_history() {
        let g = build_evolving();

        // At t=3 nodes {0,1} and edge 0-1 should exist.
        let s3 = g.state_at(3.0);
        assert!(s3.has_node(0), "node 0 should be present at t=3");
        assert!(s3.has_node(1), "node 1 should be present at t=3");
        assert!(!s3.has_node(2), "node 2 should not be present at t=3");
        assert!(s3.has_edge(0, 1), "edge 0-1 should be present at t=3");
        assert!((s3.edge_weight(0, 1).unwrap_or(0.0) - 1.5).abs() < 1e-9);

        // At t=5 nodes {0,1,2}, edges {0-1, 1-2}.
        let s5 = g.state_at(5.0);
        assert_eq!(s5.n_nodes(), 3);
        assert_eq!(s5.n_edges(), 2);
        assert!(s5.has_edge(1, 2));

        // At t=6 node 0 and its incident edge are removed.
        let s6 = g.state_at(6.0);
        assert!(!s6.has_node(0), "node 0 should be removed at t=6");
        assert!(!s6.has_edge(0, 1), "edge 0-1 should be removed at t=6");
        assert!(s6.has_node(1));
        assert!(s6.has_node(2));
    }

    #[test]
    fn test_update_edge_weight() {
        let mut g = EvolvingGraph::new();
        g.apply_change(1.0, GraphChange::AddEdge(0, 1, 1.0));
        g.apply_change(2.0, GraphChange::UpdateEdgeWeight(0, 1, 5.0));
        assert!((g.edge_weight(0, 1).unwrap_or(0.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_node_attribute() {
        let mut g = EvolvingGraph::new();
        g.apply_change(1.0, GraphChange::AddNode(0));
        g.apply_change(
            2.0,
            GraphChange::UpdateNodeAttribute(0, "weight".to_string(), std::f64::consts::PI),
        );
        assert!((g.node_attribute(0, "weight").unwrap_or(0.0) - std::f64::consts::PI).abs() < 1e-9);
    }

    #[test]
    fn test_remove_node_cleans_edges() {
        let mut g = EvolvingGraph::new();
        g.apply_change(1.0, GraphChange::AddEdge(0, 1, 1.0));
        g.apply_change(2.0, GraphChange::AddEdge(1, 2, 2.0));
        g.apply_change(3.0, GraphChange::RemoveNode(1));
        // Both edges incident to node 1 must be gone.
        assert!(!g.has_edge(0, 1));
        assert!(!g.has_edge(1, 2));
        assert!(g.has_node(0));
        assert!(g.has_node(2));
    }

    #[test]
    fn test_canonical_edge_direction() {
        let mut g = EvolvingGraph::new();
        g.apply_change(1.0, GraphChange::AddEdge(3, 1, 7.0));
        // Query in reverse direction should work (undirected).
        assert!(g.has_edge(1, 3));
        assert!((g.edge_weight(1, 3).unwrap_or(0.0) - 7.0).abs() < 1e-9);
    }
}
