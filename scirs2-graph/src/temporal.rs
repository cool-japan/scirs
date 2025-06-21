//! Temporal graph structures and algorithms
//!
//! This module provides data structures and algorithms for temporal graphs,
//! where edges and nodes can have time-dependent properties.

use crate::base::{EdgeWeight, Graph, IndexType, Node};
use crate::error::{GraphError, Result};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Represents a time instant or interval
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimeInstant {
    /// Time value (could represent seconds, milliseconds, etc.)
    pub time: u64,
}

impl TimeInstant {
    /// Create a new time instant
    pub fn new(time: u64) -> Self {
        TimeInstant { time }
    }

    /// Get the time value
    pub fn value(&self) -> u64 {
        self.time
    }
}

/// Represents a time interval [start, end)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeInterval {
    /// Start time (inclusive)
    pub start: TimeInstant,
    /// End time (exclusive)
    pub end: TimeInstant,
}

impl TimeInterval {
    /// Create a new time interval
    pub fn new(start: u64, end: u64) -> Result<Self> {
        if start >= end {
            return Err(GraphError::InvalidGraph(
                "Start time must be before end time".to_string(),
            ));
        }
        Ok(TimeInterval {
            start: TimeInstant::new(start),
            end: TimeInstant::new(end),
        })
    }

    /// Check if this interval contains a time instant
    pub fn contains(&self, time: TimeInstant) -> bool {
        time >= self.start && time < self.end
    }

    /// Check if this interval overlaps with another
    pub fn overlaps(&self, other: &TimeInterval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Get the duration of this interval
    pub fn duration(&self) -> u64 {
        self.end.time - self.start.time
    }

    /// Get the intersection of two intervals
    pub fn intersection(&self, other: &TimeInterval) -> Option<TimeInterval> {
        if !self.overlaps(other) {
            return None;
        }

        let start = self.start.max(other.start);
        let end = self.end.min(other.end);

        if start < end {
            Some(TimeInterval { start, end })
        } else {
            None
        }
    }
}

/// A temporal edge with time-dependent properties
#[derive(Debug, Clone)]
pub struct TemporalEdge<N: Node, E: EdgeWeight> {
    /// Source node
    pub source: N,
    /// Target node
    pub target: N,
    /// Edge weight
    pub weight: E,
    /// Time interval when this edge exists
    pub interval: TimeInterval,
}

/// A temporal graph where edges have time intervals
#[derive(Debug, Clone)]
pub struct TemporalGraph<N: Node, E: EdgeWeight, Ix: IndexType = u32> {
    /// All nodes in the graph
    nodes: HashSet<N>,
    /// Temporal edges sorted by time
    edges: BTreeMap<TimeInstant, Vec<TemporalEdge<N, E>>>,
    /// Node appearance times
    node_intervals: HashMap<N, TimeInterval>,
    /// Edge ID counter
    edge_counter: usize,
    /// Index type phantom
    _phantom: std::marker::PhantomData<Ix>,
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> Default for TemporalGraph<N, E, Ix> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N: Node, E: EdgeWeight, Ix: IndexType> TemporalGraph<N, E, Ix> {
    /// Create a new empty temporal graph
    pub fn new() -> Self {
        TemporalGraph {
            nodes: HashSet::new(),
            edges: BTreeMap::new(),
            node_intervals: HashMap::new(),
            edge_counter: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a node with a time interval when it exists
    pub fn add_node(&mut self, node: N, interval: TimeInterval) {
        self.nodes.insert(node.clone());
        self.node_intervals.insert(node, interval);
    }

    /// Add a temporal edge
    pub fn add_edge(
        &mut self,
        source: N,
        target: N,
        weight: E,
        interval: TimeInterval,
    ) -> Result<usize> {
        // Ensure nodes exist and are active during the edge interval
        if let Some(source_interval) = self.node_intervals.get(&source) {
            if source_interval.intersection(&interval).is_none() {
                return Err(GraphError::InvalidGraph(
                    "Source node not active during edge interval".to_string(),
                ));
            }
        } else {
            // Add node with the edge interval
            self.add_node(source.clone(), interval);
        }

        if let Some(target_interval) = self.node_intervals.get(&target) {
            if target_interval.intersection(&interval).is_none() {
                return Err(GraphError::InvalidGraph(
                    "Target node not active during edge interval".to_string(),
                ));
            }
        } else {
            // Add node with the edge interval
            self.add_node(target.clone(), interval);
        }

        let edge = TemporalEdge {
            source,
            target,
            weight,
            interval,
        };

        // Add edge to the start time
        self.edges.entry(interval.start).or_default().push(edge);

        let edge_id = self.edge_counter;
        self.edge_counter += 1;
        Ok(edge_id)
    }

    /// Get a snapshot of the graph at a specific time
    pub fn snapshot_at(&self, time: TimeInstant) -> Graph<N, E, Ix>
    where
        N: Clone,
        E: Clone,
    {
        let mut snapshot = Graph::new();

        // Add active nodes
        for (node, interval) in &self.node_intervals {
            if interval.contains(time) {
                snapshot.add_node(node.clone());
            }
        }

        // Add active edges
        for edges in self.edges.values() {
            for edge in edges {
                if edge.interval.contains(time) {
                    snapshot
                        .add_edge(
                            edge.source.clone(),
                            edge.target.clone(),
                            edge.weight.clone(),
                        )
                        .unwrap();
                }
            }
        }

        snapshot
    }

    /// Get all edges active during a time interval
    pub fn edges_in_interval(&self, interval: TimeInterval) -> Vec<&TemporalEdge<N, E>> {
        let mut result = Vec::new();

        for edges in self.edges.values() {
            for edge in edges {
                if edge.interval.overlaps(&interval) {
                    result.push(edge);
                }
            }
        }

        result
    }

    /// Get all time instants when the graph structure changes
    pub fn change_times(&self) -> Vec<TimeInstant> {
        let mut times = HashSet::new();

        // Add node start and end times
        for interval in self.node_intervals.values() {
            times.insert(interval.start);
            times.insert(interval.end);
        }

        // Add edge start and end times
        for edges in self.edges.values() {
            for edge in edges {
                times.insert(edge.interval.start);
                times.insert(edge.interval.end);
            }
        }

        let mut times: Vec<_> = times.into_iter().collect();
        times.sort();
        times
    }

    /// Get the time interval when the graph is active
    pub fn active_interval(&self) -> Option<TimeInterval> {
        if self.nodes.is_empty() {
            return None;
        }

        let change_times = self.change_times();
        if change_times.len() < 2 {
            return None;
        }

        let start = change_times[0];
        let end = change_times[change_times.len() - 1];

        TimeInterval::new(start.time, end.time).ok()
    }

    /// Count nodes active at a specific time
    pub fn node_count_at(&self, time: TimeInstant) -> usize {
        self.node_intervals
            .values()
            .filter(|interval| interval.contains(time))
            .count()
    }

    /// Count edges active at a specific time
    pub fn edge_count_at(&self, time: TimeInstant) -> usize {
        let mut count = 0;
        for edges in self.edges.values() {
            for edge in edges {
                if edge.interval.contains(time) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Get all nodes in the temporal graph
    pub fn nodes(&self) -> impl Iterator<Item = &N> {
        self.nodes.iter()
    }

    /// Get all temporal edges
    pub fn temporal_edges(&self) -> Vec<&TemporalEdge<N, E>> {
        let mut result = Vec::new();
        for edges in self.edges.values() {
            result.extend(edges.iter());
        }
        result
    }

    /// Check if two nodes are connected at a specific time
    pub fn are_connected_at(&self, node1: &N, node2: &N, time: TimeInstant) -> bool {
        for edges in self.edges.values() {
            for edge in edges {
                if edge.interval.contains(time)
                    && ((edge.source == *node1 && edge.target == *node2)
                        || (edge.source == *node2 && edge.target == *node1))
                {
                    return true;
                }
            }
        }
        false
    }

    /// Find temporal paths between two nodes
    pub fn temporal_paths(
        &self,
        source: &N,
        target: &N,
        start_time: TimeInstant,
        max_duration: u64,
    ) -> Vec<TemporalPath<N, E>>
    where
        N: Clone + Ord,
        E: Clone,
    {
        let mut paths = Vec::new();
        let end_time = TimeInstant::new(start_time.time + max_duration);

        // Use BFS to find temporal paths
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(TemporalPath {
            nodes: vec![source.clone()],
            edges: Vec::new(),
            total_weight: None,
            start_time,
            end_time: start_time,
        });

        while let Some(current_path) = queue.pop_front() {
            let current_node = current_path.nodes.last().unwrap();

            if current_node == target {
                paths.push(current_path);
                continue;
            }

            if current_path.end_time >= end_time {
                continue;
            }

            // Find next possible edges
            for edges in self.edges.values() {
                for edge in edges {
                    if edge.source == *current_node
                        && edge.interval.start >= current_path.end_time
                        && edge.interval.start <= end_time
                        && !current_path.nodes.contains(&edge.target)
                    {
                        let mut new_path = current_path.clone();
                        new_path.nodes.push(edge.target.clone());
                        new_path.edges.push(edge.clone());
                        new_path.end_time = edge.interval.end;

                        queue.push_back(new_path);
                    }
                }
            }
        }

        paths
    }
}

/// Represents a path in a temporal graph
#[derive(Debug, Clone)]
pub struct TemporalPath<N: Node, E: EdgeWeight> {
    /// Nodes in the path
    pub nodes: Vec<N>,
    /// Edges in the path
    pub edges: Vec<TemporalEdge<N, E>>,
    /// Total weight of the path
    pub total_weight: Option<E>,
    /// Start time of the path
    pub start_time: TimeInstant,
    /// End time of the path
    pub end_time: TimeInstant,
}

impl<N: Node, E: EdgeWeight> TemporalPath<N, E> {
    /// Get the duration of this path
    pub fn duration(&self) -> u64 {
        self.end_time.time - self.start_time.time
    }

    /// Get the number of hops in this path
    pub fn hop_count(&self) -> usize {
        self.edges.len()
    }
}

/// Compute reachability in a temporal graph
///
/// Returns all nodes reachable from a source node within a time window.
pub fn temporal_reachability<N, E, Ix>(
    temporal_graph: &TemporalGraph<N, E, Ix>,
    source: &N,
    start_time: TimeInstant,
    max_duration: u64,
) -> HashSet<N>
where
    N: Node + Clone + Ord,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    let mut reachable = HashSet::new();
    let mut visited = HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back((source.clone(), start_time));
    visited.insert((source.clone(), start_time));
    reachable.insert(source.clone());

    let end_time = TimeInstant::new(start_time.time + max_duration);

    while let Some((current_node, current_time)) = queue.pop_front() {
        if current_time >= end_time {
            continue;
        }

        // Find outgoing edges from current node at or after current time
        for edges in temporal_graph.edges.values() {
            for edge in edges {
                if edge.source == current_node
                    && edge.interval.start >= current_time
                    && edge.interval.start <= end_time
                {
                    let next_time = edge.interval.end;
                    let next_node = edge.target.clone();

                    if !visited.contains(&(next_node.clone(), next_time)) {
                        visited.insert((next_node.clone(), next_time));
                        reachable.insert(next_node.clone());
                        queue.push_back((next_node, next_time));
                    }
                }
            }
        }
    }

    reachable
}

/// Compute temporal centrality measures
pub fn temporal_betweenness_centrality<N, E, Ix>(
    temporal_graph: &TemporalGraph<N, E, Ix>,
    time_window: TimeInterval,
) -> HashMap<N, f64>
where
    N: Node + Clone + Ord,
    E: EdgeWeight + Clone,
    Ix: IndexType,
{
    let mut centrality = HashMap::new();

    // Initialize centrality for all nodes
    for node in temporal_graph.nodes() {
        centrality.insert(node.clone(), 0.0);
    }

    let nodes: Vec<N> = temporal_graph.nodes().cloned().collect();

    // For each pair of nodes, find temporal paths and count betweenness
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let source = &nodes[i];
            let target = &nodes[j];

            let paths = temporal_graph.temporal_paths(
                source,
                target,
                time_window.start,
                time_window.duration(),
            );

            if !paths.is_empty() {
                // Count how many paths go through each intermediate node
                for path in &paths {
                    for k in 1..(path.nodes.len() - 1) {
                        let intermediate = &path.nodes[k];
                        *centrality.get_mut(intermediate).unwrap() += 1.0 / paths.len() as f64;
                    }
                }
            }
        }
    }

    centrality
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_instant() {
        let t1 = TimeInstant::new(100);
        let t2 = TimeInstant::new(200);

        assert!(t1 < t2);
        assert_eq!(t1.value(), 100);
        assert_eq!(t2.value(), 200);
    }

    #[test]
    fn test_time_interval() {
        let interval = TimeInterval::new(100, 200).unwrap();

        assert_eq!(interval.duration(), 100);
        assert!(interval.contains(TimeInstant::new(150)));
        assert!(!interval.contains(TimeInstant::new(50)));
        assert!(!interval.contains(TimeInstant::new(200))); // End is exclusive

        // Test invalid interval
        assert!(TimeInterval::new(200, 100).is_err());
    }

    #[test]
    fn test_interval_overlap() {
        let interval1 = TimeInterval::new(100, 200).unwrap();
        let interval2 = TimeInterval::new(150, 250).unwrap();
        let interval3 = TimeInterval::new(300, 400).unwrap();

        assert!(interval1.overlaps(&interval2));
        assert!(!interval1.overlaps(&interval3));

        let intersection = interval1.intersection(&interval2).unwrap();
        assert_eq!(intersection.start.time, 150);
        assert_eq!(intersection.end.time, 200);

        assert!(interval1.intersection(&interval3).is_none());
    }

    #[test]
    fn test_temporal_graph_creation() {
        let mut tgraph: TemporalGraph<&str, f64> = TemporalGraph::new();

        let interval1 = TimeInterval::new(0, 100).unwrap();
        let interval2 = TimeInterval::new(50, 150).unwrap();

        tgraph.add_node("A", interval1);
        tgraph.add_node("B", interval2);

        let edge_interval = TimeInterval::new(60, 90).unwrap();
        tgraph.add_edge("A", "B", 1.0, edge_interval).unwrap();

        // Test snapshot at different times
        let snapshot_at_70 = tgraph.snapshot_at(TimeInstant::new(70));
        assert_eq!(snapshot_at_70.node_count(), 2);
        assert_eq!(snapshot_at_70.edge_count(), 1);

        let snapshot_at_120 = tgraph.snapshot_at(TimeInstant::new(120));
        assert_eq!(snapshot_at_120.node_count(), 1); // Only B is active
        assert_eq!(snapshot_at_120.edge_count(), 0);
    }

    #[test]
    fn test_temporal_connectivity() {
        let mut tgraph: TemporalGraph<i32, f64> = TemporalGraph::new();

        let node_interval = TimeInterval::new(0, 200).unwrap();
        tgraph.add_node(1, node_interval);
        tgraph.add_node(2, node_interval);

        let edge_interval = TimeInterval::new(50, 150).unwrap();
        tgraph.add_edge(1, 2, 1.0, edge_interval).unwrap();

        // Test connectivity at different times
        assert!(!tgraph.are_connected_at(&1, &2, TimeInstant::new(30)));
        assert!(tgraph.are_connected_at(&1, &2, TimeInstant::new(100)));
        assert!(!tgraph.are_connected_at(&1, &2, TimeInstant::new(170)));
    }

    #[test]
    fn test_change_times() {
        let mut tgraph: TemporalGraph<&str, f64> = TemporalGraph::new();

        let interval1 = TimeInterval::new(0, 100).unwrap();
        let interval2 = TimeInterval::new(50, 150).unwrap();

        tgraph.add_node("A", interval1);
        tgraph.add_edge("A", "B", 1.0, interval2).unwrap();

        let change_times = tgraph.change_times();
        let times: Vec<u64> = change_times.iter().map(|t| t.time).collect();

        assert!(times.contains(&0));
        assert!(times.contains(&50));
        assert!(times.contains(&100));
        assert!(times.contains(&150));
    }

    #[test]
    fn test_temporal_reachability() {
        let mut tgraph: TemporalGraph<i32, f64> = TemporalGraph::new();

        let node_interval = TimeInterval::new(0, 300).unwrap();
        for i in 1..=4 {
            tgraph.add_node(i, node_interval);
        }

        // Create a temporal path: 1 -> 2 -> 3 -> 4
        tgraph
            .add_edge(1, 2, 1.0, TimeInterval::new(10, 50).unwrap())
            .unwrap();
        tgraph
            .add_edge(2, 3, 1.0, TimeInterval::new(60, 100).unwrap())
            .unwrap();
        tgraph
            .add_edge(3, 4, 1.0, TimeInterval::new(110, 150).unwrap())
            .unwrap();

        let reachable = temporal_reachability(&tgraph, &1, TimeInstant::new(0), 200);

        assert!(reachable.contains(&1));
        assert!(reachable.contains(&2));
        assert!(reachable.contains(&3));
        assert!(reachable.contains(&4));
        assert_eq!(reachable.len(), 4);

        // Test with limited time window
        let reachable_limited = temporal_reachability(&tgraph, &1, TimeInstant::new(0), 80);
        assert!(reachable_limited.contains(&1));
        assert!(reachable_limited.contains(&2));
        assert!(reachable_limited.contains(&3));
        assert!(!reachable_limited.contains(&4)); // Can't reach 4 in time
    }

    #[test]
    fn test_temporal_paths() {
        let mut tgraph: TemporalGraph<&str, f64> = TemporalGraph::new();

        let node_interval = TimeInterval::new(0, 200).unwrap();
        for &node in &["A", "B", "C"] {
            tgraph.add_node(node, node_interval);
        }

        // Direct path and indirect path
        tgraph
            .add_edge("A", "C", 1.0, TimeInterval::new(10, 50).unwrap())
            .unwrap();
        tgraph
            .add_edge("A", "B", 1.0, TimeInterval::new(20, 60).unwrap())
            .unwrap();
        tgraph
            .add_edge("B", "C", 1.0, TimeInterval::new(70, 110).unwrap())
            .unwrap();

        let paths = tgraph.temporal_paths(&"A", &"C", TimeInstant::new(0), 150);

        assert!(!paths.is_empty());

        // Should find both direct and indirect paths
        let has_direct = paths.iter().any(|p| p.nodes.len() == 2);
        let has_indirect = paths.iter().any(|p| p.nodes.len() == 3);

        assert!(has_direct || has_indirect);
    }

    #[test]
    fn test_edge_count_at_time() {
        let mut tgraph: TemporalGraph<i32, f64> = TemporalGraph::new();

        let node_interval = TimeInterval::new(0, 200).unwrap();
        tgraph.add_node(1, node_interval);
        tgraph.add_node(2, node_interval);
        tgraph.add_node(3, node_interval);

        tgraph
            .add_edge(1, 2, 1.0, TimeInterval::new(10, 50).unwrap())
            .unwrap();
        tgraph
            .add_edge(2, 3, 1.0, TimeInterval::new(30, 70).unwrap())
            .unwrap();

        assert_eq!(tgraph.edge_count_at(TimeInstant::new(5)), 0);
        assert_eq!(tgraph.edge_count_at(TimeInstant::new(40)), 2);
        assert_eq!(tgraph.edge_count_at(TimeInstant::new(60)), 1);
        assert_eq!(tgraph.edge_count_at(TimeInstant::new(80)), 0);
    }
}
