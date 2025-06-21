//! JSON format I/O for graphs
//!
//! This module provides functionality for reading and writing graphs in JSON format.
//! The JSON format provides a flexible, human-readable representation of graph structures
//! with support for arbitrary node and edge attributes.
//!
//! # Format Specification
//!
//! The JSON graph format uses the following structure:
//! ```json
//! {
//!   "directed": false,
//!   "nodes": [
//!     {"id": "node1", "data": {...}},
//!     {"id": "node2", "data": {...}}
//!   ],
//!   "edges": [
//!     {"source": "node1", "target": "node2", "weight": 1.5},
//!     {"source": "node2", "target": "node3", "weight": 2.0}
//!   ]
//! }
//! ```
//!
//! # Examples
//!
//! ## Unweighted graph:
//! ```json
//! {
//!   "directed": false,
//!   "nodes": [
//!     {"id": "1"},
//!     {"id": "2"},
//!     {"id": "3"}
//!   ],
//!   "edges": [
//!     {"source": "1", "target": "2"},
//!     {"source": "2", "target": "3"}
//!   ]
//! }
//! ```
//!
//! ## Weighted graph:
//! ```json
//! {
//!   "directed": true,
//!   "nodes": [
//!     {"id": "1", "label": "Start"},
//!     {"id": "2", "label": "Middle"},
//!     {"id": "3", "label": "End"}
//!   ],
//!   "edges": [
//!     {"source": "1", "target": "2", "weight": 1.5},
//!     {"source": "2", "target": "3", "weight": 2.0}
//!   ]
//! }
//! ```
//!
//! # Usage
//!
//! ```rust
//! use std::fs::File;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//! use scirs2_graph::base::Graph;
//! use scirs2_graph::io::json::{read_json_format, write_json_format};
//!
//! // Create a temporary file with JSON data
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, r#"{{"directed": false, "nodes": [{{"id": "1"}}, {{"id": "2"}}], "edges": [{{"source": "1", "target": "2"}}]}}"#).unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read the graph
//! let graph: Graph<i32, f64> = read_json_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 2);
//! assert_eq!(graph.edge_count(), 1);
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// JSON representation of a graph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonNode {
    /// Node identifier
    pub id: String,
    /// Optional node label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Additional node data (currently unused but reserved for future extensions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// JSON representation of a graph edge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonEdge {
    /// Source node identifier
    pub source: String,
    /// Target node identifier
    pub target: String,
    /// Optional edge weight
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<f64>,
    /// Optional edge label
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Additional edge data (currently unused but reserved for future extensions)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// JSON representation of a complete graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonGraph {
    /// Whether the graph is directed
    #[serde(default = "default_directed")]
    pub directed: bool,
    /// List of nodes in the graph
    pub nodes: Vec<JsonNode>,
    /// List of edges in the graph
    pub edges: Vec<JsonEdge>,
    /// Optional graph metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

/// Default value for directed field
fn default_directed() -> bool {
    false
}

/// Read an undirected graph from JSON format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from the JSON
///
/// # Returns
///
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
///
/// # Format
///
/// The JSON format supports:
/// - Node declarations with optional labels and data
/// - Edge declarations with optional weights, labels, and data
/// - Graph metadata and directedness specification
/// - Flexible attribute storage for future extensions
pub fn read_json_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let json_graph: JsonGraph = serde_json::from_reader(reader)
        .map_err(|e| GraphError::Other(format!("Failed to parse JSON: {}", e)))?;

    // Verify this is an undirected graph
    if json_graph.directed {
        return Err(GraphError::Other(
            "JSON file contains a directed graph, but undirected graph was requested".to_string(),
        ));
    }

    let mut graph = Graph::new();

    // Parse nodes first to ensure they exist
    let mut node_map = HashMap::new();
    for json_node in &json_graph.nodes {
        let node = N::from_str(&json_node.id)
            .map_err(|_| GraphError::Other(format!("Failed to parse node ID: {}", json_node.id)))?;
        node_map.insert(json_node.id.clone(), node);
    }

    // Parse edges
    for json_edge in &json_graph.edges {
        let source_node = node_map.get(&json_edge.source).ok_or_else(|| {
            GraphError::Other(format!(
                "Edge references unknown source node: {}",
                json_edge.source
            ))
        })?;

        let target_node = node_map.get(&json_edge.target).ok_or_else(|| {
            GraphError::Other(format!(
                "Edge references unknown target node: {}",
                json_edge.target
            ))
        })?;

        // Parse weight if needed
        let weight = if weighted {
            if let Some(w) = json_edge.weight {
                E::from_str(&w.to_string())
                    .map_err(|_| GraphError::Other(format!("Failed to parse edge weight: {}", w)))?
            } else {
                E::default()
            }
        } else {
            E::default()
        };

        // Add edge
        graph.add_edge(source_node.clone(), target_node.clone(), weight)?;
    }

    Ok(graph)
}

/// Read a directed graph from JSON format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from the JSON
///
/// # Returns
///
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
pub fn read_json_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let json_graph: JsonGraph = serde_json::from_reader(reader)
        .map_err(|e| GraphError::Other(format!("Failed to parse JSON: {}", e)))?;

    let mut graph = DiGraph::new();

    // Parse nodes first to ensure they exist
    let mut node_map = HashMap::new();
    for json_node in &json_graph.nodes {
        let node = N::from_str(&json_node.id)
            .map_err(|_| GraphError::Other(format!("Failed to parse node ID: {}", json_node.id)))?;
        node_map.insert(json_node.id.clone(), node);
    }

    // Parse edges
    for json_edge in &json_graph.edges {
        let source_node = node_map.get(&json_edge.source).ok_or_else(|| {
            GraphError::Other(format!(
                "Edge references unknown source node: {}",
                json_edge.source
            ))
        })?;

        let target_node = node_map.get(&json_edge.target).ok_or_else(|| {
            GraphError::Other(format!(
                "Edge references unknown target node: {}",
                json_edge.target
            ))
        })?;

        // Parse weight if needed
        let weight = if weighted {
            if let Some(w) = json_edge.weight {
                E::from_str(&w.to_string())
                    .map_err(|_| GraphError::Other(format!("Failed to parse edge weight: {}", w)))?
            } else {
                E::default()
            }
        } else {
            E::default()
        };

        // Add directed edge
        graph.add_edge(source_node.clone(), target_node.clone(), weight)?;
    }

    Ok(graph)
}

/// Write an undirected graph to JSON format
///
/// # Arguments
///
/// * `graph` - The graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in the output
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_json_format<N, E, Ix, P>(
    graph: &Graph<N, E, Ix>,
    path: P,
    weighted: bool,
) -> Result<()>
where
    N: Node + std::fmt::Debug + std::fmt::Display + Clone,
    E: EdgeWeight
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default
        + std::fmt::Display
        + Clone,
    Ix: petgraph::graph::IndexType,
    P: AsRef<Path>,
{
    let mut file = File::create(path)?;

    // Collect nodes
    let nodes: Vec<JsonNode> = graph
        .nodes()
        .iter()
        .map(|node| JsonNode {
            id: node.to_string(),
            label: None,
            data: None,
        })
        .collect();

    // Collect edges
    let edges: Vec<JsonEdge> = graph
        .edges()
        .iter()
        .map(|edge| JsonEdge {
            source: edge.source.to_string(),
            target: edge.target.to_string(),
            weight: if weighted {
                // Convert weight to f64 through string parsing
                edge.weight.to_string().parse::<f64>().ok()
            } else {
                None
            },
            label: None,
            data: None,
        })
        .collect();

    // Create JSON graph
    let json_graph = JsonGraph {
        directed: false,
        nodes,
        edges,
        metadata: None,
    };

    // Write JSON to file
    let json_string = serde_json::to_string_pretty(&json_graph)
        .map_err(|e| GraphError::Other(format!("Failed to serialize JSON: {}", e)))?;

    write!(file, "{}", json_string)?;

    Ok(())
}

/// Write a directed graph to JSON format
///
/// # Arguments
///
/// * `graph` - The directed graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in the output
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_json_format_digraph<N, E, Ix, P>(
    graph: &DiGraph<N, E, Ix>,
    path: P,
    weighted: bool,
) -> Result<()>
where
    N: Node + std::fmt::Debug + std::fmt::Display + Clone,
    E: EdgeWeight
        + std::marker::Copy
        + std::fmt::Debug
        + std::default::Default
        + std::fmt::Display
        + Clone,
    Ix: petgraph::graph::IndexType,
    P: AsRef<Path>,
{
    let mut file = File::create(path)?;

    // Collect nodes
    let nodes: Vec<JsonNode> = graph
        .nodes()
        .iter()
        .map(|node| JsonNode {
            id: node.to_string(),
            label: None,
            data: None,
        })
        .collect();

    // Collect edges
    let edges: Vec<JsonEdge> = graph
        .edges()
        .iter()
        .map(|edge| JsonEdge {
            source: edge.source.to_string(),
            target: edge.target.to_string(),
            weight: if weighted {
                // Convert weight to f64 through string parsing
                edge.weight.to_string().parse::<f64>().ok()
            } else {
                None
            },
            label: None,
            data: None,
        })
        .collect();

    // Create JSON graph
    let json_graph = JsonGraph {
        directed: true,
        nodes,
        edges,
        metadata: None,
    };

    // Write JSON to file
    let json_string = serde_json::to_string_pretty(&json_graph)
        .map_err(|e| GraphError::Other(format!("Failed to serialize JSON: {}", e)))?;

    write!(file, "{}", json_string)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_json_undirected() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"{{
                "directed": false,
                "nodes": [
                    {{"id": "1"}},
                    {{"id": "2"}},
                    {{"id": "3"}}
                ],
                "edges": [
                    {{"source": "1", "target": "2"}},
                    {{"source": "2", "target": "3"}}
                ]
            }}"#
        )
        .unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_json_format(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_read_json_directed() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"{{
                "directed": true,
                "nodes": [
                    {{"id": "1"}},
                    {{"id": "2"}},
                    {{"id": "3"}}
                ],
                "edges": [
                    {{"source": "1", "target": "2"}},
                    {{"source": "2", "target": "3"}}
                ]
            }}"#
        )
        .unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> = read_json_format_digraph(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_read_json_weighted() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"{{
                "directed": false,
                "nodes": [
                    {{"id": "1"}},
                    {{"id": "2"}},
                    {{"id": "3"}}
                ],
                "edges": [
                    {{"source": "1", "target": "2", "weight": 1.5}},
                    {{"source": "2", "target": "3", "weight": 2.0}}
                ]
            }}"#
        )
        .unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_json_format(temp_file.path(), true).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let mut original_graph: Graph<i32, f64> = Graph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_json_format(&original_graph, temp_file.path(), true).unwrap();

        let read_graph: Graph<i32, f64> = read_json_format(temp_file.path(), true).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_digraph_write_read_roundtrip() {
        let mut original_graph: DiGraph<i32, f64> = DiGraph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_json_format_digraph(&original_graph, temp_file.path(), true).unwrap();

        let read_graph: DiGraph<i32, f64> =
            read_json_format_digraph(temp_file.path(), true).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_invalid_json() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "{{invalid json").unwrap();
        temp_file.flush().unwrap();

        let result: Result<Graph<i32, f64>> = read_json_format(temp_file.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_node_reference() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"{{
                "directed": false,
                "nodes": [
                    {{"id": "1"}},
                    {{"id": "2"}}
                ],
                "edges": [
                    {{"source": "1", "target": "3"}}
                ]
            }}"#
        )
        .unwrap();
        temp_file.flush().unwrap();

        let result: Result<Graph<i32, f64>> = read_json_format(temp_file.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_directed_graph_mismatch() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            r#"{{
                "directed": true,
                "nodes": [
                    {{"id": "1"}},
                    {{"id": "2"}}
                ],
                "edges": [
                    {{"source": "1", "target": "2"}}
                ]
            }}"#
        )
        .unwrap();
        temp_file.flush().unwrap();

        // Try to read as undirected graph - should fail
        let result: Result<Graph<i32, f64>> = read_json_format(temp_file.path(), false);
        assert!(result.is_err());
    }
}
