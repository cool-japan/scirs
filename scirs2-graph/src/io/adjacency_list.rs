//! Adjacency list format I/O for graphs
//!
//! This module provides functionality for reading and writing graphs in adjacency list format.
//! In this format, each line represents a node and all its neighbors.
//!
//! # Format Specification
//!
//! ## Unweighted format:
//! ```text
//! node1: neighbor1 neighbor2 neighbor3
//! node2: neighbor1 neighbor4
//! # Comments start with #
//! node3:
//! ```
//!
//! ## Weighted format:
//! ```text
//! node1: neighbor1 weight1 neighbor2 weight2
//! node2: neighbor1 weight1
//! ```
//!
//! # Examples
//!
//! ```rust
//! use std::fs::File;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//! use scirs2_graph::base::Graph;
//! use scirs2_graph::io::adjacency_list::{read_adjacency_list_format, write_adjacency_list_format};
//!
//! // Create a temporary file with adjacency list data
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, "1: 2 3").unwrap();
//! writeln!(temp_file, "2: 1 3").unwrap();
//! writeln!(temp_file, "3: 1 2").unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read the graph
//! let graph: Graph<i32, i32> = read_adjacency_list_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.edge_count(), 3); // Each edge is only stored once in undirected graph
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Reads an undirected graph from a file in adjacency list format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether the adjacency list includes edge weights
///
/// # Returns
///
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
///
/// # Format
///
/// Each line should contain a node followed by a colon and its neighbors:
/// - Unweighted: `node: neighbor1 neighbor2 neighbor3`
/// - Weighted: `node: neighbor1 weight1 neighbor2 weight2`
///
/// Lines starting with '#' are treated as comments and ignored.
/// Empty lines are also ignored.
pub fn read_adjacency_list_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse the line
        if let Some(colon_pos) = line.find(':') {
            let node_str = line[..colon_pos].trim();
            let neighbors_str = line[colon_pos + 1..].trim();

            // Parse the source node
            let source_node = match N::from_str(node_str) {
                Ok(node) => node,
                Err(_) => {
                    return Err(GraphError::Other(format!(
                        "Failed to parse node '{}' on line {}",
                        node_str,
                        line_num + 1
                    )));
                }
            };

            // Parse neighbors
            if !neighbors_str.is_empty() {
                let tokens: Vec<&str> = neighbors_str.split_whitespace().collect();

                if weighted {
                    // For weighted format: neighbor1 weight1 neighbor2 weight2 ...
                    if tokens.len() % 2 != 0 {
                        return Err(GraphError::Other(format!(
                            "Weighted adjacency list must have even number of tokens (neighbor weight pairs) on line {}",
                            line_num + 1
                        )));
                    }

                    for chunk in tokens.chunks(2) {
                        let neighbor_str = chunk[0];
                        let weight_str = chunk[1];

                        // Parse neighbor node
                        let neighbor_node = match N::from_str(neighbor_str) {
                            Ok(node) => node,
                            Err(_) => {
                                return Err(GraphError::Other(format!(
                                    "Failed to parse neighbor '{}' on line {}",
                                    neighbor_str,
                                    line_num + 1
                                )));
                            }
                        };

                        // Parse weight
                        let weight = match E::from_str(weight_str) {
                            Ok(w) => w,
                            Err(_) => {
                                return Err(GraphError::Other(format!(
                                    "Failed to parse weight '{}' on line {}",
                                    weight_str,
                                    line_num + 1
                                )));
                            }
                        };

                        // Add edge if it doesn't exist (undirected graph)
                        if !graph.has_edge(&source_node, &neighbor_node) {
                            graph.add_edge(source_node.clone(), neighbor_node, weight)?;
                        }
                    }
                } else {
                    // For unweighted format: neighbor1 neighbor2 neighbor3 ...
                    for neighbor_str in tokens {
                        // Parse neighbor node
                        let neighbor_node = match N::from_str(neighbor_str) {
                            Ok(node) => node,
                            Err(_) => {
                                return Err(GraphError::Other(format!(
                                    "Failed to parse neighbor '{}' on line {}",
                                    neighbor_str,
                                    line_num + 1
                                )));
                            }
                        };

                        // Add edge if it doesn't exist (undirected graph)
                        if !graph.has_edge(&source_node, &neighbor_node) {
                            graph.add_edge(source_node.clone(), neighbor_node, E::default())?;
                        }
                    }
                }
            }
        } else {
            return Err(GraphError::Other(format!(
                "Missing ':' separator on line {}",
                line_num + 1
            )));
        }
    }

    Ok(graph)
}

/// Reads a directed graph from a file in adjacency list format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether the adjacency list includes edge weights
///
/// # Returns
///
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
pub fn read_adjacency_list_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Parse the line
        if let Some(colon_pos) = line.find(':') {
            let node_str = line[..colon_pos].trim();
            let neighbors_str = line[colon_pos + 1..].trim();

            // Parse the source node
            let source_node = match N::from_str(node_str) {
                Ok(node) => node,
                Err(_) => {
                    return Err(GraphError::Other(format!(
                        "Failed to parse node '{}' on line {}",
                        node_str,
                        line_num + 1
                    )));
                }
            };

            // Parse neighbors
            if !neighbors_str.is_empty() {
                let tokens: Vec<&str> = neighbors_str.split_whitespace().collect();

                if weighted {
                    // For weighted format: neighbor1 weight1 neighbor2 weight2 ...
                    if tokens.len() % 2 != 0 {
                        return Err(GraphError::Other(format!(
                            "Weighted adjacency list must have even number of tokens (neighbor weight pairs) on line {}",
                            line_num + 1
                        )));
                    }

                    for chunk in tokens.chunks(2) {
                        let neighbor_str = chunk[0];
                        let weight_str = chunk[1];

                        // Parse neighbor node
                        let neighbor_node = match N::from_str(neighbor_str) {
                            Ok(node) => node,
                            Err(_) => {
                                return Err(GraphError::Other(format!(
                                    "Failed to parse neighbor '{}' on line {}",
                                    neighbor_str,
                                    line_num + 1
                                )));
                            }
                        };

                        // Parse weight
                        let weight = match E::from_str(weight_str) {
                            Ok(w) => w,
                            Err(_) => {
                                return Err(GraphError::Other(format!(
                                    "Failed to parse weight '{}' on line {}",
                                    weight_str,
                                    line_num + 1
                                )));
                            }
                        };

                        // Add directed edge
                        graph.add_edge(source_node.clone(), neighbor_node, weight)?;
                    }
                } else {
                    // For unweighted format: neighbor1 neighbor2 neighbor3 ...
                    for neighbor_str in tokens {
                        // Parse neighbor node
                        let neighbor_node = match N::from_str(neighbor_str) {
                            Ok(node) => node,
                            Err(_) => {
                                return Err(GraphError::Other(format!(
                                    "Failed to parse neighbor '{}' on line {}",
                                    neighbor_str,
                                    line_num + 1
                                )));
                            }
                        };

                        // Add directed edge
                        graph.add_edge(source_node.clone(), neighbor_node, E::default())?;
                    }
                }
            }
        } else {
            return Err(GraphError::Other(format!(
                "Missing ':' separator on line {}",
                line_num + 1
            )));
        }
    }

    Ok(graph)
}

/// Writes an undirected graph to a file in adjacency list format
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
pub fn write_adjacency_list_format<N, E, Ix, P>(
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

    writeln!(file, "# Adjacency list format")?;
    writeln!(file, "# node: neighbor1 neighbor2 ...")?;
    if weighted {
        writeln!(file, "# Format: node: neighbor1 weight1 neighbor2 weight2")?;
    }
    writeln!(file)?;

    // Get all nodes and their neighbors
    let all_edges = graph.edges();
    let all_nodes = graph.nodes();

    // Group edges by source node
    let mut node_neighbors = std::collections::HashMap::new();
    for edge in all_edges {
        let source = &edge.source;
        let target = &edge.target;
        let weight = &edge.weight;

        node_neighbors
            .entry(source.clone())
            .or_insert_with(Vec::new)
            .push((target.clone(), *weight));
    }

    // Write adjacency list for each node
    for node in all_nodes {
        write!(file, "{}: ", node)?;

        if let Some(neighbors) = node_neighbors.get(node) {
            let neighbor_strs: Vec<String> = neighbors
                .iter()
                .map(|(neighbor, weight)| {
                    if weighted {
                        format!("{} {}", neighbor, weight)
                    } else {
                        format!("{}", neighbor)
                    }
                })
                .collect();

            if !neighbor_strs.is_empty() {
                writeln!(file, "{}", neighbor_strs.join(" "))?;
            } else {
                writeln!(file)?;
            }
        } else {
            writeln!(file)?;
        }
    }

    Ok(())
}

/// Writes a directed graph to a file in adjacency list format
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
pub fn write_adjacency_list_format_digraph<N, E, Ix, P>(
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

    writeln!(file, "# Directed adjacency list format")?;
    writeln!(file, "# node: neighbor1 neighbor2 ...")?;
    if weighted {
        writeln!(file, "# Format: node: neighbor1 weight1 neighbor2 weight2")?;
    }
    writeln!(file)?;

    // Get all edges and nodes
    let all_edges = graph.edges();
    let all_nodes = graph.nodes();

    // Group edges by source node (for directed graph, only outgoing edges)
    let mut node_neighbors = std::collections::HashMap::new();
    for edge in all_edges {
        let source = &edge.source;
        let target = &edge.target;
        let weight = &edge.weight;

        node_neighbors
            .entry(source.clone())
            .or_insert_with(Vec::new)
            .push((target.clone(), *weight));
    }

    // Write adjacency list for each node
    for node in all_nodes {
        write!(file, "{}: ", node)?;

        if let Some(neighbors) = node_neighbors.get(node) {
            let neighbor_strs: Vec<String> = neighbors
                .iter()
                .map(|(neighbor, weight)| {
                    if weighted {
                        format!("{} {}", neighbor, weight)
                    } else {
                        format!("{}", neighbor)
                    }
                })
                .collect();

            if !neighbor_strs.is_empty() {
                writeln!(file, "{}", neighbor_strs.join(" "))?;
            } else {
                writeln!(file)?;
            }
        } else {
            writeln!(file)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_unweighted_adjacency_list() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# Test adjacency list").unwrap();
        writeln!(temp_file, "1: 2 3").unwrap();
        writeln!(temp_file, "2: 1 3").unwrap();
        writeln!(temp_file, "3: 1 2").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3); // Each undirected edge stored once
    }

    #[test]
    fn test_read_weighted_adjacency_list() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1: 2 0.5 3 1.0").unwrap();
        writeln!(temp_file, "2: 1 0.5").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), true).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_read_directed_adjacency_list() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1: 2 3").unwrap();
        writeln!(temp_file, "2: 3").unwrap();
        writeln!(temp_file, "3:").unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> =
            read_adjacency_list_format_digraph(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3); // Directed edges
    }

    #[test]
    fn test_write_read_roundtrip() {
        let mut original_graph: Graph<i32, f64> = Graph::new();
        original_graph.add_edge(1i32, 2i32, 0.0f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 0.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_adjacency_list_format(&original_graph, temp_file.path(), false).unwrap();

        let read_graph: Graph<i32, f64> =
            read_adjacency_list_format(temp_file.path(), false).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_malformed_line_handling() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1 2 3").unwrap(); // Missing colon
        temp_file.flush().unwrap();

        let result: Result<Graph<i32, f64>> = read_adjacency_list_format(temp_file.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();

        let graph: Graph<i32, f64> = read_adjacency_list_format(temp_file.path(), false).unwrap();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }
}
