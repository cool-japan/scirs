//! Edge list format I/O operations
//!
//! This module provides functions for reading and writing graphs in edge list format.
//! Edge list format is a simple line-based format where each line contains:
//! - For unweighted graphs: `source target`
//! - For weighted graphs: `source target weight`
//!
//! Lines starting with '#' are treated as comments and ignored.
//! Empty lines are also ignored.
//!
//! # Examples
//!
//! ```rust
//! use scirs2_graph::io::edge_list::{read_edge_list_format, write_edge_list_format};
//! use scirs2_graph::base::Graph;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//!
//! // Create a test file
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, "1 2").unwrap();
//! writeln!(temp_file, "2 3").unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read an unweighted graph from edge list format
//! let graph: Graph<i32, f64> = read_edge_list_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.edge_count(), 2);
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Read an undirected graph from edge list format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights (third column)
///
/// # Returns
///
/// Returns a `Graph` with the parsed nodes and edges
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be opened
/// - Lines cannot be parsed
/// - Node or edge weight parsing fails
///
/// # Format
///
/// Each line should contain:
/// - `source target` for unweighted graphs
/// - `source target weight` for weighted graphs
/// - Lines starting with '#' are treated as comments
/// - Empty lines are ignored
/// - Malformed lines are skipped
pub fn read_edge_list_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines
        }

        // Parse source and target nodes
        let source_str = parts[0];
        let target_str = parts[1];

        let source_data = N::from_str(source_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse source node: {}", source_str)))?;
        let target_data = N::from_str(target_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse target node: {}", target_str)))?;

        // Parse edge weight if needed
        let edge_weight = if weighted && parts.len() > 2 {
            E::from_str(parts[2])
                .map_err(|_| GraphError::Other(format!("Cannot parse edge weight: {}", parts[2])))?
        } else {
            E::default()
        };

        // Add edge (this will automatically add nodes if they don't exist)
        graph.add_edge(source_data, target_data, edge_weight)?;
    }

    Ok(graph)
}

/// Write an undirected graph to edge list format
///
/// # Arguments
///
/// * `graph` - The graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in the output
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be created
/// - Writing to the file fails
///
/// # Format
///
/// Each line will contain:
/// - `source target` for unweighted output
/// - `source target weight` for weighted output
pub fn write_edge_list_format<N, E, Ix, P>(
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
    let mut file =
        File::create(path).map_err(|e| GraphError::Other(format!("Cannot create file: {}", e)))?;

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(file, "{} {} {}", edge.source, edge.target, edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing line: {}", e)))?;
        } else {
            writeln!(file, "{} {}", edge.source, edge.target)
                .map_err(|e| GraphError::Other(format!("Error writing line: {}", e)))?;
        }
    }

    Ok(())
}

/// Read a directed graph from edge list format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights (third column)
///
/// # Returns
///
/// Returns a `DiGraph` with the parsed nodes and edges
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be opened
/// - Lines cannot be parsed
/// - Node or edge weight parsing fails
///
/// # Format
///
/// Each line should contain:
/// - `source target` for unweighted graphs
/// - `source target weight` for weighted graphs
/// - Lines starting with '#' are treated as comments
/// - Empty lines are ignored
/// - Malformed lines are skipped
pub fn read_edge_list_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file =
        File::open(path).map_err(|e| GraphError::Other(format!("Cannot open file: {}", e)))?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();

    for line_result in reader.lines() {
        let line =
            line_result.map_err(|e| GraphError::Other(format!("Error reading line: {}", e)))?;
        let line = line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 2 {
            continue; // Skip malformed lines
        }

        // Parse source and target nodes
        let source_str = parts[0];
        let target_str = parts[1];

        let source_data = N::from_str(source_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse source node: {}", source_str)))?;
        let target_data = N::from_str(target_str)
            .map_err(|_| GraphError::Other(format!("Cannot parse target node: {}", target_str)))?;

        // Parse edge weight if needed
        let edge_weight = if weighted && parts.len() > 2 {
            E::from_str(parts[2])
                .map_err(|_| GraphError::Other(format!("Cannot parse edge weight: {}", parts[2])))?
        } else {
            E::default()
        };

        // Add edge (this will automatically add nodes if they don't exist)
        graph.add_edge(source_data, target_data, edge_weight)?;
    }

    Ok(graph)
}

/// Write a directed graph to edge list format
///
/// # Arguments
///
/// * `graph` - The directed graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in the output
///
/// # Errors
///
/// Returns an error if:
/// - The file cannot be created
/// - Writing to the file fails
///
/// # Format
///
/// Each line will contain:
/// - `source target` for unweighted output
/// - `source target weight` for weighted output
pub fn write_edge_list_format_digraph<N, E, Ix, P>(
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
    let mut file =
        File::create(path).map_err(|e| GraphError::Other(format!("Cannot create file: {}", e)))?;

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(file, "{} {} {}", edge.source, edge.target, edge.weight)
                .map_err(|e| GraphError::Other(format!("Error writing line: {}", e)))?;
        } else {
            writeln!(file, "{} {}", edge.source, edge.target)
                .map_err(|e| GraphError::Other(format!("Error writing line: {}", e)))?;
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
    fn test_edge_list_format_unweighted() {
        // Create a temporary file with edge list data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# This is a comment").unwrap();
        writeln!(temp_file, "1 2").unwrap();
        writeln!(temp_file, "2 3").unwrap();
        writeln!(temp_file, "3 1").unwrap();
        writeln!(temp_file).unwrap(); // Empty line
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_edge_list_format(temp_file.path(), false).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_edge_list_format_weighted() {
        // Create a temporary file with weighted edge list data
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1 2 1.5").unwrap();
        writeln!(temp_file, "2 3 2.0").unwrap();
        writeln!(temp_file, "3 1 0.5").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_edge_list_format(temp_file.path(), true).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_digraph_edge_list_format() {
        // Create a temporary file with edge list data for directed graph
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "1 2 1.0").unwrap();
        writeln!(temp_file, "2 3 2.0").unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> =
            read_edge_list_format_digraph(temp_file.path(), true).unwrap();
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }
}
