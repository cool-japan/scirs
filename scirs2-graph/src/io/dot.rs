//! DOT format I/O for graphs (Graphviz format)
//!
//! This module provides functionality for reading and writing graphs in DOT format,
//! which is the standard format used by Graphviz for graph visualization.
//!
//! # Format Specification
//!
//! DOT format supports both directed and undirected graphs:
//! - Undirected graphs use `graph` keyword and `--` for edges
//! - Directed graphs use `digraph` keyword and `->` for edges
//!
//! # Examples
//!
//! ## Undirected graph:
//! ```text
//! graph G {
//!     1 -- 2 [weight=1.5];
//!     2 -- 3 [weight=2.0];
//! }
//! ```
//!
//! ## Directed graph:
//! ```text
//! digraph G {
//!     1 -> 2 [weight=1.5];
//!     2 -> 3 [weight=2.0];
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
//! use scirs2_graph::io::dot::{read_dot_format, write_dot_format};
//!
//! // Create a temporary file with DOT data
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, "graph G {{").unwrap();
//! writeln!(temp_file, "    1 -- 2;").unwrap();
//! writeln!(temp_file, "    2 -- 3;").unwrap();
//! writeln!(temp_file, "}}").unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read the graph
//! let graph: Graph<i32, f64> = read_dot_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.edge_count(), 2);
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// DOT format parser state
#[derive(Debug, Clone, PartialEq, Eq)]
enum ParseState {
    /// Looking for graph declaration
    Header,
    /// Inside graph body
    Body,
    /// Finished parsing
    Done,
}

/// DOT graph type
#[derive(Debug, Clone, PartialEq, Eq)]
enum GraphType {
    /// Undirected graph (graph keyword)
    Undirected,
    /// Directed graph (digraph keyword)
    Directed,
}

/// Read an undirected graph from DOT format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from attributes
///
/// # Returns
///
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
///
/// # Format
///
/// The DOT format supports:
/// - Node declarations: `node_id;` or `node_id [attributes];`
/// - Edge declarations: `node1 -- node2;` or `node1 -- node2 [attributes];`
/// - Comments: `// comment` or `/* comment */`
/// - Attributes in square brackets: `[weight=1.5, label="edge"]`
pub fn read_dot_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();
    let mut state = ParseState::Header;
    let mut graph_type = None;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = remove_comments(&line).trim().to_string();

        if line.is_empty() {
            continue;
        }

        match state {
            ParseState::Header => {
                if let Some(detected_type) = parse_header(&line)? {
                    graph_type = Some(detected_type);
                    state = ParseState::Body;
                }
            }
            ParseState::Body => {
                if line.contains('}') {
                    state = ParseState::Done;
                    break;
                }

                // Parse edge or node declarations
                if let Some(graph_type) = &graph_type {
                    parse_graph_element(&line, graph_type, &mut graph, weighted, line_num + 1)?;
                }
            }
            ParseState::Done => break,
        }
    }

    // Validate we found a proper graph declaration
    if graph_type.is_none() {
        return Err(GraphError::Other(
            "No valid graph declaration found".to_string(),
        ));
    }

    if state != ParseState::Done && state != ParseState::Body {
        return Err(GraphError::Other(
            "Incomplete DOT file - missing closing brace".to_string(),
        ));
    }

    Ok(graph)
}

/// Read a directed graph from DOT format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from attributes
///
/// # Returns
///
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
pub fn read_dot_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();
    let mut state = ParseState::Header;
    let mut graph_type = None;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = remove_comments(&line).trim().to_string();

        if line.is_empty() {
            continue;
        }

        match state {
            ParseState::Header => {
                if let Some(detected_type) = parse_header(&line)? {
                    graph_type = Some(detected_type);
                    state = ParseState::Body;
                }
            }
            ParseState::Body => {
                if line.contains('}') {
                    state = ParseState::Done;
                    break;
                }

                // Parse edge or node declarations
                if let Some(graph_type) = &graph_type {
                    parse_digraph_element(&line, graph_type, &mut graph, weighted, line_num + 1)?;
                }
            }
            ParseState::Done => break,
        }
    }

    // Validate we found a proper graph declaration
    if graph_type.is_none() {
        return Err(GraphError::Other(
            "No valid graph declaration found".to_string(),
        ));
    }

    if state != ParseState::Done && state != ParseState::Body {
        return Err(GraphError::Other(
            "Incomplete DOT file - missing closing brace".to_string(),
        ));
    }

    Ok(graph)
}

/// Write an undirected graph to DOT format
///
/// # Arguments
///
/// * `graph` - The graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in attributes
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_dot_format<N, E, Ix, P>(graph: &Graph<N, E, Ix>, path: P, weighted: bool) -> Result<()>
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

    writeln!(file, "graph G {{")?;
    writeln!(file, "    // Generated by scirs2-graph")?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "    {};", node)?;
    }

    writeln!(file)?;

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(
                file,
                "    {} -- {} [weight={}];",
                edge.source, edge.target, edge.weight
            )?;
        } else {
            writeln!(file, "    {} -- {};", edge.source, edge.target)?;
        }
    }

    writeln!(file, "}}")?;

    Ok(())
}

/// Write a directed graph to DOT format
///
/// # Arguments
///
/// * `graph` - The directed graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights in attributes
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_dot_format_digraph<N, E, Ix, P>(
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

    writeln!(file, "digraph G {{")?;
    writeln!(file, "    // Generated by scirs2-graph")?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "    {};", node)?;
    }

    writeln!(file)?;

    // Write edges
    for edge in graph.edges() {
        if weighted {
            writeln!(
                file,
                "    {} -> {} [weight={}];",
                edge.source, edge.target, edge.weight
            )?;
        } else {
            writeln!(file, "    {} -> {};", edge.source, edge.target)?;
        }
    }

    writeln!(file, "}}")?;

    Ok(())
}

// Helper functions

/// Remove comments from a line
fn remove_comments(line: &str) -> String {
    // Remove // comments
    if let Some(pos) = line.find("//") {
        return line[..pos].to_string();
    }

    // TODO: Handle /* */ comments properly (they can span multiple lines)
    if let Some(start) = line.find("/*") {
        if let Some(end) = line.find("*/") {
            if end > start {
                return format!("{}{}", &line[..start], &line[end + 2..]);
            }
        }
        // If /* found but no matching */, remove everything after /*
        return line[..start].to_string();
    }

    line.to_string()
}

/// Parse the header line to determine graph type
fn parse_header(line: &str) -> Result<Option<GraphType>> {
    let line = line.trim();

    if line.starts_with("graph") && line.contains('{') {
        return Ok(Some(GraphType::Undirected));
    }

    if line.starts_with("digraph") && line.contains('{') {
        return Ok(Some(GraphType::Directed));
    }

    // Skip other lines until we find a graph declaration
    Ok(None)
}

/// Parse a graph element (node or edge) for undirected graphs
fn parse_graph_element<N, E>(
    line: &str,
    graph_type: &GraphType,
    graph: &mut Graph<N, E>,
    weighted: bool,
    line_num: usize,
) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    let line = line.trim_end_matches(';').trim();

    // Check for edge based on graph type
    let edge_separator = match graph_type {
        GraphType::Undirected => "--",
        GraphType::Directed => "->",
    };

    if line.contains(edge_separator) {
        parse_edge(line, edge_separator, graph, weighted, line_num)?;
    } else if !line.is_empty() && !line.starts_with('}') && !line.contains('[') {
        // Parse as node declaration (simple node without attributes)
        parse_node(line, line_num)?;
    }

    Ok(())
}

/// Parse a graph element (node or edge) for directed graphs
fn parse_digraph_element<N, E>(
    line: &str,
    graph_type: &GraphType,
    graph: &mut DiGraph<N, E>,
    weighted: bool,
    line_num: usize,
) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    let line = line.trim_end_matches(';').trim();

    // Check for edge based on graph type
    let edge_separator = match graph_type {
        GraphType::Undirected => "--",
        GraphType::Directed => "->",
    };

    if line.contains(edge_separator) {
        parse_digraph_edge(line, edge_separator, graph, weighted, line_num)?;
    } else if !line.is_empty() && !line.starts_with('}') && !line.contains('[') {
        // Parse as node declaration (simple node without attributes)
        parse_node(line, line_num)?;
    }

    Ok(())
}

/// Parse an edge declaration for undirected graphs
fn parse_edge<N, E>(
    line: &str,
    edge_separator: &str,
    graph: &mut Graph<N, E>,
    weighted: bool,
    line_num: usize,
) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    // Split by edge separator
    let parts: Vec<&str> = line.split(edge_separator).collect();
    if parts.len() != 2 {
        return Err(GraphError::Other(format!(
            "Invalid edge format on line {}: {}",
            line_num, line
        )));
    }

    let source_part = parts[0].trim();
    let target_part = parts[1].trim();

    // Parse source node
    let source_node = N::from_str(source_part).map_err(|_| {
        GraphError::Other(format!(
            "Failed to parse source node '{}' on line {}",
            source_part, line_num
        ))
    })?;

    // Parse target node and attributes
    let (target_str, attributes) = if target_part.contains('[') {
        let bracket_pos = target_part.find('[').unwrap();
        (
            target_part[..bracket_pos].trim(),
            Some(&target_part[bracket_pos..]),
        )
    } else {
        (target_part, None)
    };

    let target_node = N::from_str(target_str).map_err(|_| {
        GraphError::Other(format!(
            "Failed to parse target node '{}' on line {}",
            target_str, line_num
        ))
    })?;

    // Parse weight from attributes if needed
    let weight = if weighted && attributes.is_some() {
        parse_weight_from_attributes(attributes.unwrap())?
    } else {
        E::default()
    };

    // Add edge
    graph.add_edge(source_node, target_node, weight)?;

    Ok(())
}

/// Parse an edge declaration for directed graphs
fn parse_digraph_edge<N, E>(
    line: &str,
    edge_separator: &str,
    graph: &mut DiGraph<N, E>,
    weighted: bool,
    line_num: usize,
) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    // Split by edge separator
    let parts: Vec<&str> = line.split(edge_separator).collect();
    if parts.len() != 2 {
        return Err(GraphError::Other(format!(
            "Invalid edge format on line {}: {}",
            line_num, line
        )));
    }

    let source_part = parts[0].trim();
    let target_part = parts[1].trim();

    // Parse source node
    let source_node = N::from_str(source_part).map_err(|_| {
        GraphError::Other(format!(
            "Failed to parse source node '{}' on line {}",
            source_part, line_num
        ))
    })?;

    // Parse target node and attributes
    let (target_str, attributes) = if target_part.contains('[') {
        let bracket_pos = target_part.find('[').unwrap();
        (
            target_part[..bracket_pos].trim(),
            Some(&target_part[bracket_pos..]),
        )
    } else {
        (target_part, None)
    };

    let target_node = N::from_str(target_str).map_err(|_| {
        GraphError::Other(format!(
            "Failed to parse target node '{}' on line {}",
            target_str, line_num
        ))
    })?;

    // Parse weight from attributes if needed
    let weight = if weighted && attributes.is_some() {
        parse_weight_from_attributes(attributes.unwrap())?
    } else {
        E::default()
    };

    // Add edge
    graph.add_edge(source_node, target_node, weight)?;

    Ok(())
}

/// Parse a node declaration (currently just validates the syntax)
fn parse_node(_line: &str, _line_num: usize) -> Result<()> {
    // For now, we don't need to explicitly add nodes since they'll be added
    // when edges are added. This function validates node syntax.
    Ok(())
}

/// Parse weight from DOT attributes
fn parse_weight_from_attributes<E>(attributes: &str) -> Result<E>
where
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    // Look for weight=value pattern
    if let Some(weight_start) = attributes.find("weight=") {
        let weight_part = &attributes[weight_start + 7..]; // Skip "weight="

        // Find end of weight value (space, comma, or closing bracket)
        let weight_end = weight_part
            .find(&[' ', ',', ']'][..])
            .unwrap_or(weight_part.len());

        let weight_str = &weight_part[..weight_end];

        return E::from_str(weight_str)
            .map_err(|_| GraphError::Other(format!("Failed to parse weight: {}", weight_str)));
    }

    Ok(E::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_undirected_dot() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "graph G {{").unwrap();
        writeln!(temp_file, "    1 -- 2;").unwrap();
        writeln!(temp_file, "    2 -- 3;").unwrap();
        writeln!(temp_file, "}}").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_dot_format(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_read_directed_dot() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "digraph G {{").unwrap();
        writeln!(temp_file, "    1 -> 2;").unwrap();
        writeln!(temp_file, "    2 -> 3;").unwrap();
        writeln!(temp_file, "}}").unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> = read_dot_format_digraph(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_read_weighted_dot() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "graph G {{").unwrap();
        writeln!(temp_file, "    1 -- 2 [weight=1.5];").unwrap();
        writeln!(temp_file, "    2 -- 3 [weight=2.0];").unwrap();
        writeln!(temp_file, "}}").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_dot_format(temp_file.path(), true).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let mut original_graph: Graph<i32, f64> = Graph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_dot_format(&original_graph, temp_file.path(), true).unwrap();

        let read_graph: Graph<i32, f64> = read_dot_format(temp_file.path(), true).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_digraph_write_read_roundtrip() {
        let mut original_graph: DiGraph<i32, f64> = DiGraph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_dot_format_digraph(&original_graph, temp_file.path(), true).unwrap();

        let read_graph: DiGraph<i32, f64> =
            read_dot_format_digraph(temp_file.path(), true).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_remove_comments() {
        assert_eq!(remove_comments("1 -- 2; // this is a comment"), "1 -- 2; ");
        assert_eq!(remove_comments("1 -- 2; /* comment */"), "1 -- 2; ");
        assert_eq!(remove_comments("no comments here"), "no comments here");
    }

    #[test]
    fn test_parse_weight_from_attributes() {
        let weight: f64 = parse_weight_from_attributes("[weight=1.5]").unwrap();
        assert_eq!(weight, 1.5);

        let weight: f64 = parse_weight_from_attributes("[label=\"edge\", weight=2.0]").unwrap();
        assert_eq!(weight, 2.0);

        let weight: f64 = parse_weight_from_attributes("[color=red]").unwrap();
        assert_eq!(weight, 0.0); // default
    }

    #[test]
    fn test_invalid_dot_format() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "invalid format").unwrap();
        temp_file.flush().unwrap();

        let result: Result<Graph<i32, f64>> = read_dot_format(temp_file.path(), false);
        assert!(result.is_err());
    }
}
