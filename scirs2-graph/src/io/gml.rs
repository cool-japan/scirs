//! GML (Graph Modeling Language) format I/O for graphs
//!
//! This module provides functionality for reading and writing graphs in GML format.
//! GML is a hierarchical ASCII-based file format for describing graphs with a simple
//! attribute-value syntax, widely used in graph visualization tools like yEd.
//!
//! # Format Specification
//!
//! GML uses a hierarchical structure with the following elements:
//! - `graph` - Root element containing the entire graph
//! - `directed` - Boolean indicating if the graph is directed (0/1)
//! - `node` - Node elements with `id` and optional attributes
//! - `edge` - Edge elements with `source`, `target`, and optional attributes
//! - Attributes can be strings, integers, or floats
//!
//! # Examples
//!
//! ## Basic GML structure:
//! ```text
//! graph [
//!   directed 0
//!   node [
//!     id 1
//!     label "Node 1"
//!   ]
//!   node [
//!     id 2
//!     label "Node 2"
//!   ]
//!   edge [
//!     source 1
//!     target 2
//!     weight 1.5
//!   ]
//! ]
//! ```
//!
//! ## Weighted directed graph:
//! ```text
//! graph [
//!   directed 1
//!   node [
//!     id 1
//!   ]
//!   node [
//!     id 2
//!   ]
//!   edge [
//!     source 1
//!     target 2
//!     weight 2.0
//!   ]
//! ]
//! ```
//!
//! # Usage
//!
//! ```rust
//! use std::fs::File;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//! use scirs2_graph::base::Graph;
//! use scirs2_graph::io::gml::{read_gml_format, write_gml_format};
//!
//! // Create a temporary file with GML data
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, "graph [").unwrap();
//! writeln!(temp_file, "  directed 0").unwrap();
//! writeln!(temp_file, "  node [ id 1 ]").unwrap();
//! writeln!(temp_file, "  node [ id 2 ]").unwrap();
//! writeln!(temp_file, "  edge [ source 1 target 2 ]").unwrap();
//! writeln!(temp_file, "]").unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read the graph
//! let graph: Graph<i32, f64> = read_gml_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 2);
//! assert_eq!(graph.edge_count(), 1);
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// GML parser state machine
#[derive(Debug, Clone, PartialEq, Eq)]
enum ParseState {
    /// Looking for graph opening
    SearchingGraph,
    /// Inside graph, parsing top-level attributes
    InGraph,
    /// Inside a node block
    InNode,
    /// Inside an edge block
    InEdge,
    /// Finished parsing
    Done,
}

/// GML token types for parsing
#[derive(Debug, Clone, PartialEq)]
enum GmlToken {
    /// Opening bracket [
    OpenBracket,
    /// Closing bracket ]
    CloseBracket,
    /// Identifier (keyword)
    Identifier(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
}

/// Simple GML lexer
struct GmlLexer {
    input: String,
    position: usize,
}

impl GmlLexer {
    fn new(input: String) -> Self {
        Self { input, position: 0 }
    }

    fn next_token(&mut self) -> Option<GmlToken> {
        self.skip_whitespace_and_comments();

        if self.position >= self.input.len() {
            return None;
        }

        let remaining = &self.input[self.position..];

        // Handle brackets
        if remaining.starts_with('[') {
            self.position += 1;
            return Some(GmlToken::OpenBracket);
        }
        if remaining.starts_with(']') {
            self.position += 1;
            return Some(GmlToken::CloseBracket);
        }

        // Handle strings (quoted)
        if remaining.starts_with('"') {
            return self.parse_string();
        }

        // Handle numbers and identifiers
        self.parse_number_or_identifier()
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.position < self.input.len() {
            let ch = self.input.chars().nth(self.position).unwrap();
            if ch.is_whitespace() {
                self.position += 1;
            } else if ch == '#' {
                // Skip comment line
                while self.position < self.input.len() {
                    let ch = self.input.chars().nth(self.position).unwrap();
                    self.position += 1;
                    if ch == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn parse_string(&mut self) -> Option<GmlToken> {
        self.position += 1; // Skip opening quote
        let start = self.position;

        while self.position < self.input.len() {
            let ch = self.input.chars().nth(self.position).unwrap();
            if ch == '"' {
                let value = self.input[start..self.position].to_string();
                self.position += 1; // Skip closing quote
                return Some(GmlToken::String(value));
            }
            self.position += 1;
        }

        None // Unterminated string
    }

    fn parse_number_or_identifier(&mut self) -> Option<GmlToken> {
        let start = self.position;

        while self.position < self.input.len() {
            let ch = self.input.chars().nth(self.position).unwrap();
            if ch.is_alphanumeric() || ch == '_' || ch == '.' || ch == '-' {
                self.position += 1;
            } else {
                break;
            }
        }

        let value = &self.input[start..self.position];

        // Try to parse as number
        if let Ok(int_val) = value.parse::<i64>() {
            return Some(GmlToken::Integer(int_val));
        }
        if let Ok(float_val) = value.parse::<f64>() {
            return Some(GmlToken::Float(float_val));
        }

        // Otherwise it's an identifier
        Some(GmlToken::Identifier(value.to_string()))
    }
}

/// Read an undirected graph from GML format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from GML attributes
///
/// # Returns
///
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
///
/// # Format
///
/// The GML format supports:
/// - Hierarchical attribute-value structure
/// - Node and edge attributes
/// - Both directed and undirected graphs
/// - Comments starting with #
/// - String, integer, and float values
pub fn read_gml_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let content = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?
        .join("\n");

    let mut lexer = GmlLexer::new(content);
    let mut graph = Graph::new();
    let mut state = ParseState::SearchingGraph;
    let mut is_directed = false;
    let mut _current_node_id: Option<String> = None;
    let mut current_edge_source: Option<String> = None;
    let mut current_edge_target: Option<String> = None;
    let mut current_edge_weight: Option<String> = None;

    while let Some(token) = lexer.next_token() {
        match (&state, &token) {
            (ParseState::SearchingGraph, GmlToken::Identifier(id)) if id == "graph" => {
                // Expect opening bracket
                if let Some(GmlToken::OpenBracket) = lexer.next_token() {
                    state = ParseState::InGraph;
                } else {
                    return Err(GraphError::Other("Expected '[' after 'graph'".to_string()));
                }
            }
            (ParseState::InGraph, GmlToken::Identifier(id)) => {
                match id.as_str() {
                    "directed" => {
                        if let Some(GmlToken::Integer(val)) = lexer.next_token() {
                            is_directed = val != 0;
                        }
                    }
                    "node" => {
                        if let Some(GmlToken::OpenBracket) = lexer.next_token() {
                            state = ParseState::InNode;
                            _current_node_id = None;
                        }
                    }
                    "edge" => {
                        if let Some(GmlToken::OpenBracket) = lexer.next_token() {
                            state = ParseState::InEdge;
                            current_edge_source = None;
                            current_edge_target = None;
                            current_edge_weight = None;
                        }
                    }
                    _ => {
                        // Skip unknown attributes
                        lexer.next_token();
                    }
                }
            }
            (ParseState::InNode, GmlToken::Identifier(id)) => {
                match id.as_str() {
                    "id" => {
                        if let Some(token) = lexer.next_token() {
                            _current_node_id = match token {
                                GmlToken::Integer(val) => Some(val.to_string()),
                                GmlToken::String(val) => Some(val),
                                GmlToken::Identifier(val) => Some(val),
                                _ => None,
                            };
                        }
                    }
                    _ => {
                        // Skip other node attributes
                        lexer.next_token();
                    }
                }
            }
            (ParseState::InEdge, GmlToken::Identifier(id)) => {
                match id.as_str() {
                    "source" => {
                        if let Some(token) = lexer.next_token() {
                            current_edge_source = match token {
                                GmlToken::Integer(val) => Some(val.to_string()),
                                GmlToken::String(val) => Some(val),
                                GmlToken::Identifier(val) => Some(val),
                                _ => None,
                            };
                        }
                    }
                    "target" => {
                        if let Some(token) = lexer.next_token() {
                            current_edge_target = match token {
                                GmlToken::Integer(val) => Some(val.to_string()),
                                GmlToken::String(val) => Some(val),
                                GmlToken::Identifier(val) => Some(val),
                                _ => None,
                            };
                        }
                    }
                    "weight" => {
                        if weighted {
                            if let Some(token) = lexer.next_token() {
                                current_edge_weight = match token {
                                    GmlToken::Float(val) => Some(val.to_string()),
                                    GmlToken::Integer(val) => Some(val.to_string()),
                                    GmlToken::String(val) => Some(val),
                                    _ => None,
                                };
                            }
                        } else {
                            lexer.next_token(); // Skip value
                        }
                    }
                    _ => {
                        // Skip other edge attributes
                        lexer.next_token();
                    }
                }
            }
            (ParseState::InNode, GmlToken::CloseBracket) => {
                // Node complete - nodes are added automatically when edges are created
                state = ParseState::InGraph;
            }
            (ParseState::InEdge, GmlToken::CloseBracket) => {
                // Edge complete - add to graph
                if let (Some(source_str), Some(target_str)) =
                    (&current_edge_source, &current_edge_target)
                {
                    let source_node = N::from_str(source_str).map_err(|_| {
                        GraphError::Other(format!("Failed to parse source node: {}", source_str))
                    })?;
                    let target_node = N::from_str(target_str).map_err(|_| {
                        GraphError::Other(format!("Failed to parse target node: {}", target_str))
                    })?;

                    let weight = if let Some(weight_str) = &current_edge_weight {
                        E::from_str(weight_str).map_err(|_| {
                            GraphError::Other(format!(
                                "Failed to parse edge weight: {}",
                                weight_str
                            ))
                        })?
                    } else {
                        E::default()
                    };

                    graph.add_edge(source_node, target_node, weight)?;
                }
                state = ParseState::InGraph;
            }
            (ParseState::InGraph, GmlToken::CloseBracket) => {
                state = ParseState::Done;
                break;
            }
            _ => {
                // Ignore other tokens
            }
        }
    }

    // Verify we found a valid graph structure
    if state == ParseState::SearchingGraph {
        return Err(GraphError::Other(
            "No valid GML graph structure found".to_string(),
        ));
    }

    // Check if this was actually a directed graph but we're reading as undirected
    if is_directed {
        return Err(GraphError::Other(
            "GML file contains a directed graph, but undirected graph was requested".to_string(),
        ));
    }

    Ok(graph)
}

/// Read a directed graph from GML format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from GML attributes
///
/// # Returns
///
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
pub fn read_gml_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let content = reader
        .lines()
        .collect::<std::io::Result<Vec<_>>>()?
        .join("\n");

    let mut lexer = GmlLexer::new(content);
    let mut graph = DiGraph::new();
    let mut state = ParseState::SearchingGraph;
    let mut _current_node_id: Option<String> = None;
    let mut current_edge_source: Option<String> = None;
    let mut current_edge_target: Option<String> = None;
    let mut current_edge_weight: Option<String> = None;

    while let Some(token) = lexer.next_token() {
        match (&state, &token) {
            (ParseState::SearchingGraph, GmlToken::Identifier(id)) if id == "graph" => {
                // Expect opening bracket
                if let Some(GmlToken::OpenBracket) = lexer.next_token() {
                    state = ParseState::InGraph;
                } else {
                    return Err(GraphError::Other("Expected '[' after 'graph'".to_string()));
                }
            }
            (ParseState::InGraph, GmlToken::Identifier(id)) => {
                match id.as_str() {
                    "directed" => {
                        // Skip the value - we assume directed graph for DiGraph
                        lexer.next_token();
                    }
                    "node" => {
                        if let Some(GmlToken::OpenBracket) = lexer.next_token() {
                            state = ParseState::InNode;
                            _current_node_id = None;
                        }
                    }
                    "edge" => {
                        if let Some(GmlToken::OpenBracket) = lexer.next_token() {
                            state = ParseState::InEdge;
                            current_edge_source = None;
                            current_edge_target = None;
                            current_edge_weight = None;
                        }
                    }
                    _ => {
                        // Skip unknown attributes
                        lexer.next_token();
                    }
                }
            }
            (ParseState::InNode, GmlToken::Identifier(id)) => {
                match id.as_str() {
                    "id" => {
                        if let Some(token) = lexer.next_token() {
                            _current_node_id = match token {
                                GmlToken::Integer(val) => Some(val.to_string()),
                                GmlToken::String(val) => Some(val),
                                GmlToken::Identifier(val) => Some(val),
                                _ => None,
                            };
                        }
                    }
                    _ => {
                        // Skip other node attributes
                        lexer.next_token();
                    }
                }
            }
            (ParseState::InEdge, GmlToken::Identifier(id)) => {
                match id.as_str() {
                    "source" => {
                        if let Some(token) = lexer.next_token() {
                            current_edge_source = match token {
                                GmlToken::Integer(val) => Some(val.to_string()),
                                GmlToken::String(val) => Some(val),
                                GmlToken::Identifier(val) => Some(val),
                                _ => None,
                            };
                        }
                    }
                    "target" => {
                        if let Some(token) = lexer.next_token() {
                            current_edge_target = match token {
                                GmlToken::Integer(val) => Some(val.to_string()),
                                GmlToken::String(val) => Some(val),
                                GmlToken::Identifier(val) => Some(val),
                                _ => None,
                            };
                        }
                    }
                    "weight" => {
                        if weighted {
                            if let Some(token) = lexer.next_token() {
                                current_edge_weight = match token {
                                    GmlToken::Float(val) => Some(val.to_string()),
                                    GmlToken::Integer(val) => Some(val.to_string()),
                                    GmlToken::String(val) => Some(val),
                                    _ => None,
                                };
                            }
                        } else {
                            lexer.next_token(); // Skip value
                        }
                    }
                    _ => {
                        // Skip other edge attributes
                        lexer.next_token();
                    }
                }
            }
            (ParseState::InNode, GmlToken::CloseBracket) => {
                // Node complete
                state = ParseState::InGraph;
            }
            (ParseState::InEdge, GmlToken::CloseBracket) => {
                // Edge complete - add to graph
                if let (Some(source_str), Some(target_str)) =
                    (&current_edge_source, &current_edge_target)
                {
                    let source_node = N::from_str(source_str).map_err(|_| {
                        GraphError::Other(format!("Failed to parse source node: {}", source_str))
                    })?;
                    let target_node = N::from_str(target_str).map_err(|_| {
                        GraphError::Other(format!("Failed to parse target node: {}", target_str))
                    })?;

                    let weight = if let Some(weight_str) = &current_edge_weight {
                        E::from_str(weight_str).map_err(|_| {
                            GraphError::Other(format!(
                                "Failed to parse edge weight: {}",
                                weight_str
                            ))
                        })?
                    } else {
                        E::default()
                    };

                    graph.add_edge(source_node, target_node, weight)?;
                }
                state = ParseState::InGraph;
            }
            (ParseState::InGraph, GmlToken::CloseBracket) => {
                state = ParseState::Done;
                break;
            }
            _ => {
                // Ignore other tokens
            }
        }
    }

    // Verify we found a valid graph structure
    if state == ParseState::SearchingGraph {
        return Err(GraphError::Other(
            "No valid GML graph structure found".to_string(),
        ));
    }

    Ok(graph)
}

/// Write an undirected graph to GML format
///
/// # Arguments
///
/// * `graph` - The graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights as attributes
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_gml_format<N, E, Ix, P>(graph: &Graph<N, E, Ix>, path: P, weighted: bool) -> Result<()>
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

    // Write GML header
    writeln!(file, "# Generated by scirs2-graph")?;
    writeln!(file, "graph [")?;
    writeln!(file, "  directed 0")?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "  node [")?;
        writeln!(file, "    id {}", node)?;
        writeln!(file, "  ]")?;
    }

    // Write edges
    let edges = graph.edges();
    for edge in edges {
        writeln!(file, "  edge [")?;
        writeln!(file, "    source {}", edge.source)?;
        writeln!(file, "    target {}", edge.target)?;
        if weighted {
            writeln!(file, "    weight {}", edge.weight)?;
        }
        writeln!(file, "  ]")?;
    }

    // Close graph
    writeln!(file, "]")?;

    Ok(())
}

/// Write a directed graph to GML format
///
/// # Arguments
///
/// * `graph` - The directed graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights as attributes
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_gml_format_digraph<N, E, Ix, P>(
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

    // Write GML header
    writeln!(file, "# Generated by scirs2-graph (directed)")?;
    writeln!(file, "graph [")?;
    writeln!(file, "  directed 1")?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, "  node [")?;
        writeln!(file, "    id {}", node)?;
        writeln!(file, "  ]")?;
    }

    // Write edges
    let edges = graph.edges();
    for edge in edges {
        writeln!(file, "  edge [")?;
        writeln!(file, "    source {}", edge.source)?;
        writeln!(file, "    target {}", edge.target)?;
        if weighted {
            writeln!(file, "    weight {}", edge.weight)?;
        }
        writeln!(file, "  ]")?;
    }

    // Close graph
    writeln!(file, "]")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_simple_gml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "graph [").unwrap();
        writeln!(temp_file, "  directed 0").unwrap();
        writeln!(temp_file, "  node [ id 1 ]").unwrap();
        writeln!(temp_file, "  node [ id 2 ]").unwrap();
        writeln!(temp_file, "  node [ id 3 ]").unwrap();
        writeln!(temp_file, "  edge [ source 1 target 2 ]").unwrap();
        writeln!(temp_file, "  edge [ source 2 target 3 ]").unwrap();
        writeln!(temp_file, "]").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_gml_format(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_read_weighted_gml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "graph [").unwrap();
        writeln!(temp_file, "  directed 0").unwrap();
        writeln!(temp_file, "  node [ id 1 ]").unwrap();
        writeln!(temp_file, "  node [ id 2 ]").unwrap();
        writeln!(temp_file, "  edge [ source 1 target 2 weight 1.5 ]").unwrap();
        writeln!(temp_file, "]").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_gml_format(temp_file.path(), true).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_read_directed_gml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "graph [").unwrap();
        writeln!(temp_file, "  directed 1").unwrap();
        writeln!(temp_file, "  node [ id 1 ]").unwrap();
        writeln!(temp_file, "  node [ id 2 ]").unwrap();
        writeln!(temp_file, "  edge [ source 1 target 2 ]").unwrap();
        writeln!(temp_file, "]").unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> = read_gml_format_digraph(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let mut original_graph: Graph<i32, f64> = Graph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_gml_format(&original_graph, temp_file.path(), false).unwrap();

        let read_graph: Graph<i32, f64> = read_gml_format(temp_file.path(), false).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_digraph_write_read_roundtrip() {
        let mut original_graph: DiGraph<i32, f64> = DiGraph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_gml_format_digraph(&original_graph, temp_file.path(), false).unwrap();

        let read_graph: DiGraph<i32, f64> =
            read_gml_format_digraph(temp_file.path(), false).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_invalid_gml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "invalid gml format").unwrap();
        temp_file.flush().unwrap();

        let result: Result<Graph<i32, f64>> = read_gml_format(temp_file.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_directed_mismatch() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "graph [").unwrap();
        writeln!(temp_file, "  directed 1").unwrap();
        writeln!(temp_file, "  node [ id 1 ]").unwrap();
        writeln!(temp_file, "  node [ id 2 ]").unwrap();
        writeln!(temp_file, "  edge [ source 1 target 2 ]").unwrap();
        writeln!(temp_file, "]").unwrap();
        temp_file.flush().unwrap();

        // Try to read as undirected graph - should fail
        let result: Result<Graph<i32, f64>> = read_gml_format(temp_file.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_gml_with_comments() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "# This is a comment").unwrap();
        writeln!(temp_file, "graph [").unwrap();
        writeln!(temp_file, "  # Another comment").unwrap();
        writeln!(temp_file, "  directed 0").unwrap();
        writeln!(temp_file, "  node [ id 1 label \"Node 1\" ]").unwrap();
        writeln!(temp_file, "  node [ id 2 label \"Node 2\" ]").unwrap();
        writeln!(temp_file, "  edge [ source 1 target 2 ]").unwrap();
        writeln!(temp_file, "]").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_gml_format(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
    }
}
