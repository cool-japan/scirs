//! Input/output operations for graphs
//!
//! This module provides functions for reading and writing graph data
//! in various formats.

use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::Result;

pub mod adjacency_list;
pub mod dot;
pub mod edge_list;
pub mod gml;
pub mod graphml;
pub mod json;
pub mod matrix_market;

use adjacency_list::{
    read_adjacency_list_format, read_adjacency_list_format_digraph, write_adjacency_list_format,
    write_adjacency_list_format_digraph,
};
use dot::{read_dot_format, read_dot_format_digraph, write_dot_format, write_dot_format_digraph};
use edge_list::{
    read_edge_list_format, read_edge_list_format_digraph, write_edge_list_format,
    write_edge_list_format_digraph,
};
use gml::{read_gml_format, read_gml_format_digraph, write_gml_format, write_gml_format_digraph};
use graphml::{
    read_graphml_format, read_graphml_format_digraph, write_graphml_format,
    write_graphml_format_digraph,
};
use json::{
    read_json_format, read_json_format_digraph, write_json_format, write_json_format_digraph,
};
use matrix_market::{
    read_matrix_market_format, read_matrix_market_format_digraph, write_matrix_market_format,
    write_matrix_market_format_digraph,
};

/// Supported file formats for graph I/O
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphFormat {
    /// Edge list format (one edge per line: source target [weight])
    EdgeList,
    /// Adjacency list format (source: target1 target2 ...)
    AdjacencyList,
    /// Matrix Market format (sparse matrix format)
    MatrixMarket,
    /// GraphML format (XML-based format for graphs)
    GraphML,
    /// GML format (Graph Modeling Language)
    Gml,
    /// DOT format (Graphviz format)
    Dot,
    /// JSON graph format
    Json,
}

/// Reads a graph from a file
///
/// # Arguments
/// * `path` - Path to the file
/// * `format` - Format of the file
/// * `weighted` - Whether the graph has edge weights
/// * `directed` - Whether the graph is directed
///
/// # Returns
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading the file
pub fn read_graph<N, E, P>(path: P, format: GraphFormat, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    match format {
        GraphFormat::EdgeList => read_edge_list_format(path, weighted),
        GraphFormat::AdjacencyList => read_adjacency_list_format(path, weighted),
        GraphFormat::Dot => read_dot_format(path, weighted),
        GraphFormat::Json => read_json_format(path, weighted),
        GraphFormat::GraphML => read_graphml_format(path, weighted),
        GraphFormat::Gml => read_gml_format(path, weighted),
        GraphFormat::MatrixMarket => read_matrix_market_format(path, weighted),
    }
}

/// Reads a directed graph from a file
///
/// # Arguments
/// * `path` - Path to the file
/// * `format` - Format of the file
/// * `weighted` - Whether the graph has edge weights
///
/// # Returns
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading the file
pub fn read_digraph<N, E, P>(path: P, format: GraphFormat, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    match format {
        GraphFormat::EdgeList => read_edge_list_format_digraph(path, weighted),
        GraphFormat::AdjacencyList => read_adjacency_list_format_digraph(path, weighted),
        GraphFormat::Dot => read_dot_format_digraph(path, weighted),
        GraphFormat::Json => read_json_format_digraph(path, weighted),
        GraphFormat::GraphML => read_graphml_format_digraph(path, weighted),
        GraphFormat::Gml => read_gml_format_digraph(path, weighted),
        GraphFormat::MatrixMarket => read_matrix_market_format_digraph(path, weighted),
    }
}

/// Writes a graph to a file
///
/// # Arguments
/// * `graph` - The graph to write
/// * `path` - Path to the output file
/// * `format` - Format to write the file in
/// * `weighted` - Whether to include edge weights
///
/// # Returns
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_graph<N, E, Ix, P>(
    graph: &Graph<N, E, Ix>,
    path: P,
    format: GraphFormat,
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
    match format {
        GraphFormat::EdgeList => write_edge_list_format(graph, path, weighted),
        GraphFormat::AdjacencyList => write_adjacency_list_format(graph, path, weighted),
        GraphFormat::Dot => write_dot_format(graph, path, weighted),
        GraphFormat::Json => write_json_format(graph, path, weighted),
        GraphFormat::GraphML => write_graphml_format(graph, path, weighted),
        GraphFormat::Gml => write_gml_format(graph, path, weighted),
        GraphFormat::MatrixMarket => write_matrix_market_format(graph, path, weighted),
    }
}

/// Writes a directed graph to a file
///
/// # Arguments
/// * `graph` - The directed graph to write
/// * `path` - Path to the output file
/// * `format` - Format to write the file in
/// * `weighted` - Whether to include edge weights
///
/// # Returns
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_digraph<N, E, Ix, P>(
    graph: &DiGraph<N, E, Ix>,
    path: P,
    format: GraphFormat,
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
    match format {
        GraphFormat::EdgeList => write_edge_list_format_digraph(graph, path, weighted),
        GraphFormat::AdjacencyList => write_adjacency_list_format_digraph(graph, path, weighted),
        GraphFormat::Dot => write_dot_format_digraph(graph, path, weighted),
        GraphFormat::Json => write_json_format_digraph(graph, path, weighted),
        GraphFormat::GraphML => write_graphml_format_digraph(graph, path, weighted),
        GraphFormat::Gml => write_gml_format_digraph(graph, path, weighted),
        GraphFormat::MatrixMarket => write_matrix_market_format_digraph(graph, path, weighted),
    }
}

// All I/O formats are now implemented in their respective modules:
// - edge_list: Simple edge list format
// - adjacency_list: Adjacency list format
// - matrix_market: Matrix Market sparse format
// - dot: Graphviz DOT format
// - json: JSON graph format
// - graphml: GraphML XML format
// - gml: Graph Modeling Language format
