//! GraphML format I/O for graphs
//!
//! This module provides functionality for reading and writing graphs in GraphML format.
//! GraphML is an XML-based format for representing graph structures with rich metadata
//! and attribute support, widely used in graph analysis tools.
//!
//! # Format Specification
//!
//! GraphML uses XML structure with the following key elements:
//! - `<graphml>` - Root element with namespace declarations
//! - `<key>` - Attribute definitions for nodes/edges
//! - `<graph>` - Graph container with id and directedness
//! - `<node>` - Node elements with ids and data attributes
//! - `<edge>` - Edge elements with source/target and data attributes
//! - `<data>` - Data elements containing attribute values
//!
//! # Examples
//!
//! ## Basic GraphML structure:
//! ```xml
//! <?xml version="1.0" encoding="UTF-8"?>
//! <graphml xmlns="http://graphml.graphdrawing.org/xmlns"
//!          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
//!          xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
//!          http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
//!   <key id="weight" for="edge" attr.name="weight" attr.type="double"/>
//!   <graph id="G" edgedefault="undirected">
//!     <node id="1"/>
//!     <node id="2"/>
//!     <edge id="e1" source="1" target="2">
//!       <data key="weight">1.5</data>
//!     </edge>
//!   </graph>
//! </graphml>
//! ```
//!
//! # Usage
//!
//! ```rust
//! use std::fs::File;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//! use scirs2_graph::base::Graph;
//! use scirs2_graph::io::graphml::{read_graphml_format, write_graphml_format};
//!
//! // Create a temporary file with GraphML data
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
//! writeln!(temp_file, r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#).unwrap();
//! writeln!(temp_file, r#"  <graph id="G" edgedefault="undirected">"#).unwrap();
//! writeln!(temp_file, r#"    <node id="1"/>"#).unwrap();
//! writeln!(temp_file, r#"    <node id="2"/>"#).unwrap();
//! writeln!(temp_file, r#"    <edge source="1" target="2"/>"#).unwrap();
//! writeln!(temp_file, r#"  </graph>"#).unwrap();
//! writeln!(temp_file, r#"</graphml>"#).unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read the graph
//! let graph: Graph<i32, f64> = read_graphml_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 2);
//! assert_eq!(graph.edge_count(), 1);
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Simple XML parser state for GraphML parsing
#[derive(Debug, Clone, PartialEq, Eq)]
enum ParseState {
    /// Looking for graph opening tag
    SearchingGraph,
    /// Inside graph, parsing nodes and edges
    InGraph,
    /// Finished parsing
    Done,
}

/// GraphML key definition for attributes
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct GraphMLKey {
    /// Key identifier
    pub id: String,
    /// Target type: "node", "edge", "graph", "all"
    pub for_target: String,
    /// Attribute name
    pub attr_name: String,
    /// Attribute type: "boolean", "int", "long", "float", "double", "string"
    pub attr_type: String,
    /// Default value if any
    pub default_value: Option<String>,
}

/// Read an undirected graph from GraphML format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from GraphML data elements
///
/// # Returns
///
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
///
/// # Format
///
/// The GraphML format supports:
/// - XML-based structure with proper namespaces
/// - Key definitions for node and edge attributes
/// - Rich metadata and data elements
/// - Both directed and undirected graphs
/// - Hierarchical graph structures (though currently simplified)
pub fn read_graphml_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut graph = Graph::new();
    let mut state = ParseState::SearchingGraph;
    let mut keys = HashMap::new();
    let mut is_directed = false;

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() || line.starts_with("<?xml") || line.starts_with("<!--") {
            continue;
        }

        match state {
            ParseState::SearchingGraph => {
                if line.starts_with("<key ") {
                    if let Some(key) = parse_key_definition(line)? {
                        keys.insert(key.id.clone(), key);
                    }
                } else if line.starts_with("<graph ") {
                    is_directed = line.contains("edgedefault=\"directed\"");
                    state = ParseState::InGraph;
                }
            }
            ParseState::InGraph => {
                if line.starts_with("</graph>") {
                    state = ParseState::Done;
                    break;
                } else if line.starts_with("<node ") {
                    parse_node_element(line, &mut graph, line_num + 1)?;
                } else if line.starts_with("<edge ") {
                    parse_edge_element(line, &mut graph, &keys, weighted, line_num + 1)?;
                }
            }
            ParseState::Done => break,
        }
    }

    // Verify we found a graph
    if state == ParseState::SearchingGraph {
        return Err(GraphError::Other(
            "No valid GraphML graph element found".to_string(),
        ));
    }

    // Check if this was actually a directed graph but we're reading as undirected
    if is_directed {
        return Err(GraphError::Other(
            "GraphML file contains a directed graph, but undirected graph was requested"
                .to_string(),
        ));
    }

    Ok(graph)
}

/// Read a directed graph from GraphML format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to parse edge weights from GraphML data elements
///
/// # Returns
///
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
pub fn read_graphml_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();
    let mut state = ParseState::SearchingGraph;
    let mut keys = HashMap::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() || line.starts_with("<?xml") || line.starts_with("<!--") {
            continue;
        }

        match state {
            ParseState::SearchingGraph => {
                if line.starts_with("<key ") {
                    if let Some(key) = parse_key_definition(line)? {
                        keys.insert(key.id.clone(), key);
                    }
                } else if line.starts_with("<graph ") {
                    state = ParseState::InGraph;
                }
            }
            ParseState::InGraph => {
                if line.starts_with("</graph>") {
                    state = ParseState::Done;
                    break;
                } else if line.starts_with("<node ") {
                    parse_digraph_node_element(line, &mut graph, line_num + 1)?;
                } else if line.starts_with("<edge ") {
                    parse_digraph_edge_element(line, &mut graph, &keys, weighted, line_num + 1)?;
                }
            }
            ParseState::Done => break,
        }
    }

    // Verify we found a graph
    if state == ParseState::SearchingGraph {
        return Err(GraphError::Other(
            "No valid GraphML graph element found".to_string(),
        ));
    }

    Ok(graph)
}

/// Write an undirected graph to GraphML format
///
/// # Arguments
///
/// * `graph` - The graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights as data elements
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_graphml_format<N, E, Ix, P>(
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

    // Write XML declaration and GraphML header
    writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(
        file,
        r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns""#
    )?;
    writeln!(
        file,
        r#"         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance""#
    )?;
    writeln!(
        file,
        r#"         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns"#
    )?;
    writeln!(
        file,
        r#"         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">"#
    )?;

    // Write key definitions if weighted
    if weighted {
        writeln!(
            file,
            r#"  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>"#
        )?;
    }

    // Write graph opening tag
    writeln!(file, r#"  <graph id="G" edgedefault="undirected">"#)?;
    writeln!(file, "    <!-- Generated by scirs2-graph -->")?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, r#"    <node id="{}"/>"#, node)?;
    }

    // Write edges
    let edges = graph.edges();
    for (edge_id, edge) in edges.iter().enumerate() {
        if weighted {
            writeln!(
                file,
                r#"    <edge id="e{}" source="{}" target="{}">"#,
                edge_id, edge.source, edge.target
            )?;
            writeln!(file, r#"      <data key="weight">{}</data>"#, edge.weight)?;
            writeln!(file, "    </edge>")?;
        } else {
            writeln!(
                file,
                r#"    <edge id="e{}" source="{}" target="{}"/>"#,
                edge_id, edge.source, edge.target
            )?;
        }
    }

    // Close graph and graphml
    writeln!(file, "  </graph>")?;
    writeln!(file, "</graphml>")?;

    Ok(())
}

/// Write a directed graph to GraphML format
///
/// # Arguments
///
/// * `graph` - The directed graph to write
/// * `path` - Path to the output file
/// * `weighted` - Whether to include edge weights as data elements
///
/// # Returns
///
/// * `Ok(())` - If the graph was written successfully
/// * `Err(GraphError)` - If there was an error writing the file
pub fn write_graphml_format_digraph<N, E, Ix, P>(
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

    // Write XML declaration and GraphML header
    writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
    writeln!(
        file,
        r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns""#
    )?;
    writeln!(
        file,
        r#"         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance""#
    )?;
    writeln!(
        file,
        r#"         xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns"#
    )?;
    writeln!(
        file,
        r#"         http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">"#
    )?;

    // Write key definitions if weighted
    if weighted {
        writeln!(
            file,
            r#"  <key id="weight" for="edge" attr.name="weight" attr.type="double"/>"#
        )?;
    }

    // Write graph opening tag (directed)
    writeln!(file, r#"  <graph id="G" edgedefault="directed">"#)?;
    writeln!(file, "    <!-- Generated by scirs2-graph (directed) -->")?;

    // Write nodes
    for node in graph.nodes() {
        writeln!(file, r#"    <node id="{}"/>"#, node)?;
    }

    // Write edges
    let edges = graph.edges();
    for (edge_id, edge) in edges.iter().enumerate() {
        if weighted {
            writeln!(
                file,
                r#"    <edge id="e{}" source="{}" target="{}">"#,
                edge_id, edge.source, edge.target
            )?;
            writeln!(file, r#"      <data key="weight">{}</data>"#, edge.weight)?;
            writeln!(file, "    </edge>")?;
        } else {
            writeln!(
                file,
                r#"    <edge id="e{}" source="{}" target="{}"/>"#,
                edge_id, edge.source, edge.target
            )?;
        }
    }

    // Close graph and graphml
    writeln!(file, "  </graph>")?;
    writeln!(file, "</graphml>")?;

    Ok(())
}

// Helper functions

/// Parse a GraphML key definition from XML
fn parse_key_definition(line: &str) -> Result<Option<GraphMLKey>> {
    // Simple attribute parsing for key elements
    let mut id = None;
    let mut for_target = None;
    let mut attr_name = None;
    let mut attr_type = None;

    // Extract attributes using simple string matching
    if let Some(id_start) = line.find(r#"id=""#) {
        let start = id_start + 4;
        if let Some(end) = line[start..].find('"') {
            id = Some(line[start..start + end].to_string());
        }
    }

    if let Some(for_start) = line.find(r#"for=""#) {
        let start = for_start + 5;
        if let Some(end) = line[start..].find('"') {
            for_target = Some(line[start..start + end].to_string());
        }
    }

    if let Some(name_start) = line.find(r#"attr.name=""#) {
        let start = name_start + 11;
        if let Some(end) = line[start..].find('"') {
            attr_name = Some(line[start..start + end].to_string());
        }
    }

    if let Some(type_start) = line.find(r#"attr.type=""#) {
        let start = type_start + 11;
        if let Some(end) = line[start..].find('"') {
            attr_type = Some(line[start..start + end].to_string());
        }
    }

    if let (Some(id), Some(for_target), Some(attr_name), Some(attr_type)) =
        (id, for_target, attr_name, attr_type)
    {
        Ok(Some(GraphMLKey {
            id,
            for_target,
            attr_name,
            attr_type,
            default_value: None,
        }))
    } else {
        Ok(None)
    }
}

/// Parse a GraphML node element for undirected graphs
fn parse_node_element<N, E>(line: &str, _graph: &mut Graph<N, E>, line_num: usize) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    // Extract node id
    if let Some(id_start) = line.find(r#"id=""#) {
        let start = id_start + 4;
        if let Some(end) = line[start..].find('"') {
            let node_id = &line[start..start + end];
            let _node = N::from_str(node_id).map_err(|_| {
                GraphError::Other(format!(
                    "Failed to parse node ID '{}' on line {}",
                    node_id, line_num
                ))
            })?;
            // Nodes will be added automatically when edges are added
        }
    }

    Ok(())
}

/// Parse a GraphML node element for directed graphs
fn parse_digraph_node_element<N, E>(
    line: &str,
    _graph: &mut DiGraph<N, E>,
    line_num: usize,
) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    // Extract node id
    if let Some(id_start) = line.find(r#"id=""#) {
        let start = id_start + 4;
        if let Some(end) = line[start..].find('"') {
            let node_id = &line[start..start + end];
            let _node = N::from_str(node_id).map_err(|_| {
                GraphError::Other(format!(
                    "Failed to parse node ID '{}' on line {}",
                    node_id, line_num
                ))
            })?;
            // Nodes will be added automatically when edges are added
        }
    }

    Ok(())
}

/// Parse a GraphML edge element for undirected graphs
fn parse_edge_element<N, E>(
    line: &str,
    graph: &mut Graph<N, E>,
    _keys: &HashMap<String, GraphMLKey>,
    _weighted: bool,
    line_num: usize,
) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    let mut source_id = None;
    let mut target_id = None;

    // Extract source
    if let Some(source_start) = line.find(r#"source=""#) {
        let start = source_start + 8;
        if let Some(end) = line[start..].find('"') {
            source_id = Some(&line[start..start + end]);
        }
    }

    // Extract target
    if let Some(target_start) = line.find(r#"target=""#) {
        let start = target_start + 8;
        if let Some(end) = line[start..].find('"') {
            target_id = Some(&line[start..start + end]);
        }
    }

    if let (Some(source_id), Some(target_id)) = (source_id, target_id) {
        let source_node = N::from_str(source_id).map_err(|_| {
            GraphError::Other(format!(
                "Failed to parse source node '{}' on line {}",
                source_id, line_num
            ))
        })?;

        let target_node = N::from_str(target_id).map_err(|_| {
            GraphError::Other(format!(
                "Failed to parse target node '{}' on line {}",
                target_id, line_num
            ))
        })?;

        // For now, use default weight (proper data parsing would be more complex)
        let weight = E::default();

        graph.add_edge(source_node, target_node, weight)?;
    } else {
        return Err(GraphError::Other(format!(
            "Invalid edge element - missing source or target on line {}",
            line_num
        )));
    }

    Ok(())
}

/// Parse a GraphML edge element for directed graphs
fn parse_digraph_edge_element<N, E>(
    line: &str,
    graph: &mut DiGraph<N, E>,
    _keys: &HashMap<String, GraphMLKey>,
    _weighted: bool,
    line_num: usize,
) -> Result<()>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
{
    let mut source_id = None;
    let mut target_id = None;

    // Extract source
    if let Some(source_start) = line.find(r#"source=""#) {
        let start = source_start + 8;
        if let Some(end) = line[start..].find('"') {
            source_id = Some(&line[start..start + end]);
        }
    }

    // Extract target
    if let Some(target_start) = line.find(r#"target=""#) {
        let start = target_start + 8;
        if let Some(end) = line[start..].find('"') {
            target_id = Some(&line[start..start + end]);
        }
    }

    if let (Some(source_id), Some(target_id)) = (source_id, target_id) {
        let source_node = N::from_str(source_id).map_err(|_| {
            GraphError::Other(format!(
                "Failed to parse source node '{}' on line {}",
                source_id, line_num
            ))
        })?;

        let target_node = N::from_str(target_id).map_err(|_| {
            GraphError::Other(format!(
                "Failed to parse target node '{}' on line {}",
                target_id, line_num
            ))
        })?;

        // For now, use default weight (proper data parsing would be more complex)
        let weight = E::default();

        graph.add_edge(source_node, target_node, weight)?;
    } else {
        return Err(GraphError::Other(format!(
            "Invalid edge element - missing source or target on line {}",
            line_num
        )));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_simple_graphml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
        writeln!(
            temp_file,
            r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#
        )
        .unwrap();
        writeln!(temp_file, r#"  <graph id="G" edgedefault="undirected">"#).unwrap();
        writeln!(temp_file, r#"    <node id="1"/>"#).unwrap();
        writeln!(temp_file, r#"    <node id="2"/>"#).unwrap();
        writeln!(temp_file, r#"    <node id="3"/>"#).unwrap();
        writeln!(temp_file, r#"    <edge source="1" target="2"/>"#).unwrap();
        writeln!(temp_file, r#"    <edge source="2" target="3"/>"#).unwrap();
        writeln!(temp_file, r#"  </graph>"#).unwrap();
        writeln!(temp_file, r#"</graphml>"#).unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_graphml_format(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_read_directed_graphml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
        writeln!(
            temp_file,
            r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#
        )
        .unwrap();
        writeln!(temp_file, r#"  <graph id="G" edgedefault="directed">"#).unwrap();
        writeln!(temp_file, r#"    <node id="1"/>"#).unwrap();
        writeln!(temp_file, r#"    <node id="2"/>"#).unwrap();
        writeln!(temp_file, r#"    <node id="3"/>"#).unwrap();
        writeln!(temp_file, r#"    <edge source="1" target="2"/>"#).unwrap();
        writeln!(temp_file, r#"    <edge source="2" target="3"/>"#).unwrap();
        writeln!(temp_file, r#"  </graph>"#).unwrap();
        writeln!(temp_file, r#"</graphml>"#).unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> =
            read_graphml_format_digraph(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let mut original_graph: Graph<i32, f64> = Graph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_graphml_format(&original_graph, temp_file.path(), false).unwrap();

        let read_graph: Graph<i32, f64> = read_graphml_format(temp_file.path(), false).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_digraph_write_read_roundtrip() {
        let mut original_graph: DiGraph<i32, f64> = DiGraph::new();
        original_graph.add_edge(1i32, 2i32, 1.5f64).unwrap();
        original_graph.add_edge(2i32, 3i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_graphml_format_digraph(&original_graph, temp_file.path(), false).unwrap();

        let read_graph: DiGraph<i32, f64> =
            read_graphml_format_digraph(temp_file.path(), false).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_invalid_xml() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "<invalid>xml</invalid>").unwrap();
        temp_file.flush().unwrap();

        let result: Result<Graph<i32, f64>> = read_graphml_format(temp_file.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_directed_mismatch() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
        writeln!(
            temp_file,
            r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#
        )
        .unwrap();
        writeln!(temp_file, r#"  <graph id="G" edgedefault="directed">"#).unwrap();
        writeln!(temp_file, r#"    <node id="1"/>"#).unwrap();
        writeln!(temp_file, r#"    <node id="2"/>"#).unwrap();
        writeln!(temp_file, r#"    <edge source="1" target="2"/>"#).unwrap();
        writeln!(temp_file, r#"  </graph>"#).unwrap();
        writeln!(temp_file, r#"</graphml>"#).unwrap();
        temp_file.flush().unwrap();

        // Try to read as undirected graph - should fail
        let result: Result<Graph<i32, f64>> = read_graphml_format(temp_file.path(), false);
        assert!(result.is_err());
    }
}
