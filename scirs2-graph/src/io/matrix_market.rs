//! Matrix Market format I/O for graphs
//!
//! This module provides functionality for reading and writing graphs in Matrix Market format.
//! Matrix Market is a standard format for storing sparse matrices, commonly used in scientific computing.
//!
//! # Format Specification
//!
//! The Matrix Market format consists of:
//! 1. Header line: `%%MatrixMarket matrix coordinate [pattern|real|integer] [general|symmetric|skew-symmetric|hermitian]`
//! 2. Optional comment lines starting with `%`
//! 3. Size line: `rows cols nnz` (number of rows, columns, non-zeros)
//! 4. Data lines: `row col [value]` (1-indexed coordinates, optional value)
//!
//! # Examples
//!
//! ## Pattern matrix (unweighted graph):
//! ```text
//! %%MatrixMarket matrix coordinate pattern general
//! % This is a comment
//! 3 3 4
//! 1 2
//! 2 3
//! 3 1
//! 1 3
//! ```
//!
//! ## Real matrix (weighted graph):
//! ```text
//! %%MatrixMarket matrix coordinate real general
//! 3 3 4
//! 1 2 1.5
//! 2 3 2.0
//! 3 1 0.5
//! 1 3 1.0
//! ```
//!
//! # Usage
//!
//! ```rust
//! use std::fs::File;
//! use std::io::Write;
//! use tempfile::NamedTempFile;
//! use scirs2_graph::base::Graph;
//! use scirs2_graph::io::matrix_market::{read_matrix_market_format, write_matrix_market_format};
//!
//! // Create a temporary file with Matrix Market data
//! let mut temp_file = NamedTempFile::new().unwrap();
//! writeln!(temp_file, "%%MatrixMarket matrix coordinate pattern general").unwrap();
//! writeln!(temp_file, "3 3 3").unwrap();
//! writeln!(temp_file, "1 2").unwrap();
//! writeln!(temp_file, "2 3").unwrap();
//! writeln!(temp_file, "3 1").unwrap();
//! temp_file.flush().unwrap();
//!
//! // Read the graph
//! let graph: Graph<i32, f64> = read_matrix_market_format(temp_file.path(), false).unwrap();
//! assert_eq!(graph.node_count(), 3);
//! assert_eq!(graph.edge_count(), 3);
//! ```

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::str::FromStr;

use crate::base::{DiGraph, EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

/// Matrix Market format specification
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixMarketHeader {
    /// Object type (always "matrix" for our purposes)
    pub object: String,
    /// Format type (always "coordinate" for sparse matrices)
    pub format: String,
    /// Field type: "pattern", "real", "integer", "complex"
    pub field: String,
    /// Symmetry type: "general", "symmetric", "skew-symmetric", "hermitian"
    pub symmetry: String,
    /// Number of rows
    pub rows: usize,
    /// Number of columns
    pub cols: usize,
    /// Number of non-zero entries
    pub nnz: usize,
}

impl MatrixMarketHeader {
    /// Parse a Matrix Market header from a string
    pub fn parse_header_line(line: &str) -> Result<(String, String, String, String)> {
        if !line.starts_with("%%MatrixMarket") {
            return Err(GraphError::Other(
                "Invalid Matrix Market header - must start with %%MatrixMarket".to_string(),
            ));
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 5 {
            return Err(GraphError::Other(
                "Invalid Matrix Market header - expected 5 parts".to_string(),
            ));
        }

        Ok((
            parts[1].to_lowercase(), // object
            parts[2].to_lowercase(), // format
            parts[3].to_lowercase(), // field
            parts[4].to_lowercase(), // symmetry
        ))
    }

    /// Parse size line (rows cols nnz)
    pub fn parse_size_line(line: &str) -> Result<(usize, usize, usize)> {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            return Err(GraphError::Other(
                "Invalid Matrix Market size line - expected 3 numbers".to_string(),
            ));
        }

        let rows = parts[0].parse::<usize>().map_err(|_| {
            GraphError::Other(format!("Failed to parse number of rows: {}", parts[0]))
        })?;
        let cols = parts[1].parse::<usize>().map_err(|_| {
            GraphError::Other(format!("Failed to parse number of columns: {}", parts[1]))
        })?;
        let nnz = parts[2].parse::<usize>().map_err(|_| {
            GraphError::Other(format!("Failed to parse number of non-zeros: {}", parts[2]))
        })?;

        Ok((rows, cols, nnz))
    }

    /// Check if the matrix format is supported
    pub fn validate(&self) -> Result<()> {
        if self.object != "matrix" {
            return Err(GraphError::Other(format!(
                "Unsupported object type: {}",
                self.object
            )));
        }

        if self.format != "coordinate" {
            return Err(GraphError::Other(format!(
                "Unsupported format type: {}",
                self.format
            )));
        }

        if !matches!(
            self.field.as_str(),
            "pattern" | "real" | "integer" | "complex"
        ) {
            return Err(GraphError::Other(format!(
                "Unsupported field type: {}",
                self.field
            )));
        }

        if !matches!(
            self.symmetry.as_str(),
            "general" | "symmetric" | "skew-symmetric" | "hermitian"
        ) {
            return Err(GraphError::Other(format!(
                "Unsupported symmetry type: {}",
                self.symmetry
            )));
        }

        Ok(())
    }

    /// Check if the matrix has values (not just pattern)
    pub fn has_values(&self) -> bool {
        matches!(self.field.as_str(), "real" | "integer" | "complex")
    }

    /// Check if the matrix is symmetric
    pub fn is_symmetric(&self) -> bool {
        matches!(self.symmetry.as_str(), "symmetric" | "hermitian")
    }
}

/// Read an undirected graph from Matrix Market format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to read edge weights (ignored for pattern matrices)
///
/// # Returns
///
/// * `Ok(Graph)` - The graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
///
/// # Format
///
/// The Matrix Market format supports both pattern (unweighted) and valued (weighted) matrices.
/// For pattern matrices, edges have default weights.
/// For valued matrices, the third column contains edge weights.
pub fn read_matrix_market_format<N, E, P>(path: P, weighted: bool) -> Result<Graph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut graph = Graph::new();

    // Read and parse header
    let header_line = lines
        .next()
        .ok_or_else(|| GraphError::Other("Empty file".to_string()))?
        .map_err(|e| GraphError::Other(format!("Error reading header line: {}", e)))?;

    let (object, format, field, symmetry) = MatrixMarketHeader::parse_header_line(&header_line)?;

    // Create header struct
    let mut header = MatrixMarketHeader {
        object,
        format,
        field,
        symmetry,
        rows: 0,
        cols: 0,
        nnz: 0,
    };

    // Validate header
    header.validate()?;

    // Skip comment lines
    let mut size_line = String::new();
    for line_result in lines.by_ref() {
        let line = line_result?;
        if !line.trim().starts_with('%') && !line.trim().is_empty() {
            size_line = line;
            break;
        }
    }

    if size_line.is_empty() {
        return Err(GraphError::Other("No size line found".to_string()));
    }

    // Parse size line
    let (rows, cols, nnz) = MatrixMarketHeader::parse_size_line(&size_line)?;
    header.rows = rows;
    header.cols = cols;
    header.nnz = nnz;

    // Read data entries
    let mut entries_read = 0;
    for line_result in lines {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        // Parse row and column (1-indexed in Matrix Market format)
        if parts.len() < 2 {
            return Err(GraphError::Other(format!(
                "Invalid data line - expected at least 2 columns: {}",
                line
            )));
        }

        let row: usize = parts[0]
            .parse()
            .map_err(|_| GraphError::Other(format!("Failed to parse row index: {}", parts[0])))?;
        let col: usize = parts[1].parse().map_err(|_| {
            GraphError::Other(format!("Failed to parse column index: {}", parts[1]))
        })?;

        // Convert to 0-indexed and create nodes
        let source_node = N::from_str(&(row - 1).to_string()).map_err(|_| {
            GraphError::Other(format!(
                "Failed to create source node from index: {}",
                row - 1
            ))
        })?;
        let target_node = N::from_str(&(col - 1).to_string()).map_err(|_| {
            GraphError::Other(format!(
                "Failed to create target node from index: {}",
                col - 1
            ))
        })?;

        // Parse weight if available and requested
        let weight = if header.has_values() && weighted && parts.len() > 2 {
            E::from_str(parts[2])
                .map_err(|_| GraphError::Other(format!("Failed to parse weight: {}", parts[2])))?
        } else {
            E::default()
        };

        // Add edge(s) - handle symmetry
        if !graph.has_edge(&source_node, &target_node) {
            graph.add_edge(source_node.clone(), target_node.clone(), weight)?;
        }

        // Add symmetric edge if the matrix is symmetric and it's not a diagonal entry
        if header.is_symmetric() && row != col {
            graph.add_edge(target_node, source_node, weight)?;
        }

        entries_read += 1;
    }

    // Verify we read the expected number of entries
    if entries_read != nnz {
        return Err(GraphError::Other(format!(
            "Expected {} entries, but read {}",
            nnz, entries_read
        )));
    }

    Ok(graph)
}

/// Read a directed graph from Matrix Market format
///
/// # Arguments
///
/// * `path` - Path to the input file
/// * `weighted` - Whether to read edge weights (ignored for pattern matrices)
///
/// # Returns
///
/// * `Ok(DiGraph)` - The directed graph read from the file
/// * `Err(GraphError)` - If there was an error reading or parsing the file
pub fn read_matrix_market_format_digraph<N, E, P>(path: P, weighted: bool) -> Result<DiGraph<N, E>>
where
    N: Node + std::fmt::Debug + FromStr + Clone,
    E: EdgeWeight + std::marker::Copy + std::fmt::Debug + std::default::Default + FromStr,
    P: AsRef<Path>,
{
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let mut graph = DiGraph::new();

    // Read and parse header
    let header_line = lines
        .next()
        .ok_or_else(|| GraphError::Other("Empty file".to_string()))?
        .map_err(|e| GraphError::Other(format!("Error reading header line: {}", e)))?;

    let (object, format, field, symmetry) = MatrixMarketHeader::parse_header_line(&header_line)?;

    // Create header struct
    let mut header = MatrixMarketHeader {
        object,
        format,
        field,
        symmetry,
        rows: 0,
        cols: 0,
        nnz: 0,
    };

    // Validate header
    header.validate()?;

    // Skip comment lines
    let mut size_line = String::new();
    for line_result in lines.by_ref() {
        let line = line_result?;
        if !line.trim().starts_with('%') && !line.trim().is_empty() {
            size_line = line;
            break;
        }
    }

    if size_line.is_empty() {
        return Err(GraphError::Other("No size line found".to_string()));
    }

    // Parse size line
    let (rows, cols, nnz) = MatrixMarketHeader::parse_size_line(&size_line)?;
    header.rows = rows;
    header.cols = cols;
    header.nnz = nnz;

    // Read data entries
    let mut entries_read = 0;
    for line_result in lines {
        let line = line_result?;
        let line = line.trim();

        if line.is_empty() || line.starts_with('%') {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();

        // Parse row and column (1-indexed in Matrix Market format)
        if parts.len() < 2 {
            return Err(GraphError::Other(format!(
                "Invalid data line - expected at least 2 columns: {}",
                line
            )));
        }

        let row: usize = parts[0]
            .parse()
            .map_err(|_| GraphError::Other(format!("Failed to parse row index: {}", parts[0])))?;
        let col: usize = parts[1].parse().map_err(|_| {
            GraphError::Other(format!("Failed to parse column index: {}", parts[1]))
        })?;

        // Convert to 0-indexed and create nodes
        let source_node = N::from_str(&(row - 1).to_string()).map_err(|_| {
            GraphError::Other(format!(
                "Failed to create source node from index: {}",
                row - 1
            ))
        })?;
        let target_node = N::from_str(&(col - 1).to_string()).map_err(|_| {
            GraphError::Other(format!(
                "Failed to create target node from index: {}",
                col - 1
            ))
        })?;

        // Parse weight if available and requested
        let weight = if header.has_values() && weighted && parts.len() > 2 {
            E::from_str(parts[2])
                .map_err(|_| GraphError::Other(format!("Failed to parse weight: {}", parts[2])))?
        } else {
            E::default()
        };

        // Add directed edge
        graph.add_edge(source_node, target_node, weight)?;

        entries_read += 1;
    }

    // Verify we read the expected number of entries
    if entries_read != nnz {
        return Err(GraphError::Other(format!(
            "Expected {} entries, but read {}",
            nnz, entries_read
        )));
    }

    Ok(graph)
}

/// Write an undirected graph to Matrix Market format
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
pub fn write_matrix_market_format<N, E, Ix, P>(
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

    // Write header
    let field_type = if weighted { "real" } else { "pattern" };
    writeln!(
        file,
        "%%MatrixMarket matrix coordinate {} general",
        field_type
    )?;

    // Write comment
    writeln!(file, "% Generated by scirs2-graph")?;

    // Collect all edges
    let edges = graph.edges();
    let nodes = graph.nodes();
    let node_count = nodes.len();
    let edge_count = edges.len();

    // Write size line
    writeln!(file, "{} {} {}", node_count, node_count, edge_count)?;

    // Create node index mapping
    let mut node_to_index = std::collections::HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        node_to_index.insert((*node).clone(), idx + 1); // 1-indexed
    }

    // Write edges
    for edge in edges {
        let source_idx = node_to_index.get(&edge.source).ok_or_else(|| {
            GraphError::Other(format!("Source node not found: {:?}", edge.source))
        })?;
        let target_idx = node_to_index.get(&edge.target).ok_or_else(|| {
            GraphError::Other(format!("Target node not found: {:?}", edge.target))
        })?;

        if weighted {
            writeln!(file, "{} {} {}", source_idx, target_idx, edge.weight)?;
        } else {
            writeln!(file, "{} {}", source_idx, target_idx)?;
        }
    }

    Ok(())
}

/// Write a directed graph to Matrix Market format
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
pub fn write_matrix_market_format_digraph<N, E, Ix, P>(
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

    // Write header
    let field_type = if weighted { "real" } else { "pattern" };
    writeln!(
        file,
        "%%MatrixMarket matrix coordinate {} general",
        field_type
    )?;

    // Write comment
    writeln!(file, "% Generated by scires2-graph (directed)")?;

    // Collect all edges
    let edges = graph.edges();
    let nodes = graph.nodes();
    let node_count = nodes.len();
    let edge_count = edges.len();

    // Write size line
    writeln!(file, "{} {} {}", node_count, node_count, edge_count)?;

    // Create node index mapping
    let mut node_to_index = std::collections::HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        node_to_index.insert((*node).clone(), idx + 1); // 1-indexed
    }

    // Write edges
    for edge in edges {
        let source_idx = node_to_index.get(&edge.source).ok_or_else(|| {
            GraphError::Other(format!("Source node not found: {:?}", edge.source))
        })?;
        let target_idx = node_to_index.get(&edge.target).ok_or_else(|| {
            GraphError::Other(format!("Target node not found: {:?}", edge.target))
        })?;

        if weighted {
            writeln!(file, "{} {} {}", source_idx, target_idx, edge.weight)?;
        } else {
            writeln!(file, "{} {}", source_idx, target_idx)?;
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
    fn test_parse_header_line() {
        let header = "%%MatrixMarket matrix coordinate real general";
        let (object, format, field, symmetry) =
            MatrixMarketHeader::parse_header_line(header).unwrap();

        assert_eq!(object, "matrix");
        assert_eq!(format, "coordinate");
        assert_eq!(field, "real");
        assert_eq!(symmetry, "general");
    }

    #[test]
    fn test_parse_size_line() {
        let size_line = "10 10 20";
        let (rows, cols, nnz) = MatrixMarketHeader::parse_size_line(size_line).unwrap();

        assert_eq!(rows, 10);
        assert_eq!(cols, 10);
        assert_eq!(nnz, 20);
    }

    #[test]
    fn test_read_pattern_matrix_market() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(
            temp_file,
            "%%MatrixMarket matrix coordinate pattern general"
        )
        .unwrap();
        writeln!(temp_file, "% Test pattern matrix").unwrap();
        writeln!(temp_file, "3 3 3").unwrap();
        writeln!(temp_file, "1 2").unwrap();
        writeln!(temp_file, "2 3").unwrap();
        writeln!(temp_file, "3 1").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_matrix_market_format(temp_file.path(), false).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_read_real_matrix_market() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "%%MatrixMarket matrix coordinate real general").unwrap();
        writeln!(temp_file, "3 3 3").unwrap();
        writeln!(temp_file, "1 2 1.5").unwrap();
        writeln!(temp_file, "2 3 2.0").unwrap();
        writeln!(temp_file, "3 1 0.5").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_matrix_market_format(temp_file.path(), true).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_read_symmetric_matrix_market() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "%%MatrixMarket matrix coordinate real symmetric").unwrap();
        writeln!(temp_file, "3 3 2").unwrap();
        writeln!(temp_file, "1 2 1.5").unwrap();
        writeln!(temp_file, "2 3 2.0").unwrap();
        temp_file.flush().unwrap();

        let graph: Graph<i32, f64> = read_matrix_market_format(temp_file.path(), true).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 4); // 2 original + 2 symmetric
    }

    #[test]
    fn test_read_digraph_matrix_market() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "%%MatrixMarket matrix coordinate real general").unwrap();
        writeln!(temp_file, "3 3 3").unwrap();
        writeln!(temp_file, "1 2 1.5").unwrap();
        writeln!(temp_file, "2 3 2.0").unwrap();
        writeln!(temp_file, "3 1 0.5").unwrap();
        temp_file.flush().unwrap();

        let graph: DiGraph<i32, f64> =
            read_matrix_market_format_digraph(temp_file.path(), true).unwrap();

        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_write_read_roundtrip() {
        let mut original_graph: Graph<i32, f64> = Graph::new();
        original_graph.add_edge(0i32, 1i32, 1.5f64).unwrap();
        original_graph.add_edge(1i32, 2i32, 2.0f64).unwrap();

        let temp_file = NamedTempFile::new().unwrap();
        write_matrix_market_format(&original_graph, temp_file.path(), true).unwrap();

        let read_graph: Graph<i32, f64> =
            read_matrix_market_format(temp_file.path(), true).unwrap();

        assert_eq!(read_graph.node_count(), original_graph.node_count());
        assert_eq!(read_graph.edge_count(), original_graph.edge_count());
    }

    #[test]
    fn test_invalid_header() {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "%%InvalidHeader").unwrap();
        temp_file.flush().unwrap();

        let result: Result<Graph<i32, f64>> = read_matrix_market_format(temp_file.path(), false);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_file() {
        let temp_file = NamedTempFile::new().unwrap();

        let result: Result<Graph<i32, f64>> = read_matrix_market_format(temp_file.path(), false);
        assert!(result.is_err());
    }
}
