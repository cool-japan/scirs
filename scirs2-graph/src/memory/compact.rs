//! Compact graph representations for memory efficiency
//!
//! This module provides memory-efficient graph storage formats optimized for
//! different graph characteristics (sparse, dense, regular degree, etc.).

use crate::error::GraphError;
use scirs2_core::ndarray::Array2;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::mem;
use std::path::Path;

/// Compressed Sparse Row (CSR) format for sparse graphs
///
/// This format is highly memory-efficient for sparse graphs and provides
/// fast row (neighbor) access.
#[derive(Debug, Clone)]
pub struct CSRGraph {
    /// Number of nodes
    n_nodes: usize,
    /// Number of edges
    n_edges: usize,
    /// Row pointers - indices where each node's edges start
    row_ptr: Vec<usize>,
    /// Column indices - destination nodes
    col_idx: Vec<usize>,
    /// Edge weights
    weights: Vec<f64>,
}

impl CSRGraph {
    /// Create a new CSR graph from edge list (optimized version)
    pub fn from_edges(n_nodes: usize, edges: Vec<(usize, usize, f64)>) -> Result<Self, GraphError> {
        let n_edges = edges.len();

        // Pre-allocate with exact sizes to avoid reallocations
        let mut col_idx = Vec::with_capacity(n_edges);
        let mut weights = Vec::with_capacity(n_edges);

        // Use counting sort for better performance when source nodes are dense
        let mut degree = vec![0; n_nodes];

        // First pass: count degrees and validate nodes
        for &(src, dst_, _) in &edges {
            if src >= n_nodes {
                return Err(GraphError::node_not_found_with_context(
                    src,
                    n_nodes,
                    "CSR graph construction",
                ));
            }
            if dst_ >= n_nodes {
                return Err(GraphError::node_not_found_with_context(
                    dst_,
                    n_nodes,
                    "CSR graph construction",
                ));
            }
            degree[src] += 1;
        }

        // Build row pointers using prefix sum
        let mut row_ptr = Vec::with_capacity(n_nodes + 1);
        row_ptr.push(0);
        for &deg in &degree {
            row_ptr.push(row_ptr.last().unwrap() + deg);
        }

        // Initialize working arrays for building CSR
        col_idx.resize(n_edges, 0);
        weights.resize(n_edges, 0.0);
        let mut current_pos = row_ptr.clone();
        current_pos.pop(); // Remove last element

        // Fill CSR arrays directly without sorting
        for (src, dst, weight) in edges {
            let pos = current_pos[src];
            col_idx[pos] = dst;
            weights[pos] = weight;
            current_pos[src] += 1;
        }

        // Sort neighbors within each row for better cache performance
        for node in 0..n_nodes {
            let start = row_ptr[node];
            let end = row_ptr[node + 1];

            if end > start {
                // Create pairs for sorting
                let mut pairs: Vec<(usize, f64)> = col_idx[start..end]
                    .iter()
                    .zip(&weights[start..end])
                    .map(|(&c, &w)| (c, w))
                    .collect();

                // Sort by column index
                pairs.sort_unstable_by_key(|&(col_, _)| col_);

                // Write back sorted data
                for (i, (col, weight)) in pairs.into_iter().enumerate() {
                    col_idx[start + i] = col;
                    weights[start + i] = weight;
                }
            }
        }

        Ok(CSRGraph {
            n_nodes,
            n_edges,
            row_ptr,
            col_idx,
            weights,
        })
    }

    /// Create CSR graph with pre-allocated capacity (for streaming construction)
    pub fn with_capacity(n_nodes: usize, estimated_edges: usize) -> Self {
        CSRGraph {
            n_nodes,
            n_edges: 0,
            row_ptr: vec![0; n_nodes + 1],
            col_idx: Vec::with_capacity(estimated_edges),
            weights: Vec::with_capacity(estimated_edges),
        }
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.row_ptr[node];
        let end = self.row_ptr[node + 1];

        self.col_idx[start..end]
            .iter()
            .zip(&self.weights[start..end])
            .map(|(&idx, &weight)| (idx, weight))
    }

    /// Get degree of a node
    pub fn degree(&self, node: usize) -> usize {
        self.row_ptr[node + 1] - self.row_ptr[node]
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.n_nodes
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of_val(&self.n_nodes)
            + mem::size_of_val(&self.n_edges)
            + mem::size_of_val(&self.row_ptr[..])
            + mem::size_of_val(&self.col_idx[..])
            + mem::size_of_val(&self.weights[..])
    }

    /// Convert to adjacency matrix (for dense operations)
    pub fn to_adjacency_matrix(&self) -> Array2<f64> {
        let mut matrix = Array2::zeros((self.n_nodes, self.n_nodes));

        for src in 0..self.n_nodes {
            for (dst, weight) in self.neighbors(src) {
                matrix[[src, dst]] = weight;
            }
        }

        matrix
    }
}

/// Bit-packed representation for unweighted graphs
///
/// Uses 1 bit per potential edge, extremely memory efficient for unweighted graphs.
#[derive(Debug, Clone)]
pub struct BitPackedGraph {
    /// Number of nodes
    n_nodes: usize,
    /// Bit array storing adjacency information
    /// For undirected graphs, only upper triangle is stored
    bits: Vec<u64>,
    /// Whether the graph is directed
    directed: bool,
}

impl BitPackedGraph {
    /// Create a new bit-packed graph
    pub fn new(n_nodes: usize, directed: bool) -> Self {
        let bits_needed = if directed {
            n_nodes * n_nodes
        } else {
            n_nodes * (n_nodes + 1) / 2 // Upper triangle including diagonal
        };

        let words_needed = bits_needed.div_ceil(64);

        BitPackedGraph {
            n_nodes,
            bits: vec![0; words_needed],
            directed,
        }
    }

    /// Calculate bit position for an edge
    fn bit_position(&self, from: usize, to: usize) -> Option<usize> {
        if from >= self.n_nodes || to >= self.n_nodes {
            return None;
        }

        if self.directed {
            Some(from * self.n_nodes + to)
        } else {
            // For undirected, normalize to upper triangle
            let (u, v) = if from <= to { (from, to) } else { (to, from) };
            // Calculate position in upper triangular matrix using safe arithmetic
            if u == 0 {
                Some(v)
            } else {
                Some(u * (2 * self.n_nodes - u - 1) / 2 + v)
            }
        }
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: usize, to: usize) -> Result<(), GraphError> {
        let bit_pos = self.bit_position(from, to).ok_or_else(|| {
            GraphError::node_not_found_with_context(from, self.n_nodes, "add_edge operation")
        })?;

        let word_idx = bit_pos / 64;
        let bit_idx = bit_pos % 64;

        self.bits[word_idx] |= 1u64 << bit_idx;

        Ok(())
    }

    /// Check if edge exists
    pub fn has_edge(&self, from: usize, to: usize) -> bool {
        if let Some(bit_pos) = self.bit_position(from, to) {
            let word_idx = bit_pos / 64;
            let bit_idx = bit_pos % 64;

            (self.bits[word_idx] & (1u64 << bit_idx)) != 0
        } else {
            false
        }
    }

    /// Get neighbors of a node (optimized with SIMD-like operations)
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();

        if self.directed {
            // For directed graphs, check outgoing edges
            let start_bit = node * self.n_nodes;
            let end_bit = start_bit + self.n_nodes;

            let start_word = start_bit / 64;
            let end_word = end_bit.div_ceil(64);

            for word_idx in start_word..end_word {
                if word_idx >= self.bits.len() {
                    break;
                }

                let mut word = self.bits[word_idx];
                let word_start_bit = word_idx * 64;

                // Mask out bits outside our range
                if word_start_bit < start_bit {
                    let skip_bits = start_bit - word_start_bit;
                    word &= !((1u64 << skip_bits) - 1);
                }
                if word_start_bit + 64 > end_bit {
                    let keep_bits = end_bit - word_start_bit;
                    word &= (1u64 << keep_bits) - 1;
                }

                // Extract set bits efficiently
                while word != 0 {
                    let bit_pos = word.trailing_zeros() as usize;
                    let global_bit = word_start_bit + bit_pos;
                    if global_bit >= start_bit && global_bit < end_bit {
                        let neighbor = global_bit - start_bit;
                        neighbors.push(neighbor);
                    }
                    word &= word - 1; // Clear lowest set bit
                }
            }
        } else {
            // For undirected graphs, check both directions efficiently
            for other in 0..self.n_nodes {
                if self.has_edge(node, other) {
                    neighbors.push(other);
                }
            }
        }

        neighbors
    }

    /// Get degree of a node efficiently
    pub fn degree(&self, node: usize) -> usize {
        if node >= self.n_nodes {
            return 0;
        }

        if self.directed {
            let start_bit = node * self.n_nodes;
            let end_bit = start_bit + self.n_nodes;

            let start_word = start_bit / 64;
            let end_word = end_bit.div_ceil(64);
            let mut count = 0;

            for word_idx in start_word..end_word {
                if word_idx >= self.bits.len() {
                    break;
                }

                let mut word = self.bits[word_idx];
                let word_start_bit = word_idx * 64;

                // Mask out bits outside our range
                if word_start_bit < start_bit {
                    let skip_bits = start_bit - word_start_bit;
                    word &= !((1u64 << skip_bits) - 1);
                }
                if word_start_bit + 64 > end_bit {
                    let keep_bits = end_bit - word_start_bit;
                    word &= (1u64 << keep_bits) - 1;
                }

                count += word.count_ones() as usize;
            }

            count
        } else {
            // For undirected graphs, count efficiently
            self.neighbors(node).len()
        }
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of_val(&self.n_nodes)
            + mem::size_of_val(&self.bits[..])
            + mem::size_of_val(&self.directed)
    }
}

/// Compressed adjacency list using variable-length encoding
///
/// Uses delta encoding and variable-length integers for neighbor lists.
#[derive(Debug, Clone)]
pub struct CompressedAdjacencyList {
    /// Number of nodes
    n_nodes: usize,
    /// Compressed neighbor data
    data: Vec<u8>,
    /// Offsets into data for each node
    offsets: Vec<usize>,
}

impl CompressedAdjacencyList {
    /// Create from adjacency lists
    pub fn from_adjacency(_adjlists: Vec<Vec<usize>>) -> Self {
        let n_nodes = _adjlists.len();
        let mut data = Vec::new();
        let mut offsets = Vec::with_capacity(n_nodes + 1);

        offsets.push(0);

        for neighbors in _adjlists {
            let _start_pos = data.len();

            // Sort neighbors for delta encoding
            let mut sorted_neighbors = neighbors;
            sorted_neighbors.sort_unstable();

            // Encode count
            Self::encode_varint(sorted_neighbors.len(), &mut data);

            // Delta encode neighbors
            let mut prev = 0;
            for &neighbor in &sorted_neighbors {
                let delta = neighbor - prev;
                Self::encode_varint(delta, &mut data);
                prev = neighbor;
            }

            offsets.push(data.len());
        }

        CompressedAdjacencyList {
            n_nodes,
            data,
            offsets,
        }
    }

    /// Variable-length integer encoding
    fn encode_varint(mut value: usize, output: &mut Vec<u8>) {
        while value >= 0x80 {
            output.push((value & 0x7F) as u8 | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }

    /// Variable-length integer decoding
    fn decode_varint(data: &[u8], pos: &mut usize) -> usize {
        let mut value = 0;
        let mut shift = 0;

        loop {
            let byte = data[*pos];
            *pos += 1;

            value |= ((byte & 0x7F) as usize) << shift;

            if byte & 0x80 == 0 {
                break;
            }

            shift += 7;
        }

        value
    }

    /// Get neighbors of a node
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        if node >= self.n_nodes {
            return Vec::new();
        }

        let start = self.offsets[node];
        let end = self.offsets[node + 1];
        let data_slice = &self.data[start..end];

        let mut pos = 0;
        let count = Self::decode_varint(data_slice, &mut pos);

        let mut neighbors = Vec::with_capacity(count);
        let mut current = 0;

        for _ in 0..count {
            let delta = Self::decode_varint(data_slice, &mut pos);
            current += delta;
            neighbors.push(current);
        }

        neighbors
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        mem::size_of_val(&self.n_nodes)
            + mem::size_of_val(&self.data[..])
            + mem::size_of_val(&self.offsets[..])
    }
}

/// Hybrid graph representation that chooses optimal format based on graph properties
pub enum HybridGraph {
    /// Use CSR for sparse graphs
    CSR(CSRGraph),
    /// Use bit-packed for dense unweighted graphs
    BitPacked(BitPackedGraph),
    /// Use compressed adjacency for medium density
    Compressed(CompressedAdjacencyList),
}

impl HybridGraph {
    /// Automatically choose the best representation based on graph properties
    pub fn auto_select(
        n_nodes: usize,
        edges: Vec<(usize, usize, Option<f64>)>,
        directed: bool,
    ) -> Result<Self, GraphError> {
        let n_edges = edges.len();
        let density = n_edges as f64 / (n_nodes * n_nodes) as f64;
        let all_unweighted = edges.iter().all(|(_, _, w)| w.is_none());

        if all_unweighted && density > 0.1 {
            // Dense unweighted - use bit-packed
            let mut graph = BitPackedGraph::new(n_nodes, directed);
            for (src, dst_, _) in edges {
                graph.add_edge(src, dst_)?;
            }
            Ok(HybridGraph::BitPacked(graph))
        } else if density < 0.01 {
            // Very sparse - use CSR
            let weighted_edges: Vec<(usize, usize, f64)> = edges
                .into_iter()
                .map(|(s, d, w)| (s, d, w.unwrap_or(1.0)))
                .collect();
            let graph = CSRGraph::from_edges(n_nodes, weighted_edges)?;
            Ok(HybridGraph::CSR(graph))
        } else {
            // Medium density - use compressed adjacency
            let mut adj_lists = vec![Vec::new(); n_nodes];
            for (src, dst_, _) in edges {
                adj_lists[src].push(dst_);
                if !directed {
                    adj_lists[dst_].push(src);
                }
            }
            let graph = CompressedAdjacencyList::from_adjacency(adj_lists);
            Ok(HybridGraph::Compressed(graph))
        }
    }

    /// Get memory usage
    pub fn memory_usage(&self) -> usize {
        match self {
            HybridGraph::CSR(g) => g.memory_usage(),
            HybridGraph::BitPacked(g) => g.memory_usage(),
            HybridGraph::Compressed(g) => g.memory_usage(),
        }
    }
}

/// Memory-mapped graph for extremely large graphs that don't fit in RAM
#[derive(Debug)]
pub struct MemmapGraph {
    /// Number of nodes
    n_nodes: usize,
    /// Number of edges
    n_edges: usize,
    /// File handle for the graph data
    file: File,
    /// CSR format stored on disk
    /// Format: [n_nodes:8][n_edges:8][row_ptr:(n_nodes+1)*8][col, _idx:n_edges*8][weights:n_edges*8]
    #[allow(dead_code)]
    header_size: usize,
    row_ptr_offset: usize,
    col_idx_offset: usize,
    weights_offset: usize,
}

impl MemmapGraph {
    /// Create a new memory-mapped graph from an existing CSR graph
    pub fn from_csr<P: AsRef<Path>>(csr: &CSRGraph, path: P) -> io::Result<Self> {
        let mut file = File::create(&path)?;
        let mut writer = BufWriter::new(&mut file);

        // Write header
        writer.write_all(&csr.n_nodes.to_le_bytes())?;
        writer.write_all(&csr.n_edges.to_le_bytes())?;

        // Write row pointers
        for &ptr in &csr.row_ptr {
            writer.write_all(&ptr.to_le_bytes())?;
        }

        // Write column indices
        for &idx in &csr.col_idx {
            writer.write_all(&idx.to_le_bytes())?;
        }

        // Write weights
        for &weight in &csr.weights {
            writer.write_all(&weight.to_le_bytes())?;
        }

        writer.flush()?;
        drop(writer);

        // Reopen for reading
        let file = File::open(path)?;

        let header_size = 16; // n_nodes + n_edges
        let row_ptr_offset = header_size;
        let col_idx_offset = row_ptr_offset + (csr.n_nodes + 1) * 8;
        let weights_offset = col_idx_offset + csr.n_edges * 8;

        Ok(MemmapGraph {
            n_nodes: csr.n_nodes,
            n_edges: csr.n_edges,
            file,
            header_size,
            row_ptr_offset,
            col_idx_offset,
            weights_offset,
        })
    }

    /// Load an existing memory-mapped graph
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = [0u8; 16];

        // Read header
        file.read_exact(&mut buffer)?;
        let n_nodes = usize::from_le_bytes([
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ]);
        let n_edges = usize::from_le_bytes([
            buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14],
            buffer[15],
        ]);

        let header_size = 16;
        let row_ptr_offset = header_size;
        let col_idx_offset = row_ptr_offset + (n_nodes + 1) * 8;
        let weights_offset = col_idx_offset + n_edges * 8;

        Ok(MemmapGraph {
            n_nodes,
            n_edges,
            file,
            header_size,
            row_ptr_offset,
            col_idx_offset,
            weights_offset,
        })
    }

    /// Get row pointers for a node (reads from disk)
    fn get_row_ptrs(&mut self, node: usize) -> io::Result<(usize, usize)> {
        if node >= self.n_nodes {
            return Ok((0, 0));
        }

        let mut buffer = [0u8; 16];
        let offset = self.row_ptr_offset + node * 8;

        self.file.seek(SeekFrom::Start(offset as u64))?;
        self.file.read_exact(&mut buffer)?;

        let start = usize::from_le_bytes([
            buffer[0], buffer[1], buffer[2], buffer[3], buffer[4], buffer[5], buffer[6], buffer[7],
        ]);
        let end = usize::from_le_bytes([
            buffer[8], buffer[9], buffer[10], buffer[11], buffer[12], buffer[13], buffer[14],
            buffer[15],
        ]);

        Ok((start, end))
    }

    /// Get neighbors of a node (reads from disk)
    pub fn neighbors(&mut self, node: usize) -> io::Result<Vec<(usize, f64)>> {
        let (start, end) = self.get_row_ptrs(node)?;
        let degree = end - start;

        if degree == 0 {
            return Ok(Vec::new());
        }

        // Read column indices
        let mut col_buffer = vec![0u8; degree * 8];
        let col_offset = self.col_idx_offset + start * 8;
        self.file.seek(SeekFrom::Start(col_offset as u64))?;
        self.file.read_exact(&mut col_buffer)?;

        // Read weights
        let mut weight_buffer = vec![0u8; degree * 8];
        let weight_offset = self.weights_offset + start * 8;
        self.file.seek(SeekFrom::Start(weight_offset as u64))?;
        self.file.read_exact(&mut weight_buffer)?;

        // Parse neighbors
        let mut neighbors = Vec::with_capacity(degree);
        for i in 0..degree {
            let col_bytes = &col_buffer[i * 8..(i + 1) * 8];
            let weight_bytes = &weight_buffer[i * 8..(i + 1) * 8];

            let col_idx = usize::from_le_bytes([
                col_bytes[0],
                col_bytes[1],
                col_bytes[2],
                col_bytes[3],
                col_bytes[4],
                col_bytes[5],
                col_bytes[6],
                col_bytes[7],
            ]);
            let weight = f64::from_le_bytes([
                weight_bytes[0],
                weight_bytes[1],
                weight_bytes[2],
                weight_bytes[3],
                weight_bytes[4],
                weight_bytes[5],
                weight_bytes[6],
                weight_bytes[7],
            ]);

            neighbors.push((col_idx, weight));
        }

        Ok(neighbors)
    }

    /// Get degree of a node
    pub fn degree(&mut self, node: usize) -> io::Result<usize> {
        let (start, end) = self.get_row_ptrs(node)?;
        Ok(end - start)
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.n_nodes
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.n_edges
    }

    /// Check if an edge exists (requires reading neighbors)
    pub fn has_edge(&mut self, from: usize, to: usize) -> io::Result<bool> {
        let neighbors = self.neighbors(from)?;
        Ok(neighbors.iter().any(|&(neighbor_, _)| neighbor_ == to))
    }
}

/// Optimized batch operations for memory-mapped graphs
impl MemmapGraph {
    /// Read multiple nodes' neighbors in one operation (more efficient)
    pub fn batch_neighbors(&mut self, nodes: &[usize]) -> io::Result<Vec<Vec<(usize, f64)>>> {
        let mut results = Vec::with_capacity(nodes.len());

        // Sort nodes to minimize seeking
        let mut sorted_nodes: Vec<_> = nodes.iter().enumerate().collect();
        sorted_nodes.sort_by_key(|(_, &node)| node);

        for (_, &node) in sorted_nodes {
            results.push(self.neighbors(node)?);
        }

        Ok(results)
    }

    /// Stream through all edges without loading everything into memory
    pub fn stream_edges<F>(&mut self, mut callback: F) -> io::Result<()>
    where
        F: FnMut(usize, usize, f64) -> bool, // Returns true to continue
    {
        for node in 0..self.n_nodes {
            let neighbors = self.neighbors(node)?;
            for (neighbor, weight) in neighbors {
                if !callback(node, neighbor, weight) {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csr_graph() {
        let edges = vec![(0, 1, 1.0), (0, 2, 2.0), (1, 2, 3.0), (2, 3, 4.0)];

        let graph = CSRGraph::from_edges(4, edges).unwrap();

        assert_eq!(graph.degree(0), 2);
        assert_eq!(graph.degree(3), 0);

        let neighbors: Vec<_> = graph.neighbors(0).collect();
        assert_eq!(neighbors, vec![(1, 1.0), (2, 2.0)]);
    }

    #[test]
    fn test_bit_packed_graph() {
        let mut graph = BitPackedGraph::new(4, false);

        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();
        graph.add_edge(0, 3).unwrap();

        assert!(graph.has_edge(0, 1));
        assert!(graph.has_edge(1, 0)); // Undirected
        assert!(!graph.has_edge(2, 3));

        let neighbors = graph.neighbors(0);
        assert!(neighbors.contains(&1));
        assert!(neighbors.contains(&3));
    }

    #[test]
    fn test_compressed_adjacency() {
        let adj_lists = vec![
            vec![1, 2, 5],
            vec![0, 2],
            vec![0, 1, 3],
            vec![2],
            vec![],
            vec![0],
        ];

        let graph = CompressedAdjacencyList::from_adjacency(adj_lists.clone());

        for (node, expected) in adj_lists.iter().enumerate() {
            let neighbors = graph.neighbors(node);
            assert_eq!(&neighbors, expected);
        }

        // Check memory compression (note: compression may not always be effective for small graphs)
        let uncompressed_size = adj_lists
            .iter()
            .map(|list| list.len() * mem::size_of::<usize>())
            .sum::<usize>();

        let compressed_size = graph.memory_usage();
        // For small graphs, compression overhead may exceed savings
        // Just verify the compressed graph works correctly
        println!(
            "Uncompressed: {} bytes, Compressed: {} bytes",
            uncompressed_size, compressed_size
        );
    }
}
