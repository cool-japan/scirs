//! Graph data structures for GNN operations.
//!
//! Provides both dense (`Graph`) and sparse CSR (`SparseGraph`) representations
//! suitable for feeding into GCN, GAT, GraphSAGE and GIN layers.

use crate::error::{NeuralError, Result};

// ──────────────────────────────────────────────────────────────────────────────
// Dense Graph
// ──────────────────────────────────────────────────────────────────────────────

/// Dense adjacency-matrix graph representation.
///
/// Node features are stored as a `Vec<Vec<f32>>` of shape `[N, F]`.  The
/// adjacency matrix is `[N, N]`.  An optional separate weight matrix can
/// override the implicit binary weights derived from the adjacency matrix.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of nodes in the graph.
    pub num_nodes: usize,
    /// Dense adjacency matrix `[N, N]`.  `adjacency[i][j] > 0` means there
    /// is an edge from node *i* to node *j*.
    pub adjacency: Vec<Vec<f32>>,
    /// Node feature matrix `[N, F]`.
    pub node_features: Vec<Vec<f32>>,
    /// Optional explicit edge weight matrix `[N, N]`.  When `None` the
    /// non-zero entries of `adjacency` are used as edge weights.
    pub edge_weights: Option<Vec<Vec<f32>>>,
}

impl Graph {
    /// Create a new empty graph with `num_nodes` nodes and `feature_dim`-
    /// dimensional feature vectors initialised to zero.
    pub fn new(num_nodes: usize, feature_dim: usize) -> Self {
        Graph {
            num_nodes,
            adjacency: vec![vec![0.0_f32; num_nodes]; num_nodes],
            node_features: vec![vec![0.0_f32; feature_dim]; num_nodes],
            edge_weights: None,
        }
    }

    /// Add a directed edge from `src` to `dst` with the given `weight`.
    ///
    /// Returns an error if either node index is out of range.
    pub fn add_edge(&mut self, src: usize, dst: usize, weight: f32) -> Result<()> {
        if src >= self.num_nodes || dst >= self.num_nodes {
            return Err(NeuralError::InvalidArgument(format!(
                "Edge ({src}, {dst}) out of bounds for graph with {} nodes",
                self.num_nodes
            )));
        }
        self.adjacency[src][dst] = weight;
        Ok(())
    }

    /// Add an undirected (symmetric) unit-weight edge between `src` and `dst`.
    pub fn add_undirected_edge(&mut self, src: usize, dst: usize) -> Result<()> {
        self.add_edge(src, dst, 1.0)?;
        self.add_edge(dst, src, 1.0)?;
        Ok(())
    }

    /// Replace the feature vector for `node` with `features`.
    ///
    /// Returns an error if the node index is out of range.
    pub fn set_node_features(&mut self, node: usize, features: Vec<f32>) -> Result<()> {
        if node >= self.num_nodes {
            return Err(NeuralError::InvalidArgument(format!(
                "Node index {node} out of bounds for graph with {} nodes",
                self.num_nodes
            )));
        }
        self.node_features[node] = features;
        Ok(())
    }

    /// Compute the degree of each node (sum of outgoing edge weights).
    ///
    /// Returns a vector of length `N` containing the diagonal entries of the
    /// degree matrix *D*.
    pub fn degree_matrix(&self) -> Vec<f32> {
        (0..self.num_nodes)
            .map(|i| self.adjacency[i].iter().copied().sum::<f32>())
            .collect()
    }

    /// Return the symmetrically normalised adjacency matrix
    /// *Ã = D̃^{-½} (A + I) D̃^{-½}* where *D̃* is the degree matrix of
    /// *(A + I)*.
    ///
    /// Self-loops are added before normalisation so that every node aggregates
    /// its own features as well.
    pub fn normalized_adjacency(&self) -> Vec<Vec<f32>> {
        let n = self.num_nodes;
        // Compute Ã = A + I and its degree
        let mut a_tilde: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let mut row = self.adjacency[i].clone();
                row[i] += 1.0; // add self-loop
                row
            })
            .collect();

        // Degree of Ã
        let d_tilde: Vec<f32> = (0..n)
            .map(|i| a_tilde[i].iter().copied().sum::<f32>())
            .collect();

        // D̃^{-½}
        let d_inv_sqrt: Vec<f32> = d_tilde
            .iter()
            .map(|&d| if d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // D̃^{-½} Ã D̃^{-½}
        for i in 0..n {
            for j in 0..n {
                a_tilde[i][j] *= d_inv_sqrt[i] * d_inv_sqrt[j];
            }
        }
        a_tilde
    }

    /// Return the list of neighbour indices of `node` (nodes *j* such that
    /// `adjacency[node][j] > 0`).
    pub fn neighbors(&self, node: usize) -> Vec<usize> {
        if node >= self.num_nodes {
            return Vec::new();
        }
        self.adjacency[node]
            .iter()
            .enumerate()
            .filter_map(|(j, &w)| if w > 0.0 { Some(j) } else { None })
            .collect()
    }

    /// Return the total number of edges (directed, non-zero adjacency entries).
    pub fn num_edges(&self) -> usize {
        self.adjacency
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&w| w > 0.0)
            .count()
    }

    /// Return the number of node features (feature dimension *F*).
    ///
    /// Returns `0` if there are no nodes.
    pub fn feature_dim(&self) -> usize {
        self.node_features.first().map(|v| v.len()).unwrap_or(0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Sparse CSR Graph
// ──────────────────────────────────────────────────────────────────────────────

/// Sparse graph in *Compressed Sparse Row* (CSR) format.
///
/// Memory-efficient for large, sparse graphs.  The `row_ptr` array has length
/// `N + 1`; edges of node *i* are stored at `col_idx[row_ptr[i]..row_ptr[i+1]]`
/// with corresponding weights in `values[row_ptr[i]..row_ptr[i+1]]`.
#[derive(Debug, Clone)]
pub struct SparseGraph {
    /// Number of nodes.
    pub num_nodes: usize,
    /// Row pointer array of length `N + 1`.
    pub row_ptr: Vec<usize>,
    /// Column-index array of length `E` (one entry per directed edge).
    pub col_idx: Vec<usize>,
    /// Edge weight array of length `E`.
    pub values: Vec<f32>,
    /// Node feature matrix `[N, F]`.
    pub node_features: Vec<Vec<f32>>,
}

impl SparseGraph {
    /// Construct a `SparseGraph` from a list of `(src, dst, weight)` tuples.
    ///
    /// Edges are sorted by `src` so that the CSR row-pointer array can be
    /// built in O(E log E).  `features` must have exactly `num_nodes` rows.
    ///
    /// # Errors
    /// Returns `NeuralError::InvalidArgument` if any node index is out of
    /// range or if `features.len() != num_nodes`.
    pub fn from_edges(
        num_nodes: usize,
        edges: &[(usize, usize, f32)],
        features: Vec<Vec<f32>>,
    ) -> Result<Self> {
        if features.len() != num_nodes {
            return Err(NeuralError::InvalidArgument(format!(
                "features.len() ({}) must equal num_nodes ({})",
                features.len(),
                num_nodes
            )));
        }
        for &(src, dst, _) in edges {
            if src >= num_nodes || dst >= num_nodes {
                return Err(NeuralError::InvalidArgument(format!(
                    "Edge ({src}, {dst}) out of bounds for graph with {num_nodes} nodes"
                )));
            }
        }

        // Sort edges by source node for CSR construction
        let mut sorted_edges: Vec<(usize, usize, f32)> = edges.to_vec();
        sorted_edges.sort_by_key(|&(s, d, _)| (s, d));

        let num_edges = sorted_edges.len();
        let mut row_ptr = vec![0usize; num_nodes + 1];
        let mut col_idx = Vec::with_capacity(num_edges);
        let mut values = Vec::with_capacity(num_edges);

        // Count edges per row
        for &(src, _, _) in &sorted_edges {
            row_ptr[src + 1] += 1;
        }
        // Prefix sum
        for i in 0..num_nodes {
            row_ptr[i + 1] += row_ptr[i];
        }

        for &(_, dst, w) in &sorted_edges {
            col_idx.push(dst);
            values.push(w);
        }

        Ok(SparseGraph {
            num_nodes,
            row_ptr,
            col_idx,
            values,
            node_features: features,
        })
    }

    /// Return the column indices (neighbour node IDs) of `node`.
    pub fn neighbors(&self, node: usize) -> &[usize] {
        if node >= self.num_nodes {
            return &[];
        }
        &self.col_idx[self.row_ptr[node]..self.row_ptr[node + 1]]
    }

    /// Return the edge weights of outgoing edges from `node`.
    pub fn edge_weights_of(&self, node: usize) -> &[f32] {
        if node >= self.num_nodes {
            return &[];
        }
        &self.values[self.row_ptr[node]..self.row_ptr[node + 1]]
    }

    /// Compute the degree of each node (sum of outgoing edge weights).
    fn degrees(&self) -> Vec<f32> {
        (0..self.num_nodes)
            .map(|i| self.edge_weights_of(i).iter().copied().sum::<f32>())
            .collect()
    }

    /// Return the normalised Laplacian *L = I − D^{-½} A D^{-½}* as a new
    /// `SparseGraph`.
    ///
    /// The resulting graph stores the negated, normalised adjacency values
    /// (off-diagonal entries of *−D^{-½} A D^{-½}*).  Diagonal self-loop
    /// entries of value `1.0` are **not** added here; callers that need the
    /// full Laplacian should add self-loops separately.
    pub fn normalized_laplacian(&self) -> Result<Self> {
        let d = self.degrees();
        let d_inv_sqrt: Vec<f32> = d
            .iter()
            .map(|&deg| if deg > 0.0 { 1.0 / deg.sqrt() } else { 0.0 })
            .collect();

        let d_ref = &d_inv_sqrt;
        let new_values: Vec<f32> = (0..self.num_nodes)
            .flat_map(|i| {
                let start = self.row_ptr[i];
                let end = self.row_ptr[i + 1];
                let di = d_ref[i];
                let col_slice: Vec<usize> = self.col_idx[start..end].to_vec();
                let val_slice: Vec<f32> = self.values[start..end].to_vec();
                let d_clone: Vec<f32> = d_ref.to_vec();
                (0..(end - start)).map(move |idx| {
                    let j = col_slice[idx];
                    let a_ij = val_slice[idx];
                    // −D^{-½}_{ii} * A_{ij} * D^{-½}_{jj}
                    -di * a_ij * d_clone[j]
                })
            })
            .collect();

        Ok(SparseGraph {
            num_nodes: self.num_nodes,
            row_ptr: self.row_ptr.clone(),
            col_idx: self.col_idx.clone(),
            values: new_values,
            node_features: self.node_features.clone(),
        })
    }

    /// Return the number of (directed) edges.
    pub fn num_edges(&self) -> usize {
        self.col_idx.len()
    }

    /// Return the feature dimension (0 if no nodes).
    pub fn feature_dim(&self) -> usize {
        self.node_features.first().map(|v| v.len()).unwrap_or(0)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_new_and_add_edge() {
        let mut g = Graph::new(4, 3);
        g.add_edge(0, 1, 1.0).expect("add_edge failed");
        g.add_edge(1, 2, 0.5).expect("add_edge failed");
        g.add_undirected_edge(2, 3).expect("add_undirected failed");
        assert_eq!(g.num_edges(), 4); // 0→1, 1→2, 2→3, 3→2
    }

    #[test]
    fn test_graph_neighbors() {
        let mut g = Graph::new(3, 2);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(0, 2, 1.0).expect("ok");
        let nb = g.neighbors(0);
        assert_eq!(nb.len(), 2);
        assert!(nb.contains(&1));
        assert!(nb.contains(&2));
        let nb1 = g.neighbors(1);
        assert!(nb1.is_empty());
    }

    #[test]
    fn test_graph_degree_matrix() {
        let mut g = Graph::new(3, 1);
        g.add_edge(0, 1, 2.0).expect("ok");
        g.add_edge(0, 2, 3.0).expect("ok");
        let d = g.degree_matrix();
        assert!((d[0] - 5.0).abs() < 1e-6);
        assert!((d[1]).abs() < 1e-6);
    }

    #[test]
    fn test_normalized_adjacency() {
        let mut g = Graph::new(2, 1);
        g.add_edge(0, 1, 1.0).expect("ok");
        g.add_edge(1, 0, 1.0).expect("ok");
        let norm = g.normalized_adjacency();
        // D̃ after adding self-loops: node 0 and 1 each have degree 2
        // D̃^{-½} = [[1/sqrt(2), 0], [0, 1/sqrt(2)]]
        // Normalised diagonal = 1.0/2.0 = 0.5
        let expected_diag = 0.5_f32;
        assert!(
            (norm[0][0] - expected_diag).abs() < 1e-5,
            "diag={}",
            norm[0][0]
        );
    }

    #[test]
    fn test_sparse_graph_from_edges() {
        let edges = vec![(0, 1, 1.0_f32), (1, 2, 1.0), (2, 0, 1.0)];
        let features = vec![vec![1.0_f32, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let sg = SparseGraph::from_edges(3, &edges, features).expect("from_edges failed");
        assert_eq!(sg.num_edges(), 3);
        assert_eq!(sg.neighbors(0), &[1]);
        assert_eq!(sg.neighbors(1), &[2]);
        assert_eq!(sg.neighbors(2), &[0]);
    }

    #[test]
    fn test_sparse_graph_invalid_features_len() {
        let edges = vec![(0, 1, 1.0_f32)];
        let features = vec![vec![1.0_f32]]; // only 1 row for 2-node graph
        let result = SparseGraph::from_edges(2, &edges, features);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_normalized_laplacian() {
        let edges = vec![(0, 1, 1.0_f32), (1, 0, 1.0)];
        let features = vec![vec![0.0_f32; 2]; 2];
        let sg = SparseGraph::from_edges(2, &edges, features).expect("ok");
        let lap = sg.normalized_laplacian().expect("laplacian ok");
        // Both off-diagonal entries should be −0.5 (−1/sqrt(1)*1*1/sqrt(1))
        // wait: degree of each node = 1, so d_inv_sqrt = 1.0
        assert!((lap.values[0] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_graph_out_of_bounds_edge() {
        let mut g = Graph::new(3, 1);
        assert!(g.add_edge(0, 99, 1.0).is_err());
    }
}
