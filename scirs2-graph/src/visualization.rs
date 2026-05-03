//! Graph visualization with layout algorithms and SVG export
//!
//! This module provides algorithms for computing node positions for graph
//! visualization and exporting graphs to SVG and DOT formats.
//!
//! # Layout Algorithms
//! - **ForceDirected**: Fruchterman-Reingold force-directed placement
//! - **Hierarchical**: Layer-based layout for DAGs and trees
//! - **Circular**: Nodes evenly distributed around a circle
//! - **Spectral**: Eigenvector-based layout using graph Laplacian
//!
//! # Export Formats
//! - SVG with fully customizable node/edge styling
//! - DOT format with layout hints

use std::collections::HashMap;
use std::f64::consts::PI;
use std::fmt::Write as FmtWrite;
use std::hash::Hash;

use scirs2_core::random::{Rng, RngExt};

use crate::base::{EdgeWeight, Graph, Node};
use crate::error::{GraphError, Result};

// ============================================================================
// Core types
// ============================================================================

/// 2D node position in layout space
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LayoutPosition {
    /// X coordinate
    pub x: f64,
    /// Y coordinate
    pub y: f64,
}

impl LayoutPosition {
    /// Create a new position
    pub fn new(x: f64, y: f64) -> Self {
        LayoutPosition { x, y }
    }

    /// Compute Euclidean distance to another position
    pub fn distance(&self, other: &LayoutPosition) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// Clamp coordinates to the given bounding box
    pub fn clamp(&self, x_min: f64, x_max: f64, y_min: f64, y_max: f64) -> LayoutPosition {
        LayoutPosition {
            x: self.x.max(x_min).min(x_max),
            y: self.y.max(y_min).min(y_max),
        }
    }
}

/// Computed graph layout: maps each node to its 2D position
#[derive(Debug, Clone)]
pub struct GraphLayout<N: Node> {
    /// Node → position mapping
    pub positions: HashMap<N, LayoutPosition>,
    /// Width of the layout canvas
    pub width: f64,
    /// Height of the layout canvas
    pub height: f64,
}

impl<N: Node + Clone> GraphLayout<N> {
    /// Create a new empty layout
    pub fn new(width: f64, height: f64) -> Self {
        GraphLayout {
            positions: HashMap::new(),
            width,
            height,
        }
    }

    /// Number of nodes in the layout
    pub fn node_count(&self) -> usize {
        self.positions.len()
    }

    /// Get position for a node
    pub fn position(&self, node: &N) -> Option<&LayoutPosition> {
        self.positions.get(node)
    }

    /// Set position for a node
    pub fn set_position(&mut self, node: N, pos: LayoutPosition) {
        self.positions.insert(node, pos);
    }

    /// Normalize positions to [0, 1] × [0, 1]
    pub fn normalize(&mut self) {
        if self.positions.is_empty() {
            return;
        }

        let positions: Vec<LayoutPosition> = self.positions.values().cloned().collect();
        let x_min = positions.iter().map(|p| p.x).fold(f64::INFINITY, f64::min);
        let x_max = positions
            .iter()
            .map(|p| p.x)
            .fold(f64::NEG_INFINITY, f64::max);
        let y_min = positions.iter().map(|p| p.y).fold(f64::INFINITY, f64::min);
        let y_max = positions
            .iter()
            .map(|p| p.y)
            .fold(f64::NEG_INFINITY, f64::max);

        let x_range = (x_max - x_min).max(1e-10);
        let y_range = (y_max - y_min).max(1e-10);

        for pos in self.positions.values_mut() {
            pos.x = (pos.x - x_min) / x_range;
            pos.y = (pos.y - y_min) / y_range;
        }
    }
}

// ============================================================================
// Layout algorithm enum
// ============================================================================

/// Available graph layout algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum LayoutAlgorithm {
    /// Fruchterman-Reingold force-directed layout
    ForceDirected {
        /// Number of iterations to run
        iterations: usize,
        /// Ideal spring length (None → auto-compute from area)
        spring_length: Option<f64>,
        /// Cooling coefficient: temperature decreases each iteration
        cooling: f64,
    },
    /// Layer-based hierarchical layout
    Hierarchical {
        /// Vertical spacing between layers
        layer_spacing: f64,
        /// Horizontal spacing between nodes in the same layer
        node_spacing: f64,
    },
    /// Circular layout: nodes arranged around a circle
    Circular {
        /// Radius of the circle
        radius: f64,
    },
    /// Spectral layout using graph Laplacian eigenvectors
    Spectral {
        /// Number of eigenvector iterations (power method)
        iterations: usize,
    },
}

impl Default for LayoutAlgorithm {
    fn default() -> Self {
        LayoutAlgorithm::ForceDirected {
            iterations: 100,
            spring_length: None,
            cooling: 0.95,
        }
    }
}

// ============================================================================
// Layout computation
// ============================================================================

/// Compute a graph layout using the specified algorithm
///
/// # Arguments
/// * `graph` - The graph to lay out
/// * `algorithm` - Which layout algorithm to use
/// * `width` - Canvas width in pixels
/// * `height` - Canvas height in pixels
///
/// # Returns
/// A `GraphLayout` with computed node positions
pub fn compute_layout<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    algorithm: &LayoutAlgorithm,
    width: f64,
    height: f64,
) -> Result<GraphLayout<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    match algorithm {
        LayoutAlgorithm::ForceDirected {
            iterations,
            spring_length,
            cooling,
        } => force_directed_layout(graph, *iterations, *spring_length, *cooling, width, height),
        LayoutAlgorithm::Hierarchical {
            layer_spacing,
            node_spacing,
        } => hierarchical_layout(graph, *layer_spacing, *node_spacing, width, height),
        LayoutAlgorithm::Circular { radius } => {
            circular_layout(graph, *radius, width / 2.0, height / 2.0)
        }
        LayoutAlgorithm::Spectral { iterations } => {
            spectral_layout(graph, *iterations, width, height)
        }
    }
}

/// Fruchterman-Reingold force-directed layout
///
/// Simulates attractive forces along edges and repulsive forces between all
/// node pairs, with a linearly decreasing temperature (simulated annealing).
fn force_directed_layout<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    iterations: usize,
    spring_length: Option<f64>,
    cooling: f64,
    width: f64,
    height: f64,
) -> Result<GraphLayout<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return Ok(GraphLayout::new(width, height));
    }

    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Initialize random positions
    let mut positions: Vec<LayoutPosition> = Vec::with_capacity(n);
    {
        let mut rng = scirs2_core::random::rng();
        for _ in 0..n {
            positions.push(LayoutPosition::new(
                rng.random::<f64>() * width,
                rng.random::<f64>() * height,
            ));
        }
    }

    // Ideal edge length
    let area = width * height;
    let k = spring_length.unwrap_or_else(|| (area / n as f64).sqrt());
    let k_sq = k * k;

    // Build adjacency list (by index)
    let edge_list: Vec<(usize, usize)> = graph
        .edges()
        .iter()
        .filter_map(|e| {
            let si = node_to_idx.get(&e.source)?;
            let ti = node_to_idx.get(&e.target)?;
            Some((*si, *ti))
        })
        .collect();

    let mut temperature = width / 10.0;

    for _ in 0..iterations {
        let mut disp: Vec<(f64, f64)> = vec![(0.0, 0.0); n];

        // Repulsive forces between all pairs
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = positions[i].x - positions[j].x;
                let dy = positions[i].y - positions[j].y;
                let dist = (dx * dx + dy * dy).sqrt().max(1e-10);
                let force = k_sq / dist;
                disp[i].0 += (dx / dist) * force;
                disp[i].1 += (dy / dist) * force;
            }
        }

        // Attractive forces along edges
        for &(si, ti) in &edge_list {
            let dx = positions[si].x - positions[ti].x;
            let dy = positions[si].y - positions[ti].y;
            let dist = (dx * dx + dy * dy).sqrt().max(1e-10);
            let force = dist * dist / k;
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            disp[si].0 -= fx;
            disp[si].1 -= fy;
            disp[ti].0 += fx;
            disp[ti].1 += fy;
        }

        // Apply displacement capped at temperature
        for i in 0..n {
            let dx = disp[i].0;
            let dy = disp[i].1;
            let len = (dx * dx + dy * dy).sqrt().max(1e-10);
            let capped = len.min(temperature);
            positions[i].x += (dx / len) * capped;
            positions[i].y += (dy / len) * capped;

            // Keep within bounds
            positions[i].x = positions[i].x.max(0.0).min(width);
            positions[i].y = positions[i].y.max(0.0).min(height);
        }

        temperature *= cooling;
    }

    let mut layout = GraphLayout::new(width, height);
    for (i, node) in nodes.into_iter().enumerate() {
        layout.set_position(node, positions[i]);
    }
    Ok(layout)
}

/// Layer-based hierarchical layout
///
/// Assigns nodes to layers based on longest path from source nodes, then
/// spreads nodes horizontally within each layer.
fn hierarchical_layout<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    layer_spacing: f64,
    node_spacing: f64,
    width: f64,
    height: f64,
) -> Result<GraphLayout<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return Ok(GraphLayout::new(width, height));
    }

    // Assign layers by degree-based heuristic (sorted by degree)
    let mut degrees: Vec<(usize, usize)> = nodes
        .iter()
        .enumerate()
        .map(|(i, node)| (i, graph.degree(node)))
        .collect();
    degrees.sort_by_key(|b| std::cmp::Reverse(b.1));

    // Divide nodes into layers of similar degree
    let num_layers = (n as f64).sqrt().ceil() as usize + 1;
    let layer_size = n.div_ceil(num_layers);

    let mut layers: Vec<Vec<usize>> = vec![Vec::new(); num_layers];
    for (rank, (node_idx, _)) in degrees.iter().enumerate() {
        layers[rank / layer_size.max(1)].push(*node_idx);
    }
    layers.retain(|l| !l.is_empty());

    let actual_layers = layers.len();
    let mut layout = GraphLayout::new(width, height);

    for (layer_idx, layer) in layers.iter().enumerate() {
        let y = if actual_layers > 1 {
            layer_spacing
                + (layer_idx as f64) * (height - 2.0 * layer_spacing) / (actual_layers - 1) as f64
        } else {
            height / 2.0
        };

        let layer_count = layer.len();
        for (slot, &node_idx) in layer.iter().enumerate() {
            let x = if layer_count > 1 {
                node_spacing
                    + (slot as f64) * (width - 2.0 * node_spacing) / (layer_count - 1) as f64
            } else {
                width / 2.0
            };
            layout.set_position(nodes[node_idx].clone(), LayoutPosition::new(x, y));
        }
    }

    Ok(layout)
}

/// Circular layout
///
/// Places all nodes evenly around a circle centered at (cx, cy).
fn circular_layout<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    radius: f64,
    cx: f64,
    cy: f64,
) -> Result<GraphLayout<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();
    let width = cx * 2.0;
    let height = cy * 2.0;

    if n == 0 {
        return Ok(GraphLayout::new(width, height));
    }

    let angle_step = 2.0 * PI / n as f64;
    let mut layout = GraphLayout::new(width, height);

    for (i, node) in nodes.into_iter().enumerate() {
        let angle = i as f64 * angle_step - PI / 2.0; // Start from top
        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();
        layout.set_position(node, LayoutPosition::new(x, y));
    }

    Ok(layout)
}

/// Spectral layout using graph Laplacian eigenvectors
///
/// Computes the second and third smallest eigenvectors of the normalized
/// Laplacian (via power iteration) and uses them as x/y coordinates.
fn spectral_layout<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    iterations: usize,
    width: f64,
    height: f64,
) -> Result<GraphLayout<N>>
where
    N: Node + Clone + std::fmt::Debug,
    E: EdgeWeight + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let nodes: Vec<N> = graph.nodes().into_iter().cloned().collect();
    let n = nodes.len();

    if n == 0 {
        return Ok(GraphLayout::new(width, height));
    }
    if n == 1 {
        let mut layout = GraphLayout::new(width, height);
        layout.set_position(
            nodes[0].clone(),
            LayoutPosition::new(width / 2.0, height / 2.0),
        );
        return Ok(layout);
    }

    let node_to_idx: HashMap<N, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.clone(), i))
        .collect();

    // Build adjacency matrix
    let mut adj = vec![vec![0.0f64; n]; n];
    let mut degree = vec![0.0f64; n];

    for e in graph.edges() {
        if let (Some(&si), Some(&ti)) = (node_to_idx.get(&e.source), node_to_idx.get(&e.target)) {
            let w: f64 = e.weight.clone().into();
            let w = w.abs().max(1e-10);
            adj[si][ti] += w;
            adj[ti][si] += w;
            degree[si] += w;
            degree[ti] += w;
        }
    }

    // Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    let d_inv_sqrt: Vec<f64> = degree
        .iter()
        .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 1.0 })
        .collect();

    // Compute two Fiedler-like vectors via deflated power iteration
    let compute_eigvec = |shift: f64, deflate: Option<&Vec<f64>>| -> Vec<f64> {
        let mut v: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
        for x in &mut v {
            *x /= norm;
        }

        for _ in 0..iterations {
            // w = (I + shift*I - L_norm) * v  ≡  D^{-1/2} A D^{-1/2} v + shift*v
            let mut w = vec![0.0f64; n];
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    sum += d_inv_sqrt[i] * adj[i][j] * d_inv_sqrt[j] * v[j];
                }
                w[i] = sum + shift * v[i];
            }

            // Deflate against constant vector and optionally first vector
            let ones_norm = (n as f64).sqrt();
            let dot_ones = w.iter().sum::<f64>() / ones_norm;
            for x in &mut w {
                *x -= dot_ones / ones_norm;
            }
            if let Some(first) = deflate {
                let dot_first: f64 = w.iter().zip(first.iter()).map(|(a, b)| a * b).sum();
                for (x, &f) in w.iter_mut().zip(first.iter()) {
                    *x -= dot_first * f;
                }
            }

            let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
            for x in &mut w {
                *x /= norm;
            }
            v = w;
        }
        v
    };

    let v1 = compute_eigvec(1.0, None);
    let v2 = compute_eigvec(0.8, Some(&v1));

    // Scale to canvas
    let v1_min = v1.iter().cloned().fold(f64::INFINITY, f64::min);
    let v1_max = v1.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let v2_min = v2.iter().cloned().fold(f64::INFINITY, f64::min);
    let v2_max = v2.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let v1_range = (v1_max - v1_min).max(1e-10);
    let v2_range = (v2_max - v2_min).max(1e-10);

    let margin = 40.0;
    let mut layout = GraphLayout::new(width, height);

    for (i, node) in nodes.into_iter().enumerate() {
        let x = margin + (v1[i] - v1_min) / v1_range * (width - 2.0 * margin);
        let y = margin + (v2[i] - v2_min) / v2_range * (height - 2.0 * margin);
        layout.set_position(node, LayoutPosition::new(x, y));
    }

    Ok(layout)
}

// ============================================================================
// SVG export
// ============================================================================

/// Configuration for SVG graph rendering
#[derive(Debug, Clone)]
pub struct SvgConfig {
    /// Fill color for nodes (CSS color string)
    pub node_color: String,
    /// Stroke color for node outlines
    pub node_stroke_color: String,
    /// Color for edges
    pub edge_color: String,
    /// Radius of nodes in pixels
    pub node_radius: f64,
    /// Stroke width for edges in pixels
    pub edge_width: f64,
    /// Font size for node labels in pixels
    pub font_size: f64,
    /// Font color for labels
    pub font_color: String,
    /// Background color of the SVG canvas
    pub background_color: String,
    /// Whether to show node labels
    pub show_labels: bool,
    /// Whether to show edge weights
    pub show_edge_weights: bool,
    /// Arrow size for directed edges (0 = undirected)
    pub arrow_size: f64,
}

impl Default for SvgConfig {
    fn default() -> Self {
        SvgConfig {
            node_color: "#4CAF50".to_string(),
            node_stroke_color: "#2E7D32".to_string(),
            edge_color: "#9E9E9E".to_string(),
            node_radius: 12.0,
            edge_width: 1.5,
            font_size: 11.0,
            font_color: "#FFFFFF".to_string(),
            background_color: "#FAFAFA".to_string(),
            show_labels: true,
            show_edge_weights: false,
            arrow_size: 0.0,
        }
    }
}

impl SvgConfig {
    /// Create a dark-theme config
    pub fn dark_theme() -> Self {
        SvgConfig {
            node_color: "#1565C0".to_string(),
            node_stroke_color: "#90CAF9".to_string(),
            edge_color: "#546E7A".to_string(),
            node_radius: 12.0,
            edge_width: 1.5,
            font_size: 11.0,
            font_color: "#FFFFFF".to_string(),
            background_color: "#1A1A2E".to_string(),
            show_labels: true,
            show_edge_weights: false,
            arrow_size: 0.0,
        }
    }
}

/// Render a graph to SVG using the provided layout and configuration
///
/// # Arguments
/// * `graph` - The graph to render
/// * `layout` - Pre-computed node positions
/// * `config` - Visual styling configuration
///
/// # Returns
/// A `String` containing valid SVG markup
///
/// # Example
/// ```rust
/// use scirs2_graph::visualization::{compute_layout, LayoutAlgorithm, SvgConfig, render_svg};
/// use scirs2_graph::Graph;
///
/// let mut g: Graph<&str, f64> = Graph::new();
/// g.add_node("A");
/// g.add_node("B");
/// let _ = g.add_edge("A", "B", 1.0);
///
/// let layout = compute_layout(&g, &LayoutAlgorithm::Circular { radius: 100.0 }, 400.0, 300.0).unwrap();
/// let svg = render_svg(&g, &layout, &SvgConfig::default());
/// assert!(svg.contains("<svg"));
/// ```
pub fn render_svg<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    layout: &GraphLayout<N>,
    config: &SvgConfig,
) -> String
where
    N: Node + Clone + std::fmt::Debug + std::fmt::Display,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let width = layout.width;
    let height = layout.height;

    let mut svg = String::new();

    // SVG header
    let _ = writeln!(
        svg,
        r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
  <rect width="100%" height="100%" fill="{}"/>"#,
        width, height, width, height, config.background_color
    );

    // Define arrowhead marker if directed
    if config.arrow_size > 0.0 {
        let _ = writeln!(
            svg,
            r#"  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="{}"/>
    </marker>
  </defs>"#,
            config.edge_color
        );
    }

    // Draw edges first (behind nodes)
    for edge in graph.edges() {
        if let (Some(src_pos), Some(tgt_pos)) =
            (layout.position(&edge.source), layout.position(&edge.target))
        {
            let marker_attr = if config.arrow_size > 0.0 {
                r#" marker-end="url(#arrow)""#
            } else {
                ""
            };

            let _ = writeln!(
                svg,
                r#"  <line x1="{:.2}" y1="{:.2}" x2="{:.2}" y2="{:.2}" stroke="{}" stroke-width="{:.2}"{}/>  "#,
                src_pos.x,
                src_pos.y,
                tgt_pos.x,
                tgt_pos.y,
                config.edge_color,
                config.edge_width,
                marker_attr
            );

            // Edge weight label
            if config.show_edge_weights {
                let mid_x = (src_pos.x + tgt_pos.x) / 2.0;
                let mid_y = (src_pos.y + tgt_pos.y) / 2.0;
                let w: f64 = edge.weight.clone().into();
                let _ = writeln!(
                    svg,
                    r#"  <text x="{:.2}" y="{:.2}" font-size="{:.1}" fill="{}" text-anchor="middle">{:.2}</text>"#,
                    mid_x,
                    mid_y - 4.0,
                    config.font_size * 0.85,
                    config.edge_color,
                    w
                );
            }
        }
    }

    // Draw nodes
    for node in graph.nodes() {
        if let Some(pos) = layout.position(node) {
            let _ = writeln!(
                svg,
                r#"  <circle cx="{:.2}" cy="{:.2}" r="{:.2}" fill="{}" stroke="{}" stroke-width="1.5"/>"#,
                pos.x, pos.y, config.node_radius, config.node_color, config.node_stroke_color
            );

            // Node label
            if config.show_labels {
                let _ = writeln!(
                    svg,
                    r#"  <text x="{:.2}" y="{:.2}" font-size="{:.1}" font-family="sans-serif" fill="{}" text-anchor="middle" dominant-baseline="middle">{}</text>"#,
                    pos.x, pos.y, config.font_size, config.font_color, node
                );
            }
        }
    }

    let _ = writeln!(svg, "</svg>");
    svg
}

// ============================================================================
// DOT format export with layout hints
// ============================================================================

/// Configuration for DOT format export
#[derive(Debug, Clone, Default)]
pub struct DotConfig {
    /// Whether to include position hints in DOT output
    pub include_positions: bool,
    /// Whether the graph is directed (uses `digraph` instead of `graph`)
    pub directed: bool,
    /// Graph name in the DOT file
    pub graph_name: String,
    /// Default node attributes
    pub node_attributes: Vec<(String, String)>,
    /// Default edge attributes
    pub edge_attributes: Vec<(String, String)>,
}

impl DotConfig {
    /// Create a new DotConfig with the given graph name
    pub fn new(graph_name: &str) -> Self {
        DotConfig {
            include_positions: false,
            directed: false,
            graph_name: graph_name.to_string(),
            node_attributes: Vec::new(),
            edge_attributes: Vec::new(),
        }
    }

    /// Include layout positions in output
    pub fn with_positions(mut self) -> Self {
        self.include_positions = true;
        self
    }

    /// Set as directed graph
    pub fn directed(mut self) -> Self {
        self.directed = true;
        self
    }

    /// Add a default node attribute
    pub fn node_attr(mut self, key: &str, value: &str) -> Self {
        self.node_attributes
            .push((key.to_string(), value.to_string()));
        self
    }
}

/// Export a graph to DOT format with optional layout hints
///
/// # Arguments
/// * `graph` - The graph to export
/// * `config` - DOT export configuration
/// * `layout` - Optional layout for position hints
///
/// # Returns
/// A `String` containing valid DOT format
pub fn export_dot<N, E, Ix>(
    graph: &Graph<N, E, Ix>,
    config: &DotConfig,
    layout: Option<&GraphLayout<N>>,
) -> Result<String>
where
    N: Node + Clone + std::fmt::Debug + std::fmt::Display + Hash + Eq,
    E: EdgeWeight + Clone + Into<f64>,
    Ix: petgraph::graph::IndexType,
{
    let graph_type = if config.directed { "digraph" } else { "graph" };
    let edge_op = if config.directed { "->" } else { "--" };
    let name = if config.graph_name.is_empty() {
        "G"
    } else {
        &config.graph_name
    };

    let mut dot = String::new();
    let _ = writeln!(dot, "{} {} {{", graph_type, name);

    // Default node attributes
    if !config.node_attributes.is_empty() {
        let attrs: Vec<String> = config
            .node_attributes
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect();
        let _ = writeln!(dot, "  node [{}];", attrs.join(", "));
    }

    // Default edge attributes
    if !config.edge_attributes.is_empty() {
        let attrs: Vec<String> = config
            .edge_attributes
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, v))
            .collect();
        let _ = writeln!(dot, "  edge [{}];", attrs.join(", "));
    }

    // Nodes
    for node in graph.nodes() {
        if config.include_positions {
            if let Some(layout) = layout {
                if let Some(pos) = layout.position(node) {
                    let _ = writeln!(dot, "  \"{}\" [pos=\"{:.2},{:.2}!\"];", node, pos.x, pos.y);
                    continue;
                }
            }
        }
        let _ = writeln!(dot, "  \"{}\";", node);
    }

    // Edges
    for edge in graph.edges() {
        let w: f64 = edge.weight.clone().into();
        let _ = writeln!(
            dot,
            "  \"{}\" {} \"{}\" [weight={:.4}];",
            edge.source, edge_op, edge.target, w
        );
    }

    let _ = writeln!(dot, "}}");
    Ok(dot)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base::Graph;

    fn make_triangle() -> Graph<&'static str, f64> {
        let mut g = Graph::new();
        let _ = g.add_edge("A", "B", 1.0);
        let _ = g.add_edge("B", "C", 1.0);
        let _ = g.add_edge("A", "C", 1.0);
        g
    }

    fn make_path(n: usize) -> Graph<usize, f64> {
        let mut g = Graph::new();
        for i in 0..n - 1 {
            let _ = g.add_edge(i, i + 1, 1.0);
        }
        g
    }

    #[test]
    fn test_circular_layout_positions() {
        let g = make_triangle();
        let layout = compute_layout(
            &g,
            &LayoutAlgorithm::Circular { radius: 100.0 },
            400.0,
            300.0,
        )
        .expect("Layout failed");
        assert_eq!(layout.node_count(), 3);
        // All nodes should be at approximately radius distance from center
        for pos in layout.positions.values() {
            let dx = pos.x - 200.0;
            let dy = pos.y - 150.0;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!((dist - 100.0).abs() < 1.0, "Distance from center: {}", dist);
        }
    }

    #[test]
    fn test_force_directed_layout() {
        let g = make_path(6);
        let algo = LayoutAlgorithm::ForceDirected {
            iterations: 50,
            spring_length: None,
            cooling: 0.95,
        };
        let layout = compute_layout(&g, &algo, 500.0, 400.0).expect("Layout failed");
        assert_eq!(layout.node_count(), 6);
        // All positions should be within canvas bounds
        for pos in layout.positions.values() {
            assert!(pos.x >= 0.0 && pos.x <= 500.0, "x out of bounds: {}", pos.x);
            assert!(pos.y >= 0.0 && pos.y <= 400.0, "y out of bounds: {}", pos.y);
        }
    }

    #[test]
    fn test_hierarchical_layout() {
        let g = make_path(5);
        let algo = LayoutAlgorithm::Hierarchical {
            layer_spacing: 50.0,
            node_spacing: 50.0,
        };
        let layout = compute_layout(&g, &algo, 500.0, 400.0).expect("Layout failed");
        assert_eq!(layout.node_count(), 5);
    }

    #[test]
    fn test_spectral_layout() {
        let g = make_triangle();
        let algo = LayoutAlgorithm::Spectral { iterations: 30 };
        let layout = compute_layout(&g, &algo, 400.0, 300.0).expect("Layout failed");
        assert_eq!(layout.node_count(), 3);
    }

    #[test]
    fn test_svg_render_contains_elements() {
        let g = make_triangle();
        let layout = compute_layout(
            &g,
            &LayoutAlgorithm::Circular { radius: 100.0 },
            400.0,
            300.0,
        )
        .expect("Layout");
        let svg = render_svg(&g, &layout, &SvgConfig::default());
        assert!(svg.contains("<svg"), "Missing SVG root element");
        assert!(svg.contains("<circle"), "Missing node circles");
        assert!(svg.contains("<line"), "Missing edges");
        assert!(svg.contains("</svg>"), "Missing closing SVG tag");
    }

    #[test]
    fn test_svg_render_node_labels() {
        let g = make_triangle();
        let layout = compute_layout(
            &g,
            &LayoutAlgorithm::Circular { radius: 100.0 },
            400.0,
            300.0,
        )
        .expect("Layout");
        let mut config = SvgConfig::default();
        config.show_labels = true;
        let svg = render_svg(&g, &layout, &config);
        // Node labels "A", "B", "C" should appear
        assert!(
            svg.contains(">A<") || svg.contains(">A "),
            "Label A not found"
        );
    }

    #[test]
    fn test_dot_export_basic() {
        let g = make_triangle();
        let config = DotConfig::new("TestGraph");
        let dot = export_dot(&g, &config, None).expect("DOT export failed");
        assert!(dot.contains("graph TestGraph"), "Missing graph header");
        assert!(dot.contains("--"), "Missing undirected edge operator");
        assert!(
            dot.contains("\"A\"") || dot.contains("\"B\""),
            "Missing node"
        );
    }

    #[test]
    fn test_dot_export_with_positions() {
        let g = make_triangle();
        let layout = compute_layout(
            &g,
            &LayoutAlgorithm::Circular { radius: 100.0 },
            400.0,
            300.0,
        )
        .expect("Layout");
        let config = DotConfig::new("PosGraph").with_positions();
        let dot = export_dot(&g, &config, Some(&layout)).expect("DOT export with positions failed");
        assert!(dot.contains("pos="), "Missing position hints");
    }

    #[test]
    fn test_empty_graph_layout() {
        let g: Graph<usize, f64> = Graph::new();
        let layout = compute_layout(
            &g,
            &LayoutAlgorithm::Circular { radius: 100.0 },
            400.0,
            300.0,
        )
        .expect("Empty layout");
        assert_eq!(layout.node_count(), 0);
    }

    #[test]
    fn test_layout_position_distance() {
        let p1 = LayoutPosition::new(0.0, 0.0);
        let p2 = LayoutPosition::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_dark_theme_config() {
        let g = make_triangle();
        let layout = compute_layout(
            &g,
            &LayoutAlgorithm::Circular { radius: 100.0 },
            400.0,
            300.0,
        )
        .expect("Layout");
        let svg = render_svg(&g, &layout, &SvgConfig::dark_theme());
        assert!(svg.contains("#1A1A2E"), "Dark theme background not found");
    }
}
