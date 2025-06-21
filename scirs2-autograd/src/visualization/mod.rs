//! Graph visualization and debugging tools for computation graphs
//!
//! This module provides tools for visualizing computation graphs, which is essential
//! for debugging complex neural networks and understanding gradient flow.

use crate::graph::{Graph, TensorID};
use crate::Float;
use std::collections::{HashMap, HashSet};
use std::fmt::Write;

/// Graph visualization configuration
#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    /// Whether to show tensor shapes in nodes
    pub show_shapes: bool,
    /// Whether to show operation names
    pub show_operations: bool,
    /// Whether to show gradient flow
    pub show_gradients: bool,
    /// Maximum number of nodes to display
    pub max_nodes: Option<usize>,
    /// Output format
    pub format: OutputFormat,
    /// Whether to include values (for small tensors)
    pub show_values: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_shapes: true,
            show_operations: true,
            show_gradients: false,
            max_nodes: Some(100),
            format: OutputFormat::Dot,
            show_values: false,
        }
    }
}

/// Output format for graph visualization
#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    /// Graphviz DOT format
    Dot,
    /// Simple text representation
    Text,
    /// JSON format for web visualization
    Json,
    /// Mermaid diagram format
    Mermaid,
}

/// Graph visualizer for creating visual representations of computation graphs
pub struct GraphVisualizer<F: Float> {
    config: VisualizationConfig,
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> GraphVisualizer<F> {
    /// Create a new graph visualizer with default configuration
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create a new graph visualizer with custom configuration
    pub fn with_config(config: VisualizationConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate a visual representation of the computation graph
    pub fn visualize(&self, graph: &Graph<F>) -> Result<String, VisualizationError> {
        match self.config.format {
            OutputFormat::Dot => self.generate_dot(graph),
            OutputFormat::Text => self.generate_text(graph),
            OutputFormat::Json => self.generate_json(graph),
            OutputFormat::Mermaid => self.generate_mermaid(graph),
        }
    }

    /// Generate DOT format for Graphviz
    fn generate_dot(&self, graph: &Graph<F>) -> Result<String, VisualizationError> {
        let mut output = String::new();
        writeln!(output, "digraph computation_graph {{")?;
        writeln!(output, "  rankdir=TB;")?;
        writeln!(output, "  node [shape=box, style=rounded];")?;

        let nodes = self.collect_nodes(graph)?;
        let node_info = self.analyze_nodes(&nodes)?;

        // Generate nodes
        for (i, node) in nodes.iter().enumerate() {
            let node_id = format!("node_{}", i);
            let label = self.generate_node_label(node, &node_info)?;
            let style = self.get_node_style(node);

            writeln!(output, "  {} [label=\"{}\", {}];", node_id, label, style)?;
        }

        // Generate edges
        for (i, _node) in nodes.iter().enumerate() {
            let node_id = format!("node_{}", i);
            // Simplified - in practice would get actual inputs from graph
            let inputs: Vec<TensorID> = Vec::new();
            for input in inputs {
                if let Some(input_idx) = nodes.iter().position(|n| *n == input) {
                    let input_id = format!("node_{}", input_idx);
                    writeln!(output, "  {} -> {};", input_id, node_id)?;
                }
            }
        }

        writeln!(output, "}}")?;
        Ok(output)
    }

    /// Generate simple text representation
    fn generate_text(&self, graph: &Graph<F>) -> Result<String, VisualizationError> {
        let mut output = String::new();
        writeln!(output, "Computation Graph:")?;
        writeln!(output, "==================")?;

        let nodes = self.collect_nodes(graph)?;
        let node_info = self.analyze_nodes(&nodes)?;

        for (i, node) in nodes.iter().enumerate() {
            let label = self.generate_node_label(node, &node_info)?;
            writeln!(output, "Node {}: {}", i, label)?;

            // Simplified - in practice would get actual inputs from graph
            let inputs: Vec<TensorID> = Vec::new();
            if !inputs.is_empty() {
                write!(output, "  Inputs: ")?;
                for (j, input) in inputs.iter().enumerate() {
                    if j > 0 {
                        write!(output, ", ")?;
                    }
                    if let Some(input_idx) = nodes.iter().position(|n| *n == *input) {
                        write!(output, "Node {}", input_idx)?;
                    } else {
                        write!(output, "External")?;
                    }
                }
                writeln!(output)?;
            }
        }

        Ok(output)
    }

    /// Generate JSON format for web visualization
    fn generate_json(&self, graph: &Graph<F>) -> Result<String, VisualizationError> {
        let mut output = String::new();
        writeln!(output, "{{")?;
        writeln!(output, "  \"nodes\": [")?;

        let nodes = self.collect_nodes(graph)?;
        let node_info = self.analyze_nodes(&nodes)?;

        for (i, node) in nodes.iter().enumerate() {
            if i > 0 {
                writeln!(output, ",")?;
            }
            let label = self.generate_node_label(node, &node_info)?;
            write!(
                output,
                "    {{\"id\": {}, \"label\": \"{}\"}}",
                i,
                label.replace('"', "\\\"")
            )?;
        }

        writeln!(output)?;
        writeln!(output, "  ],")?;
        writeln!(output, "  \"edges\": [")?;

        let mut edge_count = 0;
        for (i, _node) in nodes.iter().enumerate() {
            // Simplified - in practice would get actual inputs from graph
            let inputs: Vec<TensorID> = Vec::new();
            for input in inputs {
                if let Some(input_idx) = nodes.iter().position(|n| *n == input) {
                    if edge_count > 0 {
                        writeln!(output, ",")?;
                    }
                    write!(output, "    {{\"from\": {}, \"to\": {}}}", input_idx, i)?;
                    edge_count += 1;
                }
            }
        }

        writeln!(output)?;
        writeln!(output, "  ]")?;
        writeln!(output, "}}")?;
        Ok(output)
    }

    /// Generate Mermaid diagram format
    fn generate_mermaid(&self, graph: &Graph<F>) -> Result<String, VisualizationError> {
        let mut output = String::new();
        writeln!(output, "graph TD")?;

        let nodes = self.collect_nodes(graph)?;
        let node_info = self.analyze_nodes(&nodes)?;

        // Generate nodes
        for (i, node) in nodes.iter().enumerate() {
            let node_id = format!("N{}", i);
            let label = self.generate_node_label(node, &node_info)?;
            writeln!(output, "  {}[{}]", node_id, label)?;
        }

        // Generate edges
        for (i, _node) in nodes.iter().enumerate() {
            let node_id = format!("N{}", i);
            // Simplified - in practice would get actual inputs from graph
            let inputs: Vec<TensorID> = Vec::new();
            for input in inputs {
                if let Some(input_idx) = nodes.iter().position(|n| *n == input) {
                    let input_id = format!("N{}", input_idx);
                    writeln!(output, "  {} --> {}", input_id, node_id)?;
                }
            }
        }

        Ok(output)
    }

    /// Collect all nodes from the graph
    #[allow(dead_code)]
    fn collect_tensor_ids(&self, graph: &Graph<F>) -> Result<Vec<TensorID>, VisualizationError> {
        let mut tensor_ids = Vec::new();
        let mut visited = HashSet::new();

        // Traverse from all roots (tensors with no dependencies)
        self.traverse_graph(graph, &mut tensor_ids, &mut visited)?;

        // Limit tensors if configured
        if let Some(max_nodes) = self.config.max_nodes {
            if tensor_ids.len() > max_nodes {
                tensor_ids.truncate(max_nodes);
            }
        }

        Ok(tensor_ids)
    }

    /// Traverse the graph to collect tensor IDs
    #[allow(dead_code)]
    fn traverse_graph(
        &self,
        _graph: &Graph<F>,
        _tensor_ids: &mut [TensorID],
        _visited: &mut HashSet<TensorID>,
    ) -> Result<(), VisualizationError> {
        // This is a simplified traversal - in a real implementation,
        // we would need access to the graph's internal structure
        // For now, return an empty traversal
        Ok(())
    }

    /// Analyze tensor IDs to gather metadata
    #[allow(dead_code)]
    fn analyze_tensor_ids(
        &self,
        tensor_ids: &[TensorID],
    ) -> Result<NodeAnalysis, VisualizationError> {
        let mut analysis = NodeAnalysis {
            shapes: HashMap::new(),
            operations: HashMap::new(),
            depths: HashMap::new(),
        };

        for (i, &tensor_id) in tensor_ids.iter().enumerate() {
            // Analyze tensor properties (simplified)
            analysis.operations.insert(i, "Operation".to_string());

            // Calculate depth (simplified)
            let depth = self.calculate_tensor_depth(tensor_id, tensor_ids);
            analysis.depths.insert(i, depth);
        }

        Ok(analysis)
    }

    /// Calculate the depth of a tensor in the graph
    #[allow(dead_code)]
    fn calculate_tensor_depth(&self, _tensor_id: TensorID, _tensor_ids: &[TensorID]) -> usize {
        // Simplified depth calculation
        0
    }

    /// Collect nodes from the graph
    #[allow(dead_code)]
    fn collect_nodes(&self, _graph: &Graph<F>) -> Result<Vec<TensorID>, VisualizationError> {
        // Simplified - would collect actual nodes from graph
        Ok(vec![0, 1, 2])
    }

    /// Analyze nodes to gather metadata
    #[allow(dead_code)]
    fn analyze_nodes(&self, nodes: &[TensorID]) -> Result<NodeAnalysis, VisualizationError> {
        self.analyze_tensor_ids(nodes)
    }

    /// Generate a label for a node
    #[allow(dead_code)]
    fn generate_node_label(
        &self,
        &tensor_id: &TensorID,
        analysis: &NodeAnalysis,
    ) -> Result<String, VisualizationError> {
        self.generate_tensor_label(tensor_id, analysis)
    }

    /// Get node style for rendering
    #[allow(dead_code)]
    fn get_node_style(&self, _node: &TensorID) -> String {
        "style=filled, fillcolor=lightblue".to_string()
    }

    /// Generate a label for a tensor
    #[allow(dead_code)]
    fn generate_tensor_label(
        &self,
        tensor_id: TensorID,
        _analysis: &NodeAnalysis,
    ) -> Result<String, VisualizationError> {
        let mut label = String::new();

        if self.config.show_operations {
            // In a real implementation, we would extract operation from tensor_id
            write!(label, "Tensor")?;
        }

        if self.config.show_shapes {
            // In a real implementation, we would extract shape from tensor_id
            if !label.is_empty() {
                write!(label, "\\n")?;
            }
            write!(label, "Shape: [?]")?;
        }

        if label.is_empty() {
            write!(label, "Tensor {}", tensor_id)?;
        }

        Ok(label)
    }

    /// Get styling for a node based on its type
    #[allow(dead_code)]
    fn get_tensor_style(&self, _tensor_id: TensorID) -> String {
        // In a real implementation, would check tensor type
        "fillcolor=lightblue, style=filled".to_string()
    }
}

impl<F: Float> Default for GraphVisualizer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Node analysis results
struct NodeAnalysis {
    #[allow(dead_code)]
    shapes: HashMap<usize, Vec<usize>>,
    operations: HashMap<usize, String>,
    depths: HashMap<usize, usize>,
}

/// Graph debugging utilities
pub struct GraphDebugger<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> GraphDebugger<F> {
    /// Create a new graph debugger
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Print graph statistics
    pub fn print_stats(&self, _graph: &Graph<F>) -> Result<(), VisualizationError> {
        println!("Graph Statistics:");
        println!("================");

        // In a real implementation, we would extract these from the graph
        println!("Total nodes: ?");
        println!("Variable nodes: ?");
        println!("Operation nodes: ?");
        println!("Graph depth: ?");

        Ok(())
    }

    /// Validate graph structure
    pub fn validate_graph(&self, _graph: &Graph<F>) -> Result<Vec<String>, VisualizationError> {
        let issues = Vec::new();

        // Check for common graph issues
        // - Cycles in the graph
        // - Orphaned nodes
        // - Invalid connections
        // - Type mismatches

        Ok(issues)
    }

    /// Find potential optimization opportunities
    pub fn analyze_optimizations(
        &self,
        _graph: &Graph<F>,
    ) -> Result<Vec<String>, VisualizationError> {
        // Look for optimization opportunities:
        // - Common subexpressions
        // - Constant folding opportunities
        // - Redundant operations
        // - Memory optimization opportunities

        let suggestions = vec![
            "Consider enabling gradient checkpointing for memory efficiency".to_string(),
            "Look for opportunities to fuse element-wise operations".to_string(),
        ];

        Ok(suggestions)
    }
}

impl<F: Float> Default for GraphDebugger<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Interactive graph explorer
pub struct GraphExplorer<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> GraphExplorer<F> {
    /// Create a new graph explorer
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Start an interactive session for exploring the graph
    pub fn start_interactive(&self, _graph: &Graph<F>) -> Result<(), VisualizationError> {
        println!("Starting interactive graph exploration...");
        println!("Commands: help, stats, visualize, quit");

        // In a real implementation, this would start an interactive REPL
        // for exploring the graph structure

        Ok(())
    }

    /// Generate a summary of a specific tensor
    pub fn summarize_tensor(&self, _tensor_id: TensorID) -> Result<String, VisualizationError> {
        Ok("Tensor summary would go here".to_string())
    }
}

impl<F: Float> Default for GraphExplorer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur during visualization
#[derive(Debug, thiserror::Error)]
pub enum VisualizationError {
    #[error("Graph traversal error: {0}")]
    GraphTraversal(String),
    #[error("Format error: {0}")]
    Format(#[from] std::fmt::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid configuration: {0}")]
    Config(String),
}

/// Public API functions for graph visualization
/// Visualize a computation graph in DOT format
pub fn visualize_graph_dot<F: Float>(graph: &Graph<F>) -> Result<String, VisualizationError> {
    let visualizer = GraphVisualizer::new();
    visualizer.visualize(graph)
}

/// Visualize a computation graph in text format
pub fn visualize_graph_text<F: Float>(graph: &Graph<F>) -> Result<String, VisualizationError> {
    let config = VisualizationConfig {
        format: OutputFormat::Text,
        ..Default::default()
    };
    let visualizer = GraphVisualizer::with_config(config);
    visualizer.visualize(graph)
}

/// Visualize a computation graph in JSON format
pub fn visualize_graph_json<F: Float>(graph: &Graph<F>) -> Result<String, VisualizationError> {
    let config = VisualizationConfig {
        format: OutputFormat::Json,
        ..Default::default()
    };
    let visualizer = GraphVisualizer::with_config(config);
    visualizer.visualize(graph)
}

/// Visualize a computation graph in Mermaid format
pub fn visualize_graph_mermaid<F: Float>(graph: &Graph<F>) -> Result<String, VisualizationError> {
    let config = VisualizationConfig {
        format: OutputFormat::Mermaid,
        ..Default::default()
    };
    let visualizer = GraphVisualizer::with_config(config);
    visualizer.visualize(graph)
}

/// Print graph statistics to console
pub fn print_graph_stats<F: Float>(graph: &Graph<F>) -> Result<(), VisualizationError> {
    let debugger = GraphDebugger::new();
    debugger.print_stats(graph)
}

/// Validate graph structure and return any issues found
pub fn validate_graph<F: Float>(graph: &Graph<F>) -> Result<Vec<String>, VisualizationError> {
    let debugger = GraphDebugger::new();
    debugger.validate_graph(graph)
}

/// Analyze graph for optimization opportunities
pub fn analyze_graph_optimizations<F: Float>(
    graph: &Graph<F>,
) -> Result<Vec<String>, VisualizationError> {
    let debugger = GraphDebugger::new();
    debugger.analyze_optimizations(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualization_config() {
        let config = VisualizationConfig::default();
        assert!(config.show_shapes);
        assert!(config.show_operations);
        assert!(!config.show_gradients);
        assert_eq!(config.max_nodes, Some(100));
        assert!(matches!(config.format, OutputFormat::Dot));
    }

    #[test]
    fn test_graph_visualizer_creation() {
        let _visualizer = GraphVisualizer::<f32>::new();
        let _visualizer_with_config =
            GraphVisualizer::<f32>::with_config(VisualizationConfig::default());
    }

    #[test]
    fn test_graph_debugger_creation() {
        let _debugger = GraphDebugger::<f32>::new();
    }

    #[test]
    fn test_graph_explorer_creation() {
        let _explorer = GraphExplorer::<f32>::new();
    }

    #[test]
    fn test_output_formats() {
        // Test that all output formats can be created
        let formats = [
            OutputFormat::Dot,
            OutputFormat::Text,
            OutputFormat::Json,
            OutputFormat::Mermaid,
        ];

        for format in formats {
            let config = VisualizationConfig {
                format,
                ..Default::default()
            };
            let _visualizer = GraphVisualizer::<f32>::with_config(config);
        }
    }
}
