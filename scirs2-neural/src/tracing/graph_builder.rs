//! Static computation graph builder.
//!
//! Provides a symbolic tracing API that records neural network operations as
//! they are described (not executed), constructing a `StaticGraph` DAG.
//!
//! ## Usage
//!
//! ```rust
//! use scirs2_neural::tracing::graph_builder::GraphBuilder;
//! use scirs2_neural::tracing::types::{TensorSpec, DType};
//!
//! let mut builder = GraphBuilder::new();
//! let spec = TensorSpec::new(vec![1, 8], DType::F64);
//! let input = builder.input(spec);
//! let hidden = builder.linear(input, 8, 4);
//! let activated = builder.relu(hidden);
//! let output = builder.linear(activated, 4, 2);
//! let graph = builder.build(vec![output]);
//! assert_eq!(graph.num_nodes(), 4); // input + 2 linear + 1 relu
//! ```

use crate::error::{Error, Result};
use crate::tracing::types::{DType, OpAttr, OpNode, OpType, StaticGraph, TensorSpec};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Tensor handle
// ---------------------------------------------------------------------------

/// A lightweight handle into the graph being built.
///
/// Each `Tensor` uniquely identifies a node's output by the node's ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Tensor {
    /// Node ID that produces this tensor
    pub id: usize,
    /// Cached output spec index (always 0 for single-output ops)
    pub(crate) spec_idx: usize,
}

impl Tensor {
    fn new(id: usize) -> Self {
        Self { id, spec_idx: 0 }
    }
}

// ---------------------------------------------------------------------------
// GraphBuilder
// ---------------------------------------------------------------------------

/// Records operations and builds a `StaticGraph`.
#[derive(Debug, Default)]
pub struct GraphBuilder {
    nodes: Vec<OpNode>,
    next_id: usize,
    /// Reverse-edge map: node_id → list of consumer node IDs
    consumers: HashMap<usize, Vec<usize>>,
}

impl GraphBuilder {
    /// Create a new, empty builder.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
            consumers: HashMap::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Allocate the next node ID.
    fn alloc_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Register a new node and update consumer lists.
    fn add_node(&mut self, node: OpNode) -> Tensor {
        let id = node.id;
        // Update consumer lists for all inputs
        for &inp_id in &node.inputs {
            self.consumers.entry(inp_id).or_default().push(id);
        }
        self.nodes.push(node);
        Tensor::new(id)
    }

    /// Find the spec of a tensor (output spec of its producing node).
    fn spec_of(&self, t: Tensor) -> Result<&TensorSpec> {
        self.nodes
            .iter()
            .find(|n| n.id == t.id)
            .map(|n| &n.output_spec)
            .ok_or_else(|| Error::InvalidArgument(format!("Unknown tensor id {}", t.id)))
    }

    // -----------------------------------------------------------------------
    // Input declaration
    // -----------------------------------------------------------------------

    /// Declare a graph input with the given spec.
    ///
    /// Adds a `Constant` placeholder node. Returns a handle to it.
    pub fn input(&mut self, spec: TensorSpec) -> Tensor {
        let id = self.alloc_id();
        let node = OpNode::new(id, OpType::Constant, vec![], HashMap::new(), spec)
            .with_name(format!("input_{id}"));
        self.add_node(node)
    }

    // -----------------------------------------------------------------------
    // Layer builders
    // -----------------------------------------------------------------------

    /// Add a fully-connected (linear) layer.
    ///
    /// `in_features` and `out_features` are recorded as attributes.
    pub fn linear(&mut self, input: Tensor, in_features: usize, out_features: usize) -> Tensor {
        let id = self.alloc_id();
        // Derive output shape: same batch dims, last dim = out_features
        let input_shape = self
            .spec_of(input)
            .map(|s| s.shape.clone())
            .unwrap_or_else(|_| vec![1, in_features]);
        let mut out_shape = input_shape.clone();
        if let Some(last) = out_shape.last_mut() {
            *last = out_features;
        }

        let mut attrs = HashMap::new();
        attrs.insert("in_features".into(), OpAttr::Int(in_features as i64));
        attrs.insert("out_features".into(), OpAttr::Int(out_features as i64));

        let spec = TensorSpec::new(out_shape, DType::F64);
        let node = OpNode::new(id, OpType::Linear, vec![input.id], attrs, spec)
            .with_name(format!("linear_{id}"));
        self.add_node(node)
    }

    /// Add a ReLU activation.
    pub fn relu(&mut self, input: Tensor) -> Tensor {
        let id = self.alloc_id();
        let spec = self
            .spec_of(input)
            .cloned()
            .unwrap_or_else(|_| TensorSpec::new(vec![], DType::F64));
        let node = OpNode::new(id, OpType::ReLU, vec![input.id], HashMap::new(), spec)
            .with_name(format!("relu_{id}"));
        self.add_node(node)
    }

    /// Add a Sigmoid activation.
    pub fn sigmoid(&mut self, input: Tensor) -> Tensor {
        let id = self.alloc_id();
        let spec = self
            .spec_of(input)
            .cloned()
            .unwrap_or_else(|_| TensorSpec::new(vec![], DType::F64));
        let node = OpNode::new(id, OpType::Sigmoid, vec![input.id], HashMap::new(), spec)
            .with_name(format!("sigmoid_{id}"));
        self.add_node(node)
    }

    /// Add a Tanh activation.
    pub fn tanh(&mut self, input: Tensor) -> Tensor {
        let id = self.alloc_id();
        let spec = self
            .spec_of(input)
            .cloned()
            .unwrap_or_else(|_| TensorSpec::new(vec![], DType::F64));
        let node = OpNode::new(id, OpType::Tanh, vec![input.id], HashMap::new(), spec)
            .with_name(format!("tanh_{id}"));
        self.add_node(node)
    }

    /// Add element-wise addition of two tensors.
    ///
    /// Shapes must be identical. The output shape matches the inputs.
    pub fn add(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let id = self.alloc_id();
        let spec = self
            .spec_of(a)
            .cloned()
            .unwrap_or_else(|_| TensorSpec::new(vec![], DType::F64));
        let node = OpNode::new(id, OpType::Add, vec![a.id, b.id], HashMap::new(), spec)
            .with_name(format!("add_{id}"));
        self.add_node(node)
    }

    /// Add element-wise multiplication of two tensors.
    pub fn mul(&mut self, a: Tensor, b: Tensor) -> Tensor {
        let id = self.alloc_id();
        let spec = self
            .spec_of(a)
            .cloned()
            .unwrap_or_else(|_| TensorSpec::new(vec![], DType::F64));
        let node = OpNode::new(id, OpType::Mul, vec![a.id, b.id], HashMap::new(), spec)
            .with_name(format!("mul_{id}"));
        self.add_node(node)
    }

    /// Reshape a tensor to `new_shape`.
    ///
    /// The total number of elements must be preserved (not validated at trace
    /// time if any dimension is symbolic; enforced at execution time).
    pub fn reshape(&mut self, input: Tensor, new_shape: Vec<usize>) -> Tensor {
        let id = self.alloc_id();
        let dtype = self.spec_of(input).map(|s| s.dtype).unwrap_or(DType::F64);
        let shape_attr: Vec<i64> = new_shape.iter().map(|&d| d as i64).collect();
        let mut attrs = HashMap::new();
        attrs.insert("new_shape".into(), OpAttr::IntList(shape_attr));

        let spec = TensorSpec::new(new_shape, dtype);
        let node = OpNode::new(id, OpType::Reshape, vec![input.id], attrs, spec)
            .with_name(format!("reshape_{id}"));
        self.add_node(node)
    }

    /// Add a softmax operation along `dim`.
    pub fn softmax(&mut self, input: Tensor, dim: usize) -> Tensor {
        let id = self.alloc_id();
        let spec = self
            .spec_of(input)
            .cloned()
            .unwrap_or_else(|_| TensorSpec::new(vec![], DType::F64));
        let mut attrs = HashMap::new();
        attrs.insert("dim".into(), OpAttr::Int(dim as i64));

        let node = OpNode::new(id, OpType::Softmax, vec![input.id], attrs, spec)
            .with_name(format!("softmax_{id}"));
        self.add_node(node)
    }

    /// Add a layer normalization operation.
    ///
    /// Normalizes over the last dimension. `eps` is a numerical stability term.
    pub fn layer_norm(&mut self, input: Tensor, eps: f64) -> Tensor {
        let id = self.alloc_id();
        let spec = self
            .spec_of(input)
            .cloned()
            .unwrap_or_else(|_| TensorSpec::new(vec![], DType::F64));
        let mut attrs = HashMap::new();
        attrs.insert("eps".into(), OpAttr::Float(eps));

        let node = OpNode::new(id, OpType::LayerNorm, vec![input.id], attrs, spec)
            .with_name(format!("layer_norm_{id}"));
        self.add_node(node)
    }

    /// Add a 1-D convolution layer.
    ///
    /// The operation performs a sliding-window dot product with the kernel, plus
    /// an optional per-output-channel bias.
    ///
    /// ## Weight-store key convention
    ///
    /// Weights for a `Conv1d` node with id `N` must be stored as:
    /// - `"conv1d_{N}_weight"` — flat row-major slice of shape `[out_channels, kernel_size]`
    /// - `"conv1d_{N}_bias"`   — flat slice of shape `[out_channels]`
    ///
    /// ## Output length
    ///
    /// With `padding = 0` and `stride = 1`:
    /// ```text
    /// out_len = in_len - kernel_size + 1
    /// ```
    /// With zero-padding of `padding` on each side:
    /// ```text
    /// out_len = in_len + 2*padding - kernel_size + 1
    /// ```
    ///
    /// The output tensor spec is `[batch, out_channels, out_len]`.
    pub fn conv1d(
        &mut self,
        input: Tensor,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Tensor {
        let id = self.alloc_id();

        // Infer output length from the input spec's last dimension.
        let in_len = self
            .spec_of(input)
            .ok()
            .and_then(|s| s.shape.last().copied())
            .unwrap_or(0);
        let out_len = if in_len + 2 * padding >= kernel_size {
            (in_len + 2 * padding - kernel_size) / stride + 1
        } else {
            0
        };

        let mut attrs = HashMap::new();
        attrs.insert("in_channels".into(), OpAttr::Int(in_channels as i64));
        attrs.insert("out_channels".into(), OpAttr::Int(out_channels as i64));
        attrs.insert("kernel_size".into(), OpAttr::Int(kernel_size as i64));
        attrs.insert("stride".into(), OpAttr::Int(stride as i64));
        attrs.insert("padding".into(), OpAttr::Int(padding as i64));

        let spec = TensorSpec::new(vec![out_channels, out_len], DType::F64);
        let node = OpNode::new(id, OpType::Conv1d, vec![input.id], attrs, spec)
            .with_name(format!("conv1d_{id}"));
        self.add_node(node)
    }

    // -----------------------------------------------------------------------
    // Finalization
    // -----------------------------------------------------------------------

    /// Finalize the graph.
    ///
    /// `output_tensors` designates which tensors are graph-level outputs.
    /// The builder populates forward edges (`outputs` field on each node)
    /// and constructs the `StaticGraph`.
    pub fn build(mut self, output_tensors: Vec<Tensor>) -> StaticGraph {
        // Populate forward edges from the consumer map
        for (producer_id, consumers) in &self.consumers {
            if let Some(node) = self.nodes.iter_mut().find(|n| n.id == *producer_id) {
                node.outputs.clone_from(consumers);
            }
        }

        // Collect input and output specs
        let input_specs: Vec<TensorSpec> = self
            .nodes
            .iter()
            .filter(|n| n.op_type == OpType::Constant && n.inputs.is_empty())
            .map(|n| n.output_spec.clone())
            .collect();

        let output_specs: Vec<TensorSpec> = output_tensors
            .iter()
            .filter_map(|t| self.nodes.iter().find(|n| n.id == t.id))
            .map(|n| n.output_spec.clone())
            .collect();

        let input_node_ids: Vec<usize> = self
            .nodes
            .iter()
            .filter(|n| n.op_type == OpType::Constant && n.inputs.is_empty())
            .map(|n| n.id)
            .collect();

        let output_node_ids: Vec<usize> = output_tensors.iter().map(|t| t.id).collect();

        // Build id → index map
        let mut id_to_idx = HashMap::new();
        for (idx, node) in self.nodes.iter().enumerate() {
            id_to_idx.insert(node.id, idx);
        }

        let mut graph = StaticGraph::new(input_specs, output_specs);
        graph.nodes = self.nodes;
        graph.id_to_idx = id_to_idx;
        graph.input_node_ids = input_node_ids;
        graph.output_node_ids = output_node_ids;
        graph
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracing::types::DType;

    #[test]
    fn test_graph_builder_linear() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 8], DType::F64));
        let out = builder.linear(input, 8, 4);
        let graph = builder.build(vec![out]);
        // One input (Constant) + one Linear node
        assert_eq!(graph.num_nodes(), 2);
        let linear_node = graph
            .nodes
            .iter()
            .find(|n| n.op_type == OpType::Linear)
            .expect("test: find linear node");
        assert_eq!(linear_node.attrs.get("out_features"), Some(&OpAttr::Int(4)));
    }

    #[test]
    fn test_graph_builder_relu() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 4], DType::F64));
        let relu_out = builder.relu(input);
        let graph = builder.build(vec![relu_out]);
        let relu_node = graph
            .nodes
            .iter()
            .find(|n| n.op_type == OpType::ReLU)
            .expect("test: find relu node");
        assert!(!relu_node.inputs.is_empty());
    }

    #[test]
    fn test_graph_topological_order() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 8], DType::F64));
        let h1 = builder.linear(input, 8, 4);
        let h2 = builder.relu(h1);
        let h3 = builder.linear(h2, 4, 2);
        let graph = builder.build(vec![h3]);

        // In topological order, each node's inputs must have lower IDs
        // (Since we allocate IDs sequentially and build in order, this holds)
        for node in &graph.nodes {
            for &inp_id in &node.inputs {
                assert!(
                    inp_id < node.id,
                    "node {} has input {} which comes after it",
                    node.id,
                    inp_id
                );
            }
        }
    }

    #[test]
    fn test_graph_static_shapes() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 16], DType::F64));
        let out = builder.linear(input, 16, 8);
        let graph = builder.build(vec![out]);

        let linear = graph
            .nodes
            .iter()
            .find(|n| n.op_type == OpType::Linear)
            .expect("test: find linear");
        assert_eq!(linear.output_spec.shape, vec![1, 8]);
    }

    #[test]
    fn test_graph_softmax_attr() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 10], DType::F64));
        let out = builder.softmax(input, 1);
        let graph = builder.build(vec![out]);
        let sm = graph
            .nodes
            .iter()
            .find(|n| n.op_type == OpType::Softmax)
            .expect("test: find softmax");
        assert_eq!(sm.attrs.get("dim"), Some(&OpAttr::Int(1)));
    }

    #[test]
    fn test_graph_layer_norm_attr() {
        let mut builder = GraphBuilder::new();
        let input = builder.input(TensorSpec::new(vec![1, 10], DType::F64));
        let out = builder.layer_norm(input, 1e-5);
        let graph = builder.build(vec![out]);
        let ln = graph
            .nodes
            .iter()
            .find(|n| n.op_type == OpType::LayerNorm)
            .expect("test: find layer_norm");
        assert_eq!(ln.attrs.get("eps"), Some(&OpAttr::Float(1e-5)));
    }
}
