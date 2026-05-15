//! JIT-style operator fusion with pattern matching
//!
//! This module detects fusible operation sequences in computation graphs and
//! executes them as single fused operations, reducing memory bandwidth and
//! intermediate allocations.
//!
//! # Supported fusion patterns
//!
//! - **FMA**: `a * b + c` -> fused multiply-add
//! - **Linear + Activation**: `matmul(x, w) + bias + activation`
//! - **Fused Attention**: `softmax(Q * K^T / sqrt(d)) * V`
//! - **Element-wise chains**: sequences of unary/binary element-wise ops
//! - **Residual add**: `x + f(x)` patterns
//!
//! # Example
//!
//! ```rust
//! use scirs2_autograd::jit_fusion::*;
//!
//! let engine = JitFusionEngine::new(FusionConfig::default());
//! let graph = vec![
//!     JitNode::new(0, JitOp::Input, vec![]),
//!     JitNode::new(1, JitOp::Input, vec![]),
//!     JitNode::new(2, JitOp::Mul, vec![0, 1]),
//!     JitNode::new(3, JitOp::Input, vec![]),
//!     JitNode::new(4, JitOp::Add, vec![2, 3]),
//! ];
//!
//! let fusions = engine.detect_fusions(&graph);
//! assert!(fusions.iter().any(|f| f.kind == FusionKindJit::Fma));
//! ```

use crate::error::AutogradError;
use crate::{Float, NdArray, Result};
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// JIT operation enum
// ---------------------------------------------------------------------------

/// Operation types for the JIT fusion analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JitOp {
    /// Input / constant
    Input,
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Matrix multiplication
    MatMul,
    /// ReLU activation
    Relu,
    /// GELU activation
    Gelu,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Softmax
    Softmax,
    /// Exp
    Exp,
    /// Log
    Log,
    /// Square
    Square,
    /// Sqrt
    Sqrt,
    /// Negation
    Neg,
    /// Scale by constant
    Scale,
    /// Bias addition (specialized add for broadcasting bias)
    BiasAdd,
    /// Transpose
    Transpose,
    /// Reduce sum
    ReduceSum,
    /// Reduce mean
    ReduceMean,
    /// Batched matrix multiplication (batch × M × K) × (batch × K × N)
    BatchedMatMul,
}

impl JitOp {
    /// Whether this op is element-wise (can be fused in a chain).
    pub fn is_elementwise(self) -> bool {
        matches!(
            self,
            JitOp::Add
                | JitOp::Sub
                | JitOp::Mul
                | JitOp::Div
                | JitOp::Relu
                | JitOp::Gelu
                | JitOp::Sigmoid
                | JitOp::Tanh
                | JitOp::Exp
                | JitOp::Log
                | JitOp::Square
                | JitOp::Sqrt
                | JitOp::Neg
                | JitOp::Scale
                | JitOp::BiasAdd
        )
    }

    /// Whether this op is an activation function.
    pub fn is_activation(self) -> bool {
        matches!(
            self,
            JitOp::Relu | JitOp::Gelu | JitOp::Sigmoid | JitOp::Tanh
        )
    }

    /// Whether this op is unary (exactly one non-constant input).
    pub fn is_unary(self) -> bool {
        matches!(
            self,
            JitOp::Relu
                | JitOp::Gelu
                | JitOp::Sigmoid
                | JitOp::Tanh
                | JitOp::Exp
                | JitOp::Log
                | JitOp::Square
                | JitOp::Sqrt
                | JitOp::Neg
                | JitOp::Scale
                | JitOp::Transpose
        )
    }
}

impl fmt::Display for JitOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// JIT node
// ---------------------------------------------------------------------------

/// A node in the JIT computation graph.
#[derive(Debug, Clone)]
pub struct JitNode {
    /// Unique node ID
    pub id: usize,
    /// Operation type
    pub op: JitOp,
    /// Input node IDs
    pub inputs: Vec<usize>,
    /// Output shape (if known)
    pub shape: Option<Vec<usize>>,
    /// Number of consumers of this node's output
    pub num_consumers: usize,
}

impl JitNode {
    /// Create a new JIT node.
    pub fn new(id: usize, op: JitOp, inputs: Vec<usize>) -> Self {
        Self {
            id,
            op,
            inputs,
            shape: None,
            num_consumers: 0,
        }
    }

    /// Create a node with shape.
    pub fn with_shape(id: usize, op: JitOp, inputs: Vec<usize>, shape: Vec<usize>) -> Self {
        Self {
            id,
            op,
            inputs,
            shape: Some(shape),
            num_consumers: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion kinds
// ---------------------------------------------------------------------------

/// Types of fused operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionKindJit {
    /// Fused multiply-add: a * b + c
    Fma,
    /// Fused linear: matmul + bias
    LinearBias,
    /// Fused linear + activation: matmul + bias + relu/gelu/sigmoid
    LinearBiasActivation,
    /// Fused attention: softmax(Q*K^T / sqrt(d)) * V
    FusedAttention,
    /// Chain of element-wise operations
    ElementWiseChain,
    /// Residual connection: x + f(x)
    ResidualAdd,
    /// Scale + bias: x * scale + bias (affine transform)
    Affine,
    /// Reduce + scale: sum(x) / n (mean)
    ReduceMean,
    /// MatMul followed by a chain of element-wise epilogue ops (bias, activation, …)
    MatmulEpilogue,
    /// Batched MatMul whose output is immediately reduced (sum/mean) over the batch dim
    BatchedMatmulReduction,
}

impl fmt::Display for FusionKindJit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionKindJit::Fma => write!(f, "FMA"),
            FusionKindJit::LinearBias => write!(f, "LinearBias"),
            FusionKindJit::LinearBiasActivation => write!(f, "LinearBiasActivation"),
            FusionKindJit::FusedAttention => write!(f, "FusedAttention"),
            FusionKindJit::ElementWiseChain => write!(f, "ElementWiseChain"),
            FusionKindJit::ResidualAdd => write!(f, "ResidualAdd"),
            FusionKindJit::Affine => write!(f, "Affine"),
            FusionKindJit::ReduceMean => write!(f, "ReduceMean"),
            FusionKindJit::MatmulEpilogue => write!(f, "MatmulEpilogue"),
            FusionKindJit::BatchedMatmulReduction => write!(f, "BatchedMatmulReduction"),
        }
    }
}

// ---------------------------------------------------------------------------
// Fusion candidate
// ---------------------------------------------------------------------------

/// A detected fusion opportunity.
#[derive(Debug, Clone)]
pub struct JitFusionCandidate {
    /// The kind of fusion
    pub kind: FusionKindJit,
    /// Node IDs involved in the fusion
    pub node_ids: Vec<usize>,
    /// Estimated benefit
    pub benefit: FusionBenefit,
    /// The root (output) node of the fused subgraph
    pub root_id: usize,
}

impl JitFusionCandidate {
    /// Whether this fusion is considered beneficial.
    pub fn is_beneficial(&self) -> bool {
        self.benefit.is_beneficial()
    }
}

/// Estimated benefit of a fusion.
#[derive(Debug, Clone)]
pub struct FusionBenefit {
    /// Estimated memory bandwidth savings in bytes
    pub memory_saved_bytes: usize,
    /// Estimated speedup factor (>1.0 means faster)
    pub speedup_estimate: f64,
    /// Number of intermediate tensors eliminated
    pub intermediates_eliminated: usize,
    /// Number of kernel launches saved
    pub kernel_launches_saved: usize,
}

impl FusionBenefit {
    /// Whether the fusion is considered beneficial.
    pub fn is_beneficial(&self) -> bool {
        self.speedup_estimate > 1.05 || self.intermediates_eliminated > 0
    }

    /// A scalar benefit score for comparing fusions.
    pub fn score(&self) -> f64 {
        self.speedup_estimate * (1.0 + self.intermediates_eliminated as f64 * 0.5)
    }
}

// ---------------------------------------------------------------------------
// Fusion configuration
// ---------------------------------------------------------------------------

/// Configuration for the JIT fusion engine.
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Enable FMA detection
    pub enable_fma: bool,
    /// Enable linear+bias+activation fusion
    pub enable_linear_fusion: bool,
    /// Enable attention fusion
    pub enable_attention_fusion: bool,
    /// Enable element-wise chain fusion
    pub enable_elementwise_chain: bool,
    /// Minimum chain length for element-wise fusion
    pub min_chain_length: usize,
    /// Maximum chain length for element-wise fusion
    pub max_chain_length: usize,
    /// Minimum benefit score to accept a fusion
    pub min_benefit_score: f64,
    /// Bytes per element (for bandwidth estimation)
    pub bytes_per_element: usize,
    /// Enable matmul → elementwise epilogue fusion
    pub enable_matmul_epilogue: bool,
    /// Enable batched-matmul → reduction fusion
    pub enable_batched_matmul_reduction: bool,
    /// Maximum number of epilogue ops to fuse after a matmul (capped at 4)
    pub max_matmul_epilogue_ops: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_fma: true,
            enable_linear_fusion: true,
            enable_attention_fusion: true,
            enable_elementwise_chain: true,
            min_chain_length: 2,
            max_chain_length: 16,
            min_benefit_score: 1.0,
            bytes_per_element: 4, // FP32
            enable_matmul_epilogue: true,
            enable_batched_matmul_reduction: true,
            max_matmul_epilogue_ops: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// JIT Fusion Engine
// ---------------------------------------------------------------------------

/// Engine for detecting and executing fused operation patterns.
pub struct JitFusionEngine {
    config: FusionConfig,
}

impl JitFusionEngine {
    /// Create a new JIT fusion engine.
    pub fn new(config: FusionConfig) -> Self {
        Self { config }
    }

    /// Detect all fusion opportunities in a graph.
    pub fn detect_fusions(&self, graph: &[JitNode]) -> Vec<JitFusionCandidate> {
        let mut candidates = Vec::new();
        let consumer_map = build_consumer_map(graph);

        if self.config.enable_fma {
            self.detect_fma(graph, &consumer_map, &mut candidates);
        }
        if self.config.enable_linear_fusion {
            self.detect_linear_bias_activation(graph, &consumer_map, &mut candidates);
        }
        if self.config.enable_attention_fusion {
            self.detect_fused_attention(graph, &consumer_map, &mut candidates);
        }
        if self.config.enable_elementwise_chain {
            self.detect_elementwise_chains(graph, &consumer_map, &mut candidates);
        }
        if self.config.enable_matmul_epilogue {
            self.detect_matmul_epilogue(graph, &consumer_map, &mut candidates);
        }
        if self.config.enable_batched_matmul_reduction {
            self.detect_batched_matmul_reduction(graph, &consumer_map, &mut candidates);
        }

        // Filter by benefit
        candidates.retain(|c| c.benefit.score() >= self.config.min_benefit_score);

        // Sort by benefit (descending)
        candidates.sort_by(|a, b| {
            b.benefit
                .score()
                .partial_cmp(&a.benefit.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove overlapping fusions (greedy: keep highest benefit first)
        remove_overlapping(&mut candidates);

        candidates
    }

    // --- FMA detection ---

    fn detect_fma(
        &self,
        graph: &[JitNode],
        consumer_map: &HashMap<usize, Vec<usize>>,
        candidates: &mut Vec<JitFusionCandidate>,
    ) {
        let node_map: HashMap<usize, &JitNode> = graph.iter().map(|n| (n.id, n)).collect();

        for node in graph {
            // Pattern: Add(Mul(a, b), c) -> FMA(a, b, c)
            if node.op == JitOp::Add && node.inputs.len() == 2 {
                for (mul_idx, other_idx) in [(0, 1), (1, 0)] {
                    let mul_input_id = node.inputs[mul_idx];
                    if let Some(mul_node) = node_map.get(&mul_input_id) {
                        if mul_node.op == JitOp::Mul && mul_node.inputs.len() == 2 {
                            // Check that mul has only one consumer (this add)
                            let consumers = consumer_map.get(&mul_input_id);
                            let single_consumer = consumers.is_some_and(|c| c.len() <= 1);
                            if single_consumer {
                                let shape_elements = node
                                    .shape
                                    .as_ref()
                                    .map(|s| s.iter().product::<usize>())
                                    .unwrap_or(1024);
                                let benefit = FusionBenefit {
                                    memory_saved_bytes: shape_elements
                                        * self.config.bytes_per_element,
                                    speedup_estimate: 1.3,
                                    intermediates_eliminated: 1,
                                    kernel_launches_saved: 1,
                                };
                                candidates.push(JitFusionCandidate {
                                    kind: FusionKindJit::Fma,
                                    node_ids: vec![mul_node.id, node.id],
                                    benefit,
                                    root_id: node.id,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Linear + Bias + Activation detection ---

    fn detect_linear_bias_activation(
        &self,
        graph: &[JitNode],
        consumer_map: &HashMap<usize, Vec<usize>>,
        candidates: &mut Vec<JitFusionCandidate>,
    ) {
        let node_map: HashMap<usize, &JitNode> = graph.iter().map(|n| (n.id, n)).collect();

        for node in graph {
            // Look for MatMul nodes
            if node.op != JitOp::MatMul {
                continue;
            }

            let matmul_id = node.id;
            let matmul_consumers = consumer_map.get(&matmul_id);

            // Check if matmul feeds into a bias add
            if let Some(consumers) = matmul_consumers {
                if consumers.len() != 1 {
                    continue;
                }
                let bias_add_id = consumers[0];
                if let Some(bias_node) = node_map.get(&bias_add_id) {
                    if bias_node.op != JitOp::Add && bias_node.op != JitOp::BiasAdd {
                        continue;
                    }

                    // We have matmul + bias. Check if it feeds into an activation.
                    let bias_consumers = consumer_map.get(&bias_add_id);
                    if let Some(bias_cons) = bias_consumers {
                        if bias_cons.len() == 1 {
                            if let Some(act_node) = node_map.get(&bias_cons[0]) {
                                if act_node.op.is_activation() {
                                    // Fused: matmul + bias + activation
                                    let shape_elements = node
                                        .shape
                                        .as_ref()
                                        .map(|s| s.iter().product::<usize>())
                                        .unwrap_or(4096);
                                    let benefit = FusionBenefit {
                                        memory_saved_bytes: shape_elements
                                            * self.config.bytes_per_element
                                            * 2,
                                        speedup_estimate: 1.5,
                                        intermediates_eliminated: 2,
                                        kernel_launches_saved: 2,
                                    };
                                    candidates.push(JitFusionCandidate {
                                        kind: FusionKindJit::LinearBiasActivation,
                                        node_ids: vec![matmul_id, bias_add_id, act_node.id],
                                        benefit,
                                        root_id: act_node.id,
                                    });
                                    continue;
                                }
                            }
                        }
                    }

                    // No activation — just matmul + bias
                    let shape_elements = node
                        .shape
                        .as_ref()
                        .map(|s| s.iter().product::<usize>())
                        .unwrap_or(4096);
                    let benefit = FusionBenefit {
                        memory_saved_bytes: shape_elements * self.config.bytes_per_element,
                        speedup_estimate: 1.2,
                        intermediates_eliminated: 1,
                        kernel_launches_saved: 1,
                    };
                    candidates.push(JitFusionCandidate {
                        kind: FusionKindJit::LinearBias,
                        node_ids: vec![matmul_id, bias_add_id],
                        benefit,
                        root_id: bias_add_id,
                    });
                }
            }
        }
    }

    // --- Fused attention detection ---

    fn detect_fused_attention(
        &self,
        graph: &[JitNode],
        consumer_map: &HashMap<usize, Vec<usize>>,
        candidates: &mut Vec<JitFusionCandidate>,
    ) {
        let node_map: HashMap<usize, &JitNode> = graph.iter().map(|n| (n.id, n)).collect();

        // Pattern: MatMul(Q, K^T) -> Scale(/sqrt(d)) -> Softmax -> MatMul(_, V)
        for node in graph {
            // Start from a Softmax node
            if node.op != JitOp::Softmax {
                continue;
            }

            let softmax_id = node.id;

            // Check if softmax input is a Scale or Div (the /sqrt(d) step)
            if node.inputs.is_empty() {
                continue;
            }
            let pre_softmax_id = node.inputs[0];
            let pre_softmax_node = match node_map.get(&pre_softmax_id) {
                Some(n) => n,
                None => continue,
            };

            let (scale_or_div, qk_matmul_id) =
                if pre_softmax_node.op == JitOp::Scale || pre_softmax_node.op == JitOp::Div {
                    if pre_softmax_node.inputs.is_empty() {
                        continue;
                    }
                    (true, pre_softmax_node.inputs[0])
                } else if pre_softmax_node.op == JitOp::MatMul {
                    (false, pre_softmax_id)
                } else {
                    continue;
                };

            // Verify the Q*K^T matmul
            let qk_node = match node_map.get(&qk_matmul_id) {
                Some(n) if n.op == JitOp::MatMul => n,
                _ => continue,
            };

            // Check if softmax output feeds into another matmul (* V)
            let softmax_consumers = consumer_map.get(&softmax_id);
            if let Some(consumers) = softmax_consumers {
                for &consumer_id in consumers {
                    if let Some(final_matmul) = node_map.get(&consumer_id) {
                        if final_matmul.op == JitOp::MatMul {
                            let mut node_ids = vec![qk_matmul_id];
                            if scale_or_div {
                                node_ids.push(pre_softmax_id);
                            }
                            node_ids.push(softmax_id);
                            node_ids.push(consumer_id);

                            let shape_elements = qk_node
                                .shape
                                .as_ref()
                                .map(|s| s.iter().product::<usize>())
                                .unwrap_or(65536);
                            let benefit = FusionBenefit {
                                memory_saved_bytes: shape_elements
                                    * self.config.bytes_per_element
                                    * 3,
                                speedup_estimate: 2.0,
                                intermediates_eliminated: 3,
                                kernel_launches_saved: 3,
                            };
                            candidates.push(JitFusionCandidate {
                                kind: FusionKindJit::FusedAttention,
                                node_ids,
                                benefit,
                                root_id: consumer_id,
                            });
                        }
                    }
                }
            }
        }
    }

    // --- Element-wise chain detection ---

    fn detect_elementwise_chains(
        &self,
        graph: &[JitNode],
        consumer_map: &HashMap<usize, Vec<usize>>,
        candidates: &mut Vec<JitFusionCandidate>,
    ) {
        let node_map: HashMap<usize, &JitNode> = graph.iter().map(|n| (n.id, n)).collect();
        let mut visited = HashSet::new();

        for node in graph {
            if !node.op.is_elementwise() || node.op == JitOp::Add || node.op == JitOp::Mul {
                // Skip simple binary ops as they are handled by FMA
                // We focus on unary chains or mixed element-wise chains
                if !node.op.is_unary() {
                    continue;
                }
            }

            if visited.contains(&node.id) {
                continue;
            }

            // Walk backwards through unary element-wise chain
            let mut chain = vec![node.id];
            let mut current = node;

            // Walk forward (downstream)
            loop {
                let consumers = consumer_map.get(&current.id);
                if let Some(cons) = consumers {
                    if cons.len() == 1 {
                        if let Some(next) = node_map.get(&cons[0]) {
                            if next.op.is_elementwise()
                                && next.op.is_unary()
                                && chain.len() < self.config.max_chain_length
                            {
                                chain.push(next.id);
                                current = next;
                                continue;
                            }
                        }
                    }
                }
                break;
            }

            if chain.len() >= self.config.min_chain_length {
                for &id in &chain {
                    visited.insert(id);
                }
                let shape_elements = node
                    .shape
                    .as_ref()
                    .map(|s| s.iter().product::<usize>())
                    .unwrap_or(1024);
                let benefit = FusionBenefit {
                    memory_saved_bytes: shape_elements
                        * self.config.bytes_per_element
                        * (chain.len() - 1),
                    speedup_estimate: 1.0 + (chain.len() - 1) as f64 * 0.15,
                    intermediates_eliminated: chain.len() - 1,
                    kernel_launches_saved: chain.len() - 1,
                };
                let root_id = *chain.last().unwrap_or(&node.id);
                candidates.push(JitFusionCandidate {
                    kind: FusionKindJit::ElementWiseChain,
                    node_ids: chain,
                    benefit,
                    root_id,
                });
            }
        }
    }

    // --- MatMul → elementwise epilogue detection ---
    //
    // Detects patterns where a MatMul output feeds into a chain of up to
    // `max_matmul_epilogue_ops` element-wise operations (bias add, scale,
    // activation, etc.).  This subsumes `LinearBias` / `LinearBiasActivation`
    // at the pattern level, but those are detected first and will claim the
    // nodes via `remove_overlapping`; this detector therefore captures
    // *longer* or *different* epilogue chains that the simpler detectors miss.
    //
    // The fused "kernel" executes matmul and all epilogue ops in a single pass
    // over the output buffer, eliminating intermediate tensor allocations.
    fn detect_matmul_epilogue(
        &self,
        graph: &[JitNode],
        consumer_map: &HashMap<usize, Vec<usize>>,
        candidates: &mut Vec<JitFusionCandidate>,
    ) {
        let node_map: HashMap<usize, &JitNode> = graph.iter().map(|n| (n.id, n)).collect();
        let cap = self.config.max_matmul_epilogue_ops.min(4);

        for node in graph {
            if node.op != JitOp::MatMul {
                continue;
            }
            let matmul_id = node.id;

            // Walk forward through a single-consumer elementwise chain.
            let mut chain = vec![matmul_id];
            let mut current_id = matmul_id;

            loop {
                if chain.len() > cap {
                    break;
                }
                let consumers = match consumer_map.get(&current_id) {
                    Some(c) => c,
                    None => break,
                };
                if consumers.len() != 1 {
                    break;
                }
                let next_id = consumers[0];
                let next_node = match node_map.get(&next_id) {
                    Some(n) => n,
                    None => break,
                };
                if !next_node.op.is_elementwise() {
                    break;
                }
                chain.push(next_id);
                current_id = next_id;
            }

            // Require at least one epilogue op (chain includes the matmul itself).
            if chain.len() < 2 {
                continue;
            }

            let shape_elements = node
                .shape
                .as_ref()
                .map(|s| s.iter().product::<usize>())
                .unwrap_or(4096);
            let epilogue_count = chain.len() - 1;
            // Use a slightly lower speedup than LinearBias (1.2) for single-epilogue
            // chains so that the more specific LinearBias pattern is preferred by the
            // greedy overlap-removal pass.  Each additional epilogue op adds 0.08,
            // so a 3-op epilogue chain reaches 1.24 and beats LinearBias (1.2).
            let speedup = 1.12 + (epilogue_count.saturating_sub(1)) as f64 * 0.08;
            let benefit = FusionBenefit {
                memory_saved_bytes: shape_elements * self.config.bytes_per_element * epilogue_count,
                speedup_estimate: speedup,
                intermediates_eliminated: epilogue_count,
                kernel_launches_saved: epilogue_count,
            };
            let root_id = *chain.last().unwrap_or(&matmul_id);
            candidates.push(JitFusionCandidate {
                kind: FusionKindJit::MatmulEpilogue,
                node_ids: chain,
                benefit,
                root_id,
            });
        }
    }

    // --- Batched MatMul → sum/mean reduction detection ---
    //
    // Detects patterns of the form `MatMul → ReduceSum` or `MatMul → ReduceMean`
    // where the reduction immediately follows the matmul with a single consumer.
    // Fusing these avoids materializing the full matmul output before reducing,
    // saving `batch_size × M × N × bytes_per_element` of intermediate memory.
    //
    // The `BatchedMatMul` op variant (3-D tensors) is also covered: since the
    // JIT analysis is shape-agnostic, keying on `JitOp::MatMul` (or the new
    // `JitOp::BatchedMatMul`) covers both cases.
    fn detect_batched_matmul_reduction(
        &self,
        graph: &[JitNode],
        consumer_map: &HashMap<usize, Vec<usize>>,
        candidates: &mut Vec<JitFusionCandidate>,
    ) {
        let node_map: HashMap<usize, &JitNode> = graph.iter().map(|n| (n.id, n)).collect();

        for node in graph {
            if node.op != JitOp::MatMul && node.op != JitOp::BatchedMatMul {
                continue;
            }
            let matmul_id = node.id;

            let consumers = match consumer_map.get(&matmul_id) {
                Some(c) if c.len() == 1 => c,
                _ => continue,
            };
            let reduce_id = consumers[0];
            let reduce_node = match node_map.get(&reduce_id) {
                Some(n) => n,
                None => continue,
            };

            if reduce_node.op != JitOp::ReduceSum && reduce_node.op != JitOp::ReduceMean {
                continue;
            }

            let shape_elements = node
                .shape
                .as_ref()
                .map(|s| s.iter().product::<usize>())
                .unwrap_or(4096);
            let benefit = FusionBenefit {
                memory_saved_bytes: shape_elements * self.config.bytes_per_element,
                speedup_estimate: 1.25,
                intermediates_eliminated: 1,
                kernel_launches_saved: 1,
            };
            candidates.push(JitFusionCandidate {
                kind: FusionKindJit::BatchedMatmulReduction,
                node_ids: vec![matmul_id, reduce_id],
                benefit,
                root_id: reduce_id,
            });
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &FusionConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// JIT compiled operation
// ---------------------------------------------------------------------------

/// Represents a compiled (fused) operation that can be executed.
#[derive(Debug, Clone)]
pub struct JitCompiledOp {
    /// The fusion kind
    pub kind: FusionKindJit,
    /// Operations in the fused sequence
    pub ops: Vec<JitOp>,
    /// Human-readable name
    pub name: String,
}

impl JitCompiledOp {
    /// Create a new compiled op from a fusion candidate.
    pub fn from_candidate(candidate: &JitFusionCandidate, graph: &[JitNode]) -> Self {
        let node_map: HashMap<usize, &JitNode> = graph.iter().map(|n| (n.id, n)).collect();
        let ops: Vec<JitOp> = candidate
            .node_ids
            .iter()
            .filter_map(|id| node_map.get(id).map(|n| n.op))
            .collect();
        let name = format!("fused_{}", candidate.kind);
        Self {
            kind: candidate.kind,
            ops,
            name,
        }
    }

    /// Execute the fused FMA: a * b + c
    pub fn execute_fma<F: Float>(
        a: &NdArray<F>,
        b: &NdArray<F>,
        c: &NdArray<F>,
    ) -> Result<NdArray<F>> {
        if a.shape() != b.shape() || a.shape() != c.shape() {
            // Try scalar broadcast for c
            if c.len() == 1 {
                let cv = *c
                    .iter()
                    .next()
                    .ok_or_else(|| AutogradError::compute_error("Empty scalar".into()))?;
                let mut result = a.clone();
                result.zip_mut_with(b, |av, bv| {
                    *av = *av * *bv + cv;
                });
                return Ok(result);
            }
            if a.shape() != b.shape() {
                return Err(AutogradError::ShapeMismatch(format!(
                    "FMA shape mismatch: {:?} vs {:?} vs {:?}",
                    a.shape(),
                    b.shape(),
                    c.shape()
                )));
            }
        }

        let mut result = a.clone();
        let b_slice = b
            .as_slice()
            .ok_or_else(|| AutogradError::compute_error("FMA: non-contiguous tensor b".into()))?;
        let c_slice = c
            .as_slice()
            .ok_or_else(|| AutogradError::compute_error("FMA: non-contiguous tensor c".into()))?;
        let result_slice = result
            .as_slice_mut()
            .ok_or_else(|| AutogradError::compute_error("FMA: non-contiguous result".into()))?;

        for i in 0..result_slice.len() {
            result_slice[i] = result_slice[i] * b_slice[i] + c_slice[i];
        }

        Ok(result)
    }

    /// Execute fused linear + bias + optional activation.
    pub fn execute_linear_bias_activation<F: Float>(
        x: &NdArray<F>,
        w: &NdArray<F>,
        bias: &NdArray<F>,
        activation: Option<JitOp>,
    ) -> Result<NdArray<F>> {
        let x_shape = x.shape();
        let w_shape = w.shape();

        if x_shape.len() < 2 || w_shape.len() < 2 {
            return Err(AutogradError::ShapeMismatch(
                "Linear requires 2-D inputs".into(),
            ));
        }

        let m = x_shape[0];
        let k = x_shape[1];
        let n = w_shape[1];

        if k != w_shape[0] {
            return Err(AutogradError::ShapeMismatch(format!(
                "Linear inner dim mismatch: {} vs {}",
                k, w_shape[0]
            )));
        }

        let x_slice = x
            .as_slice()
            .ok_or_else(|| AutogradError::compute_error("Non-contiguous input x".into()))?;
        let w_slice = w
            .as_slice()
            .ok_or_else(|| AutogradError::compute_error("Non-contiguous weight w".into()))?;

        let bias_flat: Vec<F> = bias.iter().copied().collect();
        if bias_flat.len() != n {
            return Err(AutogradError::ShapeMismatch(format!(
                "Bias length {} != out_features {}",
                bias_flat.len(),
                n
            )));
        }

        let mut result = ArrayD::<F>::zeros(IxDyn(&[m, n]));
        let result_slice = result
            .as_slice_mut()
            .ok_or_else(|| AutogradError::compute_error("Non-contiguous result".into()))?;

        // Fused matmul + bias + activation in a single pass
        for i in 0..m {
            for j in 0..n {
                let mut acc = F::zero();
                for p in 0..k {
                    acc += x_slice[i * k + p] * w_slice[p * n + j];
                }
                acc += bias_flat[j];

                // Apply activation inline
                acc = match activation {
                    Some(JitOp::Relu) => {
                        if acc > F::zero() {
                            acc
                        } else {
                            F::zero()
                        }
                    }
                    Some(JitOp::Sigmoid) => F::one() / (F::one() + (-acc).exp()),
                    Some(JitOp::Tanh) => acc.tanh(),
                    Some(JitOp::Gelu) => {
                        let sqrt_2_pi = F::from(0.7978845608028654).unwrap_or(F::one());
                        let coeff = F::from(0.044715).unwrap_or(F::zero());
                        let half = F::from(0.5).unwrap_or(F::one());
                        let inner = sqrt_2_pi * (acc + coeff * acc * acc * acc);
                        half * acc * (F::one() + inner.tanh())
                    }
                    _ => acc,
                };

                result_slice[i * n + j] = acc;
            }
        }

        Ok(result)
    }

    /// Execute a chain of element-wise operations.
    pub fn execute_elementwise_chain<F: Float>(
        input: &NdArray<F>,
        ops: &[JitOp],
    ) -> Result<NdArray<F>> {
        let mut current = input.clone();

        for op in ops {
            current = match op {
                JitOp::Relu => current.mapv(|v| if v > F::zero() { v } else { F::zero() }),
                JitOp::Sigmoid => current.mapv(|v| F::one() / (F::one() + (-v).exp())),
                JitOp::Tanh => current.mapv(|v| v.tanh()),
                JitOp::Gelu => {
                    let sqrt_2_pi = F::from(0.7978845608028654).unwrap_or(F::one());
                    let coeff = F::from(0.044715).unwrap_or(F::zero());
                    let half = F::from(0.5).unwrap_or(F::one());
                    current.mapv(|x| {
                        let inner = sqrt_2_pi * (x + coeff * x * x * x);
                        half * x * (F::one() + inner.tanh())
                    })
                }
                JitOp::Exp => current.mapv(|v| v.exp()),
                JitOp::Log => current.mapv(|v| v.ln()),
                JitOp::Neg => current.mapv(|v| -v),
                JitOp::Square => current.mapv(|v| v * v),
                JitOp::Sqrt => current.mapv(|v| v.sqrt()),
                _ => current,
            };
        }

        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// Fusion benefit estimator
// ---------------------------------------------------------------------------

/// Estimates the benefit of a potential fusion.
pub struct FusionBenefitEstimator {
    /// Bytes per element
    bytes_per_element: usize,
    /// Memory bandwidth in bytes/sec (for throughput estimation)
    memory_bandwidth: f64,
    /// Compute throughput in FLOP/sec
    compute_throughput: f64,
}

impl FusionBenefitEstimator {
    /// Create a new estimator with default hardware assumptions.
    pub fn new() -> Self {
        Self {
            bytes_per_element: 4,
            memory_bandwidth: 50.0e9,   // 50 GB/s
            compute_throughput: 1.0e12, // 1 TFLOP/s
        }
    }

    /// Create with custom parameters.
    pub fn with_params(
        bytes_per_element: usize,
        memory_bandwidth: f64,
        compute_throughput: f64,
    ) -> Self {
        Self {
            bytes_per_element,
            memory_bandwidth,
            compute_throughput,
        }
    }

    /// Estimate memory bandwidth savings from fusing element-wise ops.
    pub fn elementwise_chain_savings(
        &self,
        num_elements: usize,
        chain_length: usize,
    ) -> FusionBenefit {
        // Without fusion: each op reads+writes the entire tensor
        // With fusion: one read + one write total
        let unfused_bytes = num_elements * self.bytes_per_element * 2 * chain_length;
        let fused_bytes = num_elements * self.bytes_per_element * 2;
        let saved = unfused_bytes.saturating_sub(fused_bytes);
        let speedup = if fused_bytes > 0 {
            unfused_bytes as f64 / fused_bytes as f64
        } else {
            1.0
        };

        FusionBenefit {
            memory_saved_bytes: saved,
            speedup_estimate: speedup.min(chain_length as f64),
            intermediates_eliminated: chain_length - 1,
            kernel_launches_saved: chain_length - 1,
        }
    }

    /// Estimate savings from fusing matmul + bias.
    pub fn linear_bias_savings(&self, m: usize, n: usize) -> FusionBenefit {
        let output_bytes = m * n * self.bytes_per_element;
        FusionBenefit {
            memory_saved_bytes: output_bytes, // avoid one write+read cycle
            speedup_estimate: 1.2,
            intermediates_eliminated: 1,
            kernel_launches_saved: 1,
        }
    }

    /// Estimate savings from fusing matmul + bias + activation.
    pub fn linear_bias_activation_savings(&self, m: usize, n: usize) -> FusionBenefit {
        let output_bytes = m * n * self.bytes_per_element;
        FusionBenefit {
            memory_saved_bytes: output_bytes * 2,
            speedup_estimate: 1.4,
            intermediates_eliminated: 2,
            kernel_launches_saved: 2,
        }
    }

    /// Estimate savings from fused attention.
    pub fn attention_savings(&self, seq_len: usize, d_model: usize) -> FusionBenefit {
        let qk_size = seq_len * seq_len * self.bytes_per_element;
        let attn_size = seq_len * d_model * self.bytes_per_element;
        FusionBenefit {
            memory_saved_bytes: qk_size * 2 + attn_size,
            speedup_estimate: 2.0,
            intermediates_eliminated: 3,
            kernel_launches_saved: 3,
        }
    }

    /// Estimate savings from FMA.
    pub fn fma_savings(&self, num_elements: usize) -> FusionBenefit {
        let bytes = num_elements * self.bytes_per_element;
        FusionBenefit {
            memory_saved_bytes: bytes,
            speedup_estimate: 1.3,
            intermediates_eliminated: 1,
            kernel_launches_saved: 1,
        }
    }
}

impl Default for FusionBenefitEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Pattern registry
// ---------------------------------------------------------------------------

/// A rule for pattern matching in the JIT fusion engine.
#[derive(Debug, Clone)]
pub struct FusionRule {
    /// Name of the rule
    pub name: String,
    /// Expected op sequence (root to leaves)
    pub pattern: Vec<JitOp>,
    /// Fusion kind produced
    pub kind: FusionKindJit,
    /// Estimated speedup
    pub speedup: f64,
}

/// Registry of fusion rules, allowing extensibility.
pub struct PatternRegistry {
    rules: Vec<FusionRule>,
}

impl PatternRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Create a registry with the default built-in rules.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();

        reg.add_rule(FusionRule {
            name: "fma".into(),
            pattern: vec![JitOp::Add, JitOp::Mul],
            kind: FusionKindJit::Fma,
            speedup: 1.3,
        });
        reg.add_rule(FusionRule {
            name: "linear_bias".into(),
            pattern: vec![JitOp::Add, JitOp::MatMul],
            kind: FusionKindJit::LinearBias,
            speedup: 1.2,
        });
        reg.add_rule(FusionRule {
            name: "linear_bias_relu".into(),
            pattern: vec![JitOp::Relu, JitOp::Add, JitOp::MatMul],
            kind: FusionKindJit::LinearBiasActivation,
            speedup: 1.5,
        });
        reg.add_rule(FusionRule {
            name: "linear_bias_gelu".into(),
            pattern: vec![JitOp::Gelu, JitOp::Add, JitOp::MatMul],
            kind: FusionKindJit::LinearBiasActivation,
            speedup: 1.5,
        });

        reg
    }

    /// Add a rule.
    pub fn add_rule(&mut self, rule: FusionRule) {
        self.rules.push(rule);
    }

    /// Get all rules.
    pub fn rules(&self) -> &[FusionRule] {
        &self.rules
    }

    /// Number of rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Match a rule against a sequence of ops (root-first order).
    pub fn match_rule(&self, ops: &[JitOp]) -> Option<&FusionRule> {
        self.rules.iter().find(|rule| {
            if rule.pattern.len() != ops.len() {
                return false;
            }
            rule.pattern.iter().zip(ops).all(|(r, o)| r == o)
        })
    }
}

impl Default for PatternRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a map from node ID to its consumer node IDs.
fn build_consumer_map(graph: &[JitNode]) -> HashMap<usize, Vec<usize>> {
    let mut map: HashMap<usize, Vec<usize>> = HashMap::new();
    for node in graph {
        for &input_id in &node.inputs {
            map.entry(input_id).or_default().push(node.id);
        }
    }
    map
}

/// Remove overlapping fusion candidates (greedy: keep highest-scored first).
fn remove_overlapping(candidates: &mut Vec<JitFusionCandidate>) {
    let mut used_nodes = HashSet::new();
    candidates.retain(|c| {
        let overlaps = c.node_ids.iter().any(|id| used_nodes.contains(id));
        if overlaps {
            false
        } else {
            for &id in &c.node_ids {
                used_nodes.insert(id);
            }
            true
        }
    });
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    // --- JitOp tests ---

    #[test]
    fn test_jit_op_is_elementwise() {
        assert!(JitOp::Add.is_elementwise());
        assert!(JitOp::Relu.is_elementwise());
        assert!(!JitOp::MatMul.is_elementwise());
        assert!(!JitOp::Softmax.is_elementwise());
    }

    #[test]
    fn test_jit_op_is_activation() {
        assert!(JitOp::Relu.is_activation());
        assert!(JitOp::Gelu.is_activation());
        assert!(!JitOp::Add.is_activation());
    }

    #[test]
    fn test_jit_op_display() {
        assert_eq!(format!("{}", JitOp::MatMul), "MatMul");
    }

    // --- FMA detection ---

    #[test]
    fn test_detect_fma() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),
            JitNode::new(1, JitOp::Input, vec![]),
            JitNode::new(2, JitOp::Mul, vec![0, 1]),
            JitNode::new(3, JitOp::Input, vec![]),
            JitNode::new(4, JitOp::Add, vec![2, 3]),
        ];

        let fusions = engine.detect_fusions(&graph);
        assert!(fusions.iter().any(|f| f.kind == FusionKindJit::Fma));
    }

    #[test]
    fn test_fma_not_detected_when_mul_has_multiple_consumers() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),
            JitNode::new(1, JitOp::Input, vec![]),
            JitNode::new(2, JitOp::Mul, vec![0, 1]),
            JitNode::new(3, JitOp::Input, vec![]),
            JitNode::new(4, JitOp::Add, vec![2, 3]),
            JitNode::new(5, JitOp::Neg, vec![2]), // second consumer of mul
        ];

        let fusions = engine.detect_fusions(&graph);
        assert!(!fusions.iter().any(|f| f.kind == FusionKindJit::Fma));
    }

    // --- Linear + Bias detection ---

    #[test]
    fn test_detect_linear_bias() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),
            JitNode::new(1, JitOp::Input, vec![]),
            JitNode::new(2, JitOp::MatMul, vec![0, 1]),
            JitNode::new(3, JitOp::Input, vec![]),
            JitNode::new(4, JitOp::Add, vec![2, 3]),
        ];

        let fusions = engine.detect_fusions(&graph);
        let has_linear = fusions.iter().any(|f| f.kind == FusionKindJit::LinearBias);
        assert!(has_linear);
    }

    // --- Linear + Bias + Activation detection ---

    #[test]
    fn test_detect_linear_bias_activation() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),
            JitNode::new(1, JitOp::Input, vec![]),
            JitNode::new(2, JitOp::MatMul, vec![0, 1]),
            JitNode::new(3, JitOp::Input, vec![]),
            JitNode::new(4, JitOp::Add, vec![2, 3]),
            JitNode::new(5, JitOp::Relu, vec![4]),
        ];

        let fusions = engine.detect_fusions(&graph);
        let has_lba = fusions
            .iter()
            .any(|f| f.kind == FusionKindJit::LinearBiasActivation);
        assert!(has_lba);
    }

    // --- Attention detection ---

    #[test]
    fn test_detect_fused_attention() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),      // Q
            JitNode::new(1, JitOp::Input, vec![]),      // K^T
            JitNode::new(2, JitOp::MatMul, vec![0, 1]), // Q*K^T
            JitNode::new(3, JitOp::Scale, vec![2]),     // / sqrt(d)
            JitNode::new(4, JitOp::Softmax, vec![3]),   // softmax
            JitNode::new(5, JitOp::Input, vec![]),      // V
            JitNode::new(6, JitOp::MatMul, vec![4, 5]), // * V
        ];

        let fusions = engine.detect_fusions(&graph);
        let has_attn = fusions
            .iter()
            .any(|f| f.kind == FusionKindJit::FusedAttention);
        assert!(has_attn);
    }

    // --- Element-wise chain detection ---

    #[test]
    fn test_detect_elementwise_chain() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),
            JitNode::new(1, JitOp::Relu, vec![0]),
            JitNode::new(2, JitOp::Sigmoid, vec![1]),
            JitNode::new(3, JitOp::Neg, vec![2]),
        ];

        let fusions = engine.detect_fusions(&graph);
        let has_chain = fusions
            .iter()
            .any(|f| f.kind == FusionKindJit::ElementWiseChain);
        assert!(has_chain);
    }

    #[test]
    fn test_no_chain_for_single_op() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),
            JitNode::new(1, JitOp::Relu, vec![0]),
        ];

        let fusions = engine.detect_fusions(&graph);
        let has_chain = fusions
            .iter()
            .any(|f| f.kind == FusionKindJit::ElementWiseChain);
        assert!(!has_chain);
    }

    // --- Execution tests ---

    #[test]
    fn test_execute_fma() {
        let a = Array1::from(vec![2.0_f64, 3.0, 4.0]).into_dyn();
        let b = Array1::from(vec![5.0_f64, 6.0, 7.0]).into_dyn();
        let c = Array1::from(vec![1.0_f64, 1.0, 1.0]).into_dyn();
        let result = JitCompiledOp::execute_fma(&a, &b, &c).expect("fma");
        assert!((result[[0]] - 11.0).abs() < 1e-10); // 2*5+1
        assert!((result[[1]] - 19.0).abs() < 1e-10); // 3*6+1
        assert!((result[[2]] - 29.0).abs() < 1e-10); // 4*7+1
    }

    #[test]
    fn test_execute_linear_bias() {
        let x = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f64, 0.0, 0.0, 0.0, 1.0, 0.0])
            .expect("x");
        let w =
            ArrayD::from_shape_vec(IxDyn(&[3, 2]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("w");
        let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.1, 0.2]).expect("bias");

        let result =
            JitCompiledOp::execute_linear_bias_activation(&x, &w, &bias, None).expect("linear");
        // x[0] = [1,0,0] -> matmul = [1,2] + bias = [1.1, 2.2]
        assert!((result[IxDyn(&[0, 0])] - 1.1).abs() < 1e-10);
        assert!((result[IxDyn(&[0, 1])] - 2.2).abs() < 1e-10);
    }

    #[test]
    fn test_execute_linear_bias_relu() {
        let x = ArrayD::from_shape_vec(IxDyn(&[1, 2]), vec![-1.0_f64, 1.0]).expect("x");
        let w = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 0.0, 0.0, 1.0]).expect("w");
        let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 0.0]).expect("bias");

        let result =
            JitCompiledOp::execute_linear_bias_activation(&x, &w, &bias, Some(JitOp::Relu))
                .expect("linear+relu");
        // matmul = [-1, 1] + bias [0,0] = [-1, 1], relu = [0, 1]
        assert!((result[IxDyn(&[0, 0])]).abs() < 1e-10);
        assert!((result[IxDyn(&[0, 1])] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_execute_elementwise_chain() {
        let input = Array1::from(vec![-2.0_f64, -1.0, 0.0, 1.0, 2.0]).into_dyn();
        let ops = vec![JitOp::Relu, JitOp::Square];
        let result = JitCompiledOp::execute_elementwise_chain(&input, &ops).expect("chain");
        // relu: [0, 0, 0, 1, 2], square: [0, 0, 0, 1, 4]
        assert!((result[[0]]).abs() < 1e-10);
        assert!((result[[3]] - 1.0).abs() < 1e-10);
        assert!((result[[4]] - 4.0).abs() < 1e-10);
    }

    // --- FusionBenefitEstimator tests ---

    #[test]
    fn test_estimator_elementwise_chain() {
        let estimator = FusionBenefitEstimator::new();
        let benefit = estimator.elementwise_chain_savings(1000, 5);
        assert!(benefit.memory_saved_bytes > 0);
        assert!(benefit.speedup_estimate > 1.0);
        assert_eq!(benefit.intermediates_eliminated, 4);
        assert_eq!(benefit.kernel_launches_saved, 4);
    }

    #[test]
    fn test_estimator_linear_bias() {
        let estimator = FusionBenefitEstimator::new();
        let benefit = estimator.linear_bias_savings(32, 64);
        assert!(benefit.memory_saved_bytes > 0);
    }

    #[test]
    fn test_estimator_attention() {
        let estimator = FusionBenefitEstimator::new();
        let benefit = estimator.attention_savings(128, 512);
        assert!(benefit.speedup_estimate >= 2.0);
        assert_eq!(benefit.intermediates_eliminated, 3);
    }

    #[test]
    fn test_estimator_fma() {
        let estimator = FusionBenefitEstimator::new();
        let benefit = estimator.fma_savings(1024);
        assert!(benefit.memory_saved_bytes > 0);
    }

    // --- PatternRegistry tests ---

    #[test]
    fn test_pattern_registry_defaults() {
        let reg = PatternRegistry::with_defaults();
        assert!(reg.len() >= 4);
    }

    #[test]
    fn test_pattern_registry_match() {
        let reg = PatternRegistry::with_defaults();
        let matched = reg.match_rule(&[JitOp::Add, JitOp::Mul]);
        assert!(matched.is_some());
        assert_eq!(matched.map(|r| r.kind), Some(FusionKindJit::Fma));
    }

    #[test]
    fn test_pattern_registry_no_match() {
        let reg = PatternRegistry::with_defaults();
        let matched = reg.match_rule(&[JitOp::Softmax, JitOp::Sqrt]);
        assert!(matched.is_none());
    }

    #[test]
    fn test_pattern_registry_add_custom() {
        let mut reg = PatternRegistry::new();
        assert!(reg.is_empty());
        reg.add_rule(FusionRule {
            name: "custom".into(),
            pattern: vec![JitOp::Neg, JitOp::Exp],
            kind: FusionKindJit::ElementWiseChain,
            speedup: 1.1,
        });
        assert_eq!(reg.len(), 1);
    }

    // --- FusionBenefit tests ---

    #[test]
    fn test_benefit_is_beneficial() {
        let benefit = FusionBenefit {
            memory_saved_bytes: 1000,
            speedup_estimate: 1.1,
            intermediates_eliminated: 1,
            kernel_launches_saved: 1,
        };
        assert!(benefit.is_beneficial());
    }

    #[test]
    fn test_benefit_not_beneficial() {
        let benefit = FusionBenefit {
            memory_saved_bytes: 0,
            speedup_estimate: 1.0,
            intermediates_eliminated: 0,
            kernel_launches_saved: 0,
        };
        assert!(!benefit.is_beneficial());
    }

    #[test]
    fn test_benefit_score() {
        let benefit = FusionBenefit {
            memory_saved_bytes: 100,
            speedup_estimate: 2.0,
            intermediates_eliminated: 2,
            kernel_launches_saved: 2,
        };
        // score = 2.0 * (1 + 2 * 0.5) = 2.0 * 2.0 = 4.0
        assert!((benefit.score() - 4.0).abs() < 1e-10);
    }

    // --- FusionConfig tests ---

    #[test]
    fn test_default_config() {
        let config = FusionConfig::default();
        assert!(config.enable_fma);
        assert!(config.enable_linear_fusion);
        assert!(config.enable_attention_fusion);
        assert!(config.enable_elementwise_chain);
        assert_eq!(config.min_chain_length, 2);
    }

    // --- JitCompiledOp::from_candidate ---

    #[test]
    fn test_compiled_op_from_candidate() {
        let graph = vec![
            JitNode::new(0, JitOp::Mul, vec![]),
            JitNode::new(1, JitOp::Add, vec![0]),
        ];
        let candidate = JitFusionCandidate {
            kind: FusionKindJit::Fma,
            node_ids: vec![0, 1],
            benefit: FusionBenefit {
                memory_saved_bytes: 100,
                speedup_estimate: 1.3,
                intermediates_eliminated: 1,
                kernel_launches_saved: 1,
            },
            root_id: 1,
        };
        let compiled = JitCompiledOp::from_candidate(&candidate, &graph);
        assert_eq!(compiled.kind, FusionKindJit::Fma);
        assert_eq!(compiled.ops.len(), 2);
    }

    // --- Overlapping fusion removal ---

    #[test]
    fn test_overlapping_removal() {
        let engine = JitFusionEngine::new(FusionConfig::default());
        // Construct a graph where two fusions overlap on node 2
        // Fusion A: nodes [1, 2], Fusion B: nodes [2, 3]
        // Only the higher-scored one should survive
        let graph = vec![
            JitNode::new(0, JitOp::Input, vec![]),
            JitNode::new(1, JitOp::Relu, vec![0]),
            JitNode::new(2, JitOp::Sigmoid, vec![1]),
            JitNode::new(3, JitOp::Neg, vec![2]),
        ];
        let fusions = engine.detect_fusions(&graph);
        // All should be in one chain, no overlap
        let chain_count = fusions
            .iter()
            .filter(|f| f.kind == FusionKindJit::ElementWiseChain)
            .count();
        assert!(chain_count <= 1);
    }

    // --- execute_linear_bias_activation shape mismatch ---

    #[test]
    fn test_linear_shape_mismatch() {
        let x = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0_f64; 6]).expect("x");
        let w = ArrayD::from_shape_vec(IxDyn(&[4, 2]), vec![1.0_f64; 8]).expect("w");
        let bias = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f64; 2]).expect("bias");
        let result = JitCompiledOp::execute_linear_bias_activation(&x, &w, &bias, None);
        assert!(result.is_err());
    }

    // --- FMA scalar broadcast ---

    #[test]
    fn test_fma_scalar_broadcast() {
        let a = Array1::from(vec![2.0_f64, 3.0]).into_dyn();
        let b = Array1::from(vec![4.0_f64, 5.0]).into_dyn();
        let c = ArrayD::from_elem(IxDyn(&[1]), 10.0_f64);
        let result = JitCompiledOp::execute_fma(&a, &b, &c).expect("fma broadcast");
        assert!((result[[0]] - 18.0).abs() < 1e-10); // 2*4+10
        assert!((result[[1]] - 25.0).abs() < 1e-10); // 3*5+10
    }
}
