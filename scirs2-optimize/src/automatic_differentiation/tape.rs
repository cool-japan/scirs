//! Computational tape for reverse-mode automatic differentiation
//!
//! This module implements the tape structure used to record computational
//! operations for later backpropagation in reverse-mode AD.

use crate::error::OptimizeError;
use std::collections::HashMap;

/// Type alias for batch processor function
type BatchProcessor = Box<dyn Fn(&[TapeNode]) -> Result<(), OptimizeError>>;

/// A variable in the computational tape
#[derive(Debug, Clone)]
pub struct Variable {
    /// Unique identifier for this variable
    pub id: usize,
    /// Current value
    pub value: f64,
}

impl Variable {
    /// Create a new variable
    pub fn new(id: usize, value: f64) -> Self {
        Self { id, value }
    }
}

/// A node in the computational tape representing an operation
#[derive(Debug, Clone)]
pub enum TapeNode {
    /// Input variable (leaf node)
    Input { var_id: usize },
    /// Constant value
    Constant { value: f64 },
    /// Unary operation
    UnaryOp {
        input: usize,
        result: usize,
        partial: f64, // ∂result/∂input
    },
    /// Binary operation
    BinaryOp {
        left: usize,
        right: usize,
        result: usize,
        left_partial: f64,  // ∂result/∂left
        right_partial: f64, // ∂result/∂right
    },
    /// N-ary operation (for efficiency with many inputs)
    NAryOp {
        inputs: Vec<usize>,
        result: usize,
        partials: Vec<f64>, // ∂result/∂inputs[i]
    },
}

/// Computational tape for recording operations
#[derive(Debug)]
pub struct ComputationTape {
    /// Sequence of operations in forward order
    nodes: Vec<TapeNode>,
    /// Input variables
    inputs: Vec<Variable>,
    /// Mapping from variable ID to its position in the tape
    var_positions: HashMap<usize, usize>,
    /// Maximum variable ID used
    max_var_id: usize,
}

impl ComputationTape {
    /// Create a new empty tape
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            inputs: Vec::new(),
            var_positions: HashMap::new(),
            max_var_id: 0,
        }
    }

    /// Add an input variable to the tape
    pub fn add_input(&mut self, var: Variable) {
        self.var_positions.insert(var.id, self.nodes.len());
        self.max_var_id = self.max_var_id.max(var.id);

        self.nodes.push(TapeNode::Input { var_id: var.id });
        self.inputs.push(var);
    }

    /// Add a computation node to the tape
    pub fn add_node(&mut self, node: TapeNode) {
        // Update variable positions for result variables
        match &node {
            TapeNode::UnaryOp { result, .. } => {
                self.var_positions.insert(*result, self.nodes.len());
                self.max_var_id = self.max_var_id.max(*result);
            }
            TapeNode::BinaryOp { result, .. } => {
                self.var_positions.insert(*result, self.nodes.len());
                self.max_var_id = self.max_var_id.max(*result);
            }
            TapeNode::NAryOp { result, .. } => {
                self.var_positions.insert(*result, self.nodes.len());
                self.max_var_id = self.max_var_id.max(*result);
            }
            _ => {}
        }

        self.nodes.push(node);
    }

    /// Perform backpropagation to compute gradients
    pub fn backward(&self, gradients: &mut Vec<f64>) -> Result<(), OptimizeError> {
        // Ensure gradients vector is large enough
        if gradients.len() <= self.max_var_id {
            gradients.resize(self.max_var_id + 1, 0.0);
        }

        // Reverse pass through the tape
        for node in self.nodes.iter().rev() {
            match node {
                TapeNode::Input { .. } => {
                    // Input nodes don't propagate gradients backward
                }
                TapeNode::Constant { .. } => {
                    // Constants have zero gradient
                }
                TapeNode::UnaryOp {
                    input,
                    result,
                    partial,
                } => {
                    // Propagate gradient: ∂L/∂input += ∂L/∂result * ∂result/∂input
                    gradients[*input] += gradients[*result] * partial;
                }
                TapeNode::BinaryOp {
                    left,
                    right,
                    result,
                    left_partial,
                    right_partial,
                } => {
                    // Propagate gradients to both inputs
                    gradients[*left] += gradients[*result] * left_partial;
                    gradients[*right] += gradients[*result] * right_partial;
                }
                TapeNode::NAryOp {
                    inputs,
                    result,
                    partials,
                } => {
                    // Propagate gradient to all inputs
                    for (input_id, partial) in inputs.iter().zip(partials.iter()) {
                        gradients[*input_id] += gradients[*result] * partial;
                    }
                }
            }
        }

        Ok(())
    }

    /// Forward pass to compute all variable values
    pub fn forward(&self, input_values: &[f64]) -> Result<Vec<f64>, OptimizeError> {
        let mut values = vec![0.0; self.max_var_id + 1];

        // Set input values
        for (i, var) in self.inputs.iter().enumerate() {
            if i < input_values.len() {
                values[var.id] = input_values[i];
            } else {
                values[var.id] = var.value; // Use default value
            }
        }

        // Forward pass through the tape
        for node in &self.nodes {
            match node {
                TapeNode::Input { .. } => {
                    // Already handled above
                }
                TapeNode::Constant { value: _ } => {
                    // Constants are handled implicitly
                }
                TapeNode::UnaryOp { input, result, .. } => {
                    // The actual operation would be stored elsewhere
                    // This is just the structure for gradient computation
                    values[*result] = values[*input]; // Placeholder
                }
                TapeNode::BinaryOp {
                    left,
                    right,
                    result,
                    ..
                } => {
                    // The actual operation would be stored elsewhere
                    values[*result] = values[*left] + values[*right]; // Placeholder
                }
                TapeNode::NAryOp { inputs, result, .. } => {
                    // The actual operation would be stored elsewhere
                    values[*result] = inputs.iter().map(|&id| values[id]).sum();
                    // Placeholder
                }
            }
        }

        Ok(values)
    }

    /// Optimize the tape by removing unnecessary operations
    pub fn optimize(&mut self) {
        // Remove redundant operations, constant folding, etc.
        // This is a placeholder for more sophisticated optimizations

        // Remove nodes that are never used
        let mut used_vars = std::collections::HashSet::new();

        // Mark all variables that are actually used
        for node in &self.nodes {
            match node {
                TapeNode::UnaryOp { input, result, .. } => {
                    used_vars.insert(*input);
                    used_vars.insert(*result);
                }
                TapeNode::BinaryOp {
                    left,
                    right,
                    result,
                    ..
                } => {
                    used_vars.insert(*left);
                    used_vars.insert(*right);
                    used_vars.insert(*result);
                }
                TapeNode::NAryOp { inputs, result, .. } => {
                    for &input_id in inputs {
                        used_vars.insert(input_id);
                    }
                    used_vars.insert(*result);
                }
                TapeNode::Input { var_id } => {
                    used_vars.insert(*var_id);
                }
                _ => {}
            }
        }

        // Could implement more optimizations here
    }

    /// Get the size of the tape
    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the tape is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.inputs.clear();
        self.var_positions.clear();
        self.max_var_id = 0;
    }

    /// Get statistics about the tape
    pub fn get_stats(&self) -> TapeStats {
        let mut unary_ops = 0;
        let mut binary_ops = 0;
        let mut nary_ops = 0;
        let mut constants = 0;

        for node in &self.nodes {
            match node {
                TapeNode::Input { .. } => {}
                TapeNode::Constant { .. } => constants += 1,
                TapeNode::UnaryOp { .. } => unary_ops += 1,
                TapeNode::BinaryOp { .. } => binary_ops += 1,
                TapeNode::NAryOp { .. } => nary_ops += 1,
            }
        }

        TapeStats {
            total_nodes: self.nodes.len(),
            input_vars: self.inputs.len(),
            unary_ops,
            binary_ops,
            nary_ops,
            constants,
            max_var_id: self.max_var_id,
        }
    }
}

impl Default for ComputationTape {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a computation tape
#[derive(Debug, Clone)]
pub struct TapeStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of input variables
    pub input_vars: usize,
    /// Number of unary operations
    pub unary_ops: usize,
    /// Number of binary operations
    pub binary_ops: usize,
    /// Number of n-ary operations
    pub nary_ops: usize,
    /// Number of constants
    pub constants: usize,
    /// Maximum variable ID
    pub max_var_id: usize,
}

/// Tape builder for more convenient tape construction
pub struct TapeBuilder {
    tape: ComputationTape,
    next_var_id: usize,
}

impl TapeBuilder {
    /// Create a new tape builder
    pub fn new() -> Self {
        Self {
            tape: ComputationTape::new(),
            next_var_id: 0,
        }
    }

    /// Add an input variable
    pub fn input(&mut self, value: f64) -> usize {
        let var_id = self.next_var_id;
        self.next_var_id += 1;

        let var = Variable::new(var_id, value);
        self.tape.add_input(var);

        var_id
    }

    /// Add a unary operation
    pub fn unary_op(&mut self, input: usize, partial: f64) -> usize {
        let result_id = self.next_var_id;
        self.next_var_id += 1;

        let node = TapeNode::UnaryOp {
            input,
            result: result_id,
            partial,
        };
        self.tape.add_node(node);

        result_id
    }

    /// Add a binary operation
    pub fn binary_op(
        &mut self,
        left: usize,
        right: usize,
        left_partial: f64,
        right_partial: f64,
    ) -> usize {
        let result_id = self.next_var_id;
        self.next_var_id += 1;

        let node = TapeNode::BinaryOp {
            left,
            right,
            result: result_id,
            left_partial,
            right_partial,
        };
        self.tape.add_node(node);

        result_id
    }

    /// Finish building and return the tape
    pub fn build(self) -> ComputationTape {
        self.tape
    }
}

impl Default for TapeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-efficient tape that can handle very large computations
pub struct StreamingTape {
    /// Current batch of operations
    current_batch: Vec<TapeNode>,
    /// Batch size for processing
    batch_size: usize,
    /// Function to process completed batches
    batch_processor: Option<BatchProcessor>,
}

impl StreamingTape {
    /// Create a new streaming tape
    pub fn new(batch_size: usize) -> Self {
        Self {
            current_batch: Vec::with_capacity(batch_size),
            batch_size,
            batch_processor: None,
        }
    }

    /// Set the batch processor
    pub fn set_batch_processor<F>(&mut self, processor: F)
    where
        F: Fn(&[TapeNode]) -> Result<(), OptimizeError> + 'static,
    {
        self.batch_processor = Some(Box::new(processor));
    }

    /// Add a node to the streaming tape
    pub fn add_node(&mut self, node: TapeNode) -> Result<(), OptimizeError> {
        self.current_batch.push(node);

        if self.current_batch.len() >= self.batch_size {
            self.flush_batch()?;
        }

        Ok(())
    }

    /// Flush the current batch
    pub fn flush_batch(&mut self) -> Result<(), OptimizeError> {
        if let Some(ref processor) = self.batch_processor {
            processor(&self.current_batch)?;
        }
        self.current_batch.clear();
        Ok(())
    }

    /// Finalize the streaming tape
    pub fn finalize(&mut self) -> Result<(), OptimizeError> {
        if !self.current_batch.is_empty() {
            self.flush_batch()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tape_construction() {
        let mut builder = TapeBuilder::new();

        // Build tape for: z = (x + y) * x
        let x = builder.input(2.0);
        let y = builder.input(3.0);
        let sum = builder.binary_op(x, y, 1.0, 1.0); // x + y, partials: ∂/∂x=1, ∂/∂y=1
        let _result = builder.binary_op(sum, x, 2.0, 5.0); // sum * x, partials: ∂/∂sum=x=2, ∂/∂x=sum=5

        let tape = builder.build();

        assert_eq!(tape.size(), 4); // 2 inputs + 2 operations

        let stats = tape.get_stats();
        assert_eq!(stats.input_vars, 2);
        assert_eq!(stats.binary_ops, 2);
    }

    #[test]
    fn test_backward_pass() {
        let mut tape = ComputationTape::new();

        // Add inputs: x=2, y=3
        tape.add_input(Variable::new(0, 2.0));
        tape.add_input(Variable::new(1, 3.0));

        // Add operation: z = x + y (result=2)
        tape.add_node(TapeNode::BinaryOp {
            left: 0,
            right: 1,
            result: 2,
            left_partial: 1.0,  // ∂z/∂x = 1
            right_partial: 1.0, // ∂z/∂y = 1
        });

        // Initialize gradients: ∂L/∂z = 1 (z is the output)
        let mut gradients = vec![0.0, 0.0, 1.0];

        tape.backward(&mut gradients).unwrap();

        // Check gradients
        assert_eq!(gradients[0], 1.0); // ∂L/∂x = ∂L/∂z * ∂z/∂x = 1 * 1 = 1
        assert_eq!(gradients[1], 1.0); // ∂L/∂y = ∂L/∂z * ∂z/∂y = 1 * 1 = 1
    }

    #[test]
    fn test_tape_optimization() {
        let mut tape = ComputationTape::new();

        tape.add_input(Variable::new(0, 1.0));
        tape.add_node(TapeNode::UnaryOp {
            input: 0,
            result: 1,
            partial: 1.0,
        });

        let original_size = tape.size();
        tape.optimize();

        // Optimization might not change this simple tape, but it shouldn't break it
        assert!(tape.size() <= original_size);
    }

    #[test]
    fn test_streaming_tape() {
        let mut streaming_tape = StreamingTape::new(2);

        streaming_tape.set_batch_processor(move |_batch| {
            // Just a placeholder processor for testing
            Ok(())
        });

        // Add nodes - should trigger batch processing
        streaming_tape
            .add_node(TapeNode::Input { var_id: 0 })
            .unwrap();
        streaming_tape
            .add_node(TapeNode::Input { var_id: 1 })
            .unwrap();

        // This should have triggered one batch
        streaming_tape
            .add_node(TapeNode::UnaryOp {
                input: 0,
                result: 2,
                partial: 1.0,
            })
            .unwrap();

        streaming_tape.finalize().unwrap();
    }

    #[test]
    fn test_tape_stats() {
        let mut builder = TapeBuilder::new();

        let x = builder.input(1.0);
        let y = builder.input(2.0);
        builder.binary_op(x, y, 1.0, 1.0);
        builder.unary_op(x, 2.0);

        let tape = builder.build();
        let stats = tape.get_stats();

        assert_eq!(stats.input_vars, 2);
        assert_eq!(stats.binary_ops, 1);
        assert_eq!(stats.unary_ops, 1);
        assert_eq!(stats.total_nodes, 4); // 2 inputs + 1 binary + 1 unary
    }
}
