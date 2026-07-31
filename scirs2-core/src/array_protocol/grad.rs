// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Gradient computation support for the array protocol.
//!
//! This module provides automatic differentiation capabilities for arrays
//! using the array protocol. It enables gradient computation for any array
//! type that implements the `ArrayProtocol` trait.

use crate::ndarray::compat::ArrayStatCompat;
use ::ndarray::{Array, ArrayD, Dimension, Ix1, Ix2, IxDyn};

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use crate::array_protocol::operations::matmul;
use crate::array_protocol::{ArrayProtocol, NdarrayWrapper};
use crate::error::{CoreError, CoreResult, ErrorContext};

/// Dictionary for storing parameter gradients
#[derive(Clone)]
pub struct GradientDict {
    gradients: HashMap<String, Box<dyn ArrayProtocol>>,
}

impl std::fmt::Debug for GradientDict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GradientDict")
            .field(
                "gradients",
                &format!("{{keys: {:?}}}", self.gradients.keys().collect::<Vec<_>>()),
            )
            .finish()
    }
}

impl GradientDict {
    /// Create a new empty gradient dictionary
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
        }
    }

    /// Insert a gradient for a parameter
    pub fn insert(&mut self, name: String, gradient: Box<dyn ArrayProtocol>) {
        self.gradients.insert(name, gradient);
    }

    /// Get a gradient by parameter name
    pub fn get(&self, name: &str) -> Option<&dyn ArrayProtocol> {
        self.gradients.get(name).map(|b| b.as_ref())
    }

    /// Get a mutable reference to a gradient by parameter name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Box<dyn ArrayProtocol>> {
        self.gradients.get_mut(name)
    }

    /// Iterate over parameter names and gradients
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Box<dyn ArrayProtocol>)> {
        self.gradients.iter()
    }

    /// Merge another gradient dictionary into this one
    pub fn merge(&mut self, other: GradientDict) {
        for (name, gradient) in other.gradients {
            self.gradients.insert(name, gradient);
        }
    }

    /// Check if the dictionary is empty
    pub fn is_empty(&self) -> bool {
        self.gradients.is_empty()
    }

    /// Get the number of gradients
    pub fn len(&self) -> usize {
        self.gradients.len()
    }

    /// Clear all gradients
    pub fn clear(&mut self) {
        self.gradients.clear();
    }

    /// Get all parameter names
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.gradients.keys()
    }

    /// Get all gradients
    pub fn values(&self) -> impl Iterator<Item = &Box<dyn ArrayProtocol>> {
        self.gradients.values()
    }
}

impl Default for GradientDict {
    fn default() -> Self {
        Self::new()
    }
}

// Convert Box<dyn ArrayProtocol> to Rc<dyn ArrayProtocol> with proper trait object handling
#[allow(dead_code)]
fn boxed_to_rc(boxed: Box<dyn ArrayProtocol>) -> Rc<dyn ArrayProtocol> {
    // `Rc<dyn Trait>` has a direct `From<Box<dyn Trait>>` impl in std: it
    // reallocates the box's contents behind an `Rc` while preserving the exact
    // concrete type and vtable. This previously downcast-enumerated only
    // `NdarrayWrapper<f64, IxDyn>` and silently substituted a fabricated 1x1
    // zero placeholder for every other type or dimension (e.g. the very common
    // `NdarrayWrapper<f64, Ix2>` produced by `add`/`multiply` on `Array2`
    // inputs) — a wrong-but-`Ok` result rather than a visible error.
    Rc::from(boxed)
}

// Helper function to convert Box<dyn ArrayProtocol> to Rc<dyn ArrayProtocol>
#[allow(dead_code)]
fn box_to_rc_array_protocol(boxed: Box<dyn ArrayProtocol>) -> Rc<dyn ArrayProtocol> {
    boxed_to_rc(boxed)
}

// Import the functions with the correct return type for our use
#[allow(dead_code)]
fn add(a: &dyn ArrayProtocol, b: &dyn ArrayProtocol) -> CoreResult<Box<dyn ArrayProtocol>> {
    crate::array_protocol::operations::add(a, b).map_err(|e| e.into())
}

#[allow(dead_code)]
fn multiply(a: &dyn ArrayProtocol, b: &dyn ArrayProtocol) -> CoreResult<Box<dyn ArrayProtocol>> {
    crate::array_protocol::operations::multiply(a, b).map_err(|e| e.into())
}

#[allow(dead_code)]
fn subtract(a: &dyn ArrayProtocol, b: &dyn ArrayProtocol) -> CoreResult<Box<dyn ArrayProtocol>> {
    crate::array_protocol::operations::subtract(a, b).map_err(|e| e.into())
}

#[allow(dead_code)]
fn ones_like(a: &dyn ArrayProtocol) -> CoreResult<Box<dyn ArrayProtocol>> {
    // Create an array of ones with the same shape as the input
    // Try different numeric types
    if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
        let shape = a_array.as_array().shape();
        let ones = ArrayD::<f64>::ones(IxDyn(shape));
        Ok(Box::new(NdarrayWrapper::new(ones)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>() {
        let shape = a_array.as_array().shape();
        let ones = ArrayD::<f32>::ones(IxDyn(shape));
        Ok(Box::new(NdarrayWrapper::new(ones)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>() {
        let shape = a_array.as_array().shape();
        let ones = ArrayD::<i32>::ones(IxDyn(shape));
        Ok(Box::new(NdarrayWrapper::new(ones)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>() {
        let shape = a_array.as_array().shape();
        let ones = ArrayD::<i64>::ones(IxDyn(shape));
        Ok(Box::new(NdarrayWrapper::new(ones)) as Box<dyn ArrayProtocol>)
    } else {
        // Try to get shape information using the array protocol methods
        let shape = a.shape().to_vec();
        let ones = ArrayD::<f64>::ones(IxDyn(&shape));
        Ok(Box::new(NdarrayWrapper::new(ones)) as Box<dyn ArrayProtocol>)
    }
}

#[allow(dead_code)]
fn broadcast_to(a: &dyn ArrayProtocol, shape: &[usize]) -> CoreResult<Box<dyn ArrayProtocol>> {
    // Broadcast the array to the given shape
    if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
        let array = a_array.as_array();
        // Handle scalar broadcasting
        if array.len() == 1 {
            let value = array.iter().next().cloned().unwrap_or(0.0);
            let broadcasted = ArrayD::<f64>::from_elem(IxDyn(shape), value);
            Ok(Box::new(NdarrayWrapper::new(broadcasted)) as Box<dyn ArrayProtocol>)
        } else if array.shape() == shape {
            // Shapes already match
            Ok(Box::new(NdarrayWrapper::new(array.clone())) as Box<dyn ArrayProtocol>)
        } else {
            // Implement basic broadcasting rules
            let inputshape = array.shape();
            let _ndim_diff = shape.len().saturating_sub(inputshape.len());

            // Check if broadcasting is possible
            let mut can_broadcast = true;
            for i in 0..inputshape.len() {
                let input_dim = inputshape[inputshape.len() - 1 - i];
                let target_dim = shape[shape.len() - 1 - i];
                if input_dim != 1 && input_dim != target_dim {
                    can_broadcast = false;
                    break;
                }
            }

            if can_broadcast {
                // Perform broadcasting by repeating data
                if let Some(broadcasted_view) = array.broadcast(IxDyn(shape)) {
                    let broadcasted = broadcasted_view.to_owned();
                    Ok(Box::new(NdarrayWrapper::new(broadcasted)) as Box<dyn ArrayProtocol>)
                } else {
                    Err(CoreError::NotImplementedError(ErrorContext::new(
                        "Broadcasting failed for these shapes".to_string(),
                    )))
                }
            } else {
                Err(CoreError::NotImplementedError(ErrorContext::new(
                    "Incompatible shapes for broadcasting".to_string(),
                )))
            }
        }
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>() {
        let array = a_array.as_array();
        if array.len() == 1 {
            let value = array.iter().next().cloned().unwrap_or(0.0);
            let broadcasted = ArrayD::<f32>::from_elem(IxDyn(shape), value);
            Ok(Box::new(NdarrayWrapper::new(broadcasted)) as Box<dyn ArrayProtocol>)
        } else if array.shape() == shape {
            Ok(Box::new(NdarrayWrapper::new(array.clone())) as Box<dyn ArrayProtocol>)
        } else if let Some(broadcasted_view) = array.broadcast(IxDyn(shape)) {
            let broadcasted = broadcasted_view.to_owned();
            Ok(Box::new(NdarrayWrapper::new(broadcasted)) as Box<dyn ArrayProtocol>)
        } else {
            Err(CoreError::NotImplementedError(ErrorContext::new(
                "Broadcasting failed for these shapes".to_string(),
            )))
        }
    } else {
        // Fallback: create an array of ones with the target shape
        let ones = ArrayD::<f64>::ones(IxDyn(shape));
        Ok(Box::new(NdarrayWrapper::new(ones)) as Box<dyn ArrayProtocol>)
    }
}

/// A node in the computation graph.
#[derive(Clone)]
struct Node {
    /// The array value at this node.
    value: Rc<dyn ArrayProtocol>,

    /// Gradient with respect to the output.
    grad: Option<Rc<dyn ArrayProtocol>>,

    /// Operation that created this node.
    op: Option<String>,

    /// Input nodes for the operation that created this node.
    inputs: Vec<GradientTensor>,

    /// Whether gradient computation is required for this node.
    requiresgrad: bool,

    /// Whether this node is a leaf node (parameter or input).
    is_leaf: bool,
}

impl Node {
    /// Create a new leaf node.
    fn leaf(requiresgrad: bool) -> Self {
        Self {
            value: Rc::new(NdarrayWrapper::new(
                crate::ndarray::Array0::<f64>::zeros(()),
            )) as Rc<dyn ArrayProtocol>,
            grad: None,
            op: None,
            inputs: Vec::new(),
            requiresgrad,
            is_leaf: true,
        }
    }

    /// Create a new operation node.
    fn new_op(value: Rc<dyn ArrayProtocol>, op: String, inputs: Vec<GradientTensor>) -> Self {
        let requiresgrad = inputs.iter().any(|x| x.requiresgrad());

        Self {
            value,
            grad: None,
            op: Some(op),
            inputs,
            requiresgrad,
            is_leaf: false,
        }
    }
}

/// Tensor with gradient tracking capabilities.
#[derive(Clone)]
pub struct GradientTensor {
    /// The node in the computation graph.
    node: Rc<RefCell<Node>>,
}

impl GradientTensor {
    /// Create a new gradient tensor from a value.
    pub fn new(value: Rc<dyn ArrayProtocol>, requiresgrad: bool) -> Self {
        let mut node_inner = Node::leaf(requiresgrad);
        node_inner.value = value;
        node_inner.grad = None;
        let node = Rc::new(RefCell::new(node_inner));
        Self { node }
    }

    /// Create a new gradient tensor from an array.
    pub fn from_array<T, D>(array: Array<T, D>, requiresgrad: bool) -> Self
    where
        T: Clone + Send + Sync + 'static,
        D: Dimension + Send + Sync + 'static,
    {
        let value = Rc::new(NdarrayWrapper::new(array)) as Rc<dyn ArrayProtocol>;
        Self::new(value, requiresgrad)
    }

    /// Get the value of the tensor.
    pub fn value(&self) -> Rc<dyn ArrayProtocol> {
        self.node.borrow().value.clone()
    }

    /// Get the gradient of the tensor.
    pub fn grad_2(&self) -> Option<Rc<dyn ArrayProtocol>> {
        self.node.borrow().grad.clone()
    }

    /// Check if gradient computation is required for this tensor.
    pub fn requiresgrad(&self) -> bool {
        self.node.borrow().requiresgrad
    }

    /// Set whether gradient computation is required for this tensor.
    pub fn set_requiresgrad(&mut self, requiresgrad: bool) {
        self.node.borrow_mut().requiresgrad = requiresgrad;
    }

    /// Check if this tensor is a leaf node.
    pub fn is_leaf(&self) -> bool {
        self.node.borrow().is_leaf
    }

    /// Create a new tensor from an operation.
    fn from_op(value: Rc<dyn ArrayProtocol>, op: String, inputs: Vec<GradientTensor>) -> Self {
        let node = Rc::new(RefCell::new(Node::new_op(value, op, inputs)));
        Self { node }
    }

    /// Set the value of the tensor (for updating variables during optimization).
    pub fn set_value(&mut self, newvalue: Rc<dyn ArrayProtocol>) {
        self.node.borrow_mut().grad = None; // Clear gradient when value changes
        self.node.borrow_mut().value = newvalue;
    }

    /// Backward pass to compute gradients.
    pub fn backward(&self) -> CoreResult<()> {
        // Initialize gradient as ones with the same shape as value.
        // `ArrayProtocol::shape()` works for any concrete dimension type,
        // unlike downcasting to a specific `NdarrayWrapper<f64, D>` — this
        // used to only recognize `IxDyn` and silently fall back to a
        // scalar-shaped gradient (`[1]`) for any other dimension (e.g. the
        // very common `Ix2`), corrupting the shape the entire backward pass
        // is seeded with.
        let gradshape = IxDyn(self.value().shape());

        let grad_array = Array::<f64, IxDyn>::ones(gradshape);
        let grad = Rc::new(NdarrayWrapper::new(grad_array)) as Rc<dyn ArrayProtocol>;

        // Perform backward pass
        self.backward_with_grad(grad)
    }

    /// Backward pass with a specific gradient.
    fn backward_with_grad(&self, grad: Rc<dyn ArrayProtocol>) -> CoreResult<()> {
        // Set the gradient of this tensor
        self.node.borrow_mut().grad = Some(grad.clone());

        // Create a topologically sorted list of nodes
        let mut visited = HashSet::new();
        let mut topo = Vec::new();

        // Helper function for topological sort
        fn build_topo(
            tensor: &GradientTensor,
            visited: &mut HashSet<*const RefCell<Node>>,
            topo: &mut Vec<GradientTensor>,
        ) {
            let node_ptr = Rc::as_ptr(&tensor.node);
            if !visited.contains(&node_ptr) {
                visited.insert(node_ptr);

                // Visit all inputs first
                for input in &tensor.node.borrow().inputs {
                    build_topo(input, visited, topo);
                }

                // Then add this node
                topo.push(tensor.clone());
            }
        }

        // Build topological sort
        build_topo(self, &mut visited, &mut topo);

        // Perform backward pass in reverse topological order
        for node in topo.iter().rev() {
            // Only compute gradients for nodes that require it
            if !node.requiresgrad() {
                continue;
            }

            // Get the gradient of this node
            let node_grad = match node.grad_2() {
                Some(g) => g,
                None => continue, // Skip nodes with no gradient
            };

            // If this is a leaf node, we're done
            if node.is_leaf() {
                continue;
            }

            // Get the operation and inputs
            let op = match &node.node.borrow().op {
                Some(op) => op.clone(),
                None => continue, // Skip nodes with no operation
            };

            let inputs = node.node.borrow().inputs.clone();

            // Compute gradients for inputs based on the operation
            match op.as_str() {
                "add" => {
                    // For addition, gradient flows directly to both inputs
                    for input in &inputs {
                        if input.requiresgrad() {
                            let mut input_node = input.node.borrow_mut();
                            if let Some(input_grad) = &input_node.grad {
                                // Accumulate gradients
                                if let Ok(sum) = add(input_grad.as_ref(), node_grad.as_ref()) {
                                    input_node.grad = Some(sum.into());
                                }
                            } else {
                                input_node.grad = Some(node_grad.clone());
                            }
                        }
                    }
                }
                "multiply"
                    // For element-wise multiplication, input_grad = output_grad * other_input
                    if inputs.len() == 2 => {
                        let (a, b) = (&inputs[0], &inputs[1]);

                        // Compute grad_a = grad_out * b
                        if a.requiresgrad() {
                            let b_value = b.value();
                            if let Ok(grad_a) = multiply(node_grad.as_ref(), b_value.as_ref()) {
                                let mut a_node = a.node.borrow_mut();
                                if let Some(a_grad) = &a_node.grad {
                                    // Accumulate gradients
                                    if let Ok(sum) = add(a_grad.as_ref(), grad_a.as_ref()) {
                                        a_node.grad = Some(box_to_rc_array_protocol(sum));
                                    }
                                } else {
                                    a_node.grad = Some(box_to_rc_array_protocol(grad_a));
                                }
                            }
                        }

                        // Compute grad_b = grad_out * a
                        if b.requiresgrad() {
                            let a_value = a.value();
                            if let Ok(grad_b) = multiply(node_grad.as_ref(), a_value.as_ref()) {
                                let mut b_node = b.node.borrow_mut();
                                if let Some(b_grad) = &b_node.grad {
                                    // Accumulate gradients
                                    if let Ok(sum) = add(b_grad.as_ref(), grad_b.as_ref()) {
                                        b_node.grad = Some(box_to_rc_array_protocol(sum));
                                    }
                                } else {
                                    b_node.grad = Some(box_to_rc_array_protocol(grad_b));
                                }
                            }
                        }
                    }
                "matmul"
                    // For matrix multiplication, the gradients are more complex:
                    // grad_a = grad_out @ b.T
                    // grad_b = a.T @ grad_out
                    if inputs.len() == 2 => {
                        let (a, b) = (&inputs[0], &inputs[1]);

                        // Compute grad_a = grad_out @ b.T
                        if a.requiresgrad() {
                            if let (Some(b_array), Some(grad_out_array)) = (
                                b.value()
                                    .as_any()
                                    .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                                node_grad
                                    .as_any()
                                    .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                            ) {
                                let b_array_val = b_array.as_array();
                                let grad_out_array_val = grad_out_array.as_array();

                                // Transpose b: b_t = b.t()
                                let b_t = b_array_val.t();

                                // Matrix multiplication: grad_a = grad_out @ b_t
                                // Convert to Array2 for more deterministic dot behavior
                                let grad_outshape = grad_out_array_val.shape();
                                let grad_out_rows = grad_outshape[0];
                                let grad_out_cols = if grad_outshape.len() > 1 {
                                    grad_outshape.iter().skip(1).product()
                                } else {
                                    1
                                };
                                let grad_out_2d = grad_out_array_val
                                    .clone()
                                    .into_shape_with_order((grad_out_rows, grad_out_cols))
                                    .expect("Operation failed");

                                let b_tshape = b_t.shape();
                                let b_t_rows = b_tshape[0];
                                let b_t_cols = if b_tshape.len() > 1 {
                                    b_tshape.iter().skip(1).product()
                                } else {
                                    1
                                };
                                let b_t_2d = b_t
                                    .clone()
                                    .into_shape_with_order((b_t_rows, b_t_cols))
                                    .expect("Operation failed");

                                // Now use dot with 2D arrays
                                let grad_a_val = grad_out_2d.dot(&b_t_2d);

                                // Convert back to IxDyn for consistency
                                let grad_a_dyn = grad_a_val.into_dyn();
                                let grad_a = NdarrayWrapper::new(grad_a_dyn);

                                // Update a's gradient
                                let mut a_node = a.node.borrow_mut();
                                if let Some(a_grad) = &a_node.grad {
                                    if let (Some(a_grad_array), Some(grad_a_array)) = (
                                        a_grad
                                            .as_any()
                                            .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                                        grad_a
                                            .as_any()
                                            .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                                    ) {
                                        // Accumulate gradients: a_grad += grad_a
                                        let sum = a_grad_array.as_array() + grad_a_array.as_array();
                                        a_node.grad = Some(Rc::new(NdarrayWrapper::new(sum)));
                                    }
                                } else {
                                    // Use Box<dyn ArrayProtocol> and convert to Rc
                                    a_node.grad = Some(Rc::new(grad_a));
                                }
                            }
                        }

                        // Compute grad_b = a.T @ grad_out
                        if b.requiresgrad() {
                            if let (Some(a_array), Some(grad_out_array)) = (
                                a.value()
                                    .as_any()
                                    .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                                node_grad
                                    .as_any()
                                    .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                            ) {
                                let a_array_val = a_array.as_array();
                                let grad_out_array_val = grad_out_array.as_array();

                                // Transpose a: a_t = a.t()
                                let a_t = a_array_val.t();

                                // Matrix multiplication: grad_b = a_t @ grad_out
                                // Convert to Array2 for more deterministic dot behavior
                                let grad_outshape = grad_out_array_val.shape();
                                let grad_out_rows = grad_outshape[0];
                                let grad_out_cols = if grad_outshape.len() > 1 {
                                    grad_outshape.iter().skip(1).product()
                                } else {
                                    1
                                };
                                let grad_out_2d = grad_out_array_val
                                    .clone()
                                    .into_shape_with_order((grad_out_rows, grad_out_cols))
                                    .expect("Operation failed");

                                let a_tshape = a_t.shape();
                                let a_t_rows = a_tshape[0];
                                let a_t_cols = if a_tshape.len() > 1 {
                                    a_tshape.iter().skip(1).product()
                                } else {
                                    1
                                };
                                let a_t_2d = a_t
                                    .clone()
                                    .into_shape_with_order((a_t_rows, a_t_cols))
                                    .expect("Operation failed");

                                // Now use dot with 2D arrays
                                let grad_b_val = a_t_2d.dot(&grad_out_2d);

                                // Convert back to IxDyn for consistency
                                let grad_b_dyn = grad_b_val.into_dyn();
                                let grad_b = NdarrayWrapper::new(grad_b_dyn);

                                // Update b's gradient
                                let mut b_node = b.node.borrow_mut();
                                if let Some(b_grad) = &b_node.grad {
                                    if let (Some(b_grad_array), Some(grad_b_array)) = (
                                        b_grad
                                            .as_any()
                                            .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                                        grad_b
                                            .as_any()
                                            .downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
                                    ) {
                                        // Accumulate gradients: b_grad += grad_b
                                        let sum = b_grad_array.as_array() + grad_b_array.as_array();
                                        b_node.grad = Some(Rc::new(NdarrayWrapper::new(sum)));
                                    }
                                } else {
                                    // Use Box<dyn ArrayProtocol> and convert to Rc
                                    b_node.grad = Some(Rc::new(grad_b));
                                }
                            }
                        }
                    }
                "subtract"
                    // For subtraction: a - b, grad_a = grad_out, grad_b = -grad_out
                    if inputs.len() == 2 => {
                        let (a, b) = (&inputs[0], &inputs[1]);

                        // Compute grad_a = grad_out
                        if a.requiresgrad() {
                            let mut a_node = a.node.borrow_mut();
                            if let Some(a_grad) = &a_node.grad {
                                // Accumulate gradients
                                if let Ok(sum) = add(a_grad.as_ref(), node_grad.as_ref()) {
                                    a_node.grad = Some(box_to_rc_array_protocol(sum));
                                }
                            } else {
                                a_node.grad = Some(node_grad.clone());
                            }
                        }

                        // Compute grad_b = -grad_out
                        if b.requiresgrad() {
                            if let Ok(neg_grad) = multiply_by_scalar(node_grad.as_ref(), -1.0) {
                                let mut b_node = b.node.borrow_mut();
                                if let Some(b_grad) = &b_node.grad {
                                    // Accumulate gradients
                                    if let Ok(sum) = add(b_grad.as_ref(), neg_grad.as_ref()) {
                                        b_node.grad = Some(box_to_rc_array_protocol(sum));
                                    }
                                } else {
                                    b_node.grad = Some(box_to_rc_array_protocol(neg_grad));
                                }
                            }
                        }
                    }
                "divide"
                    // For division: a / b, grad_a = grad_out / b, grad_b = -grad_out * a / b^2
                    if inputs.len() == 2 => {
                        let (a, b) = (&inputs[0], &inputs[1]);

                        // Compute grad_a = grad_out / b
                        if a.requiresgrad() {
                            let b_value = b.value();
                            if let Ok(grad_a) = divide(node_grad.as_ref(), b_value.as_ref()) {
                                let mut a_node = a.node.borrow_mut();
                                if let Some(a_grad) = &a_node.grad {
                                    // Accumulate gradients
                                    if let Ok(sum) = add(a_grad.as_ref(), grad_a.as_ref()) {
                                        a_node.grad = Some(box_to_rc_array_protocol(sum));
                                    }
                                } else {
                                    a_node.grad = Some(box_to_rc_array_protocol(grad_a));
                                }
                            }
                        }

                        // Compute grad_b = -grad_out * a / b^2
                        if b.requiresgrad() {
                            let a_value = a.value();
                            let b_value = b.value();

                            // Compute b^2
                            if let Ok(b_squared) = multiply(b_value.as_ref(), b_value.as_ref()) {
                                // Compute grad_out * a
                                if let Ok(grad_times_a) =
                                    multiply(node_grad.as_ref(), a_value.as_ref())
                                {
                                    // Compute grad_out * a / b^2
                                    if let Ok(div_result) =
                                        divide(grad_times_a.as_ref(), b_squared.as_ref())
                                    {
                                        // Negate: -grad_out * a / b^2
                                        if let Ok(grad_b) =
                                            multiply_by_scalar(div_result.as_ref(), -1.0)
                                        {
                                            let mut b_node = b.node.borrow_mut();
                                            if let Some(b_grad) = &b_node.grad {
                                                // Accumulate gradients
                                                if let Ok(sum) =
                                                    add(b_grad.as_ref(), grad_b.as_ref())
                                                {
                                                    b_node.grad =
                                                        Some(box_to_rc_array_protocol(sum));
                                                }
                                            } else {
                                                b_node.grad =
                                                    Some(box_to_rc_array_protocol(grad_b));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                "sigmoid"
                    // For sigmoid: grad_input = grad_out * sigmoid * (1 - sigmoid)
                    if inputs.len() == 1 => {
                        let input = &inputs[0];

                        if input.requiresgrad() {
                            // Get the output value (sigmoid result)
                            let sigmoid_value = node.value();

                            // Compute 1 - sigmoid
                            if let Ok(ones) = ones_like(sigmoid_value.as_ref()) {
                                if let Ok(one_minus_sigmoid) =
                                    subtract(ones.as_ref(), sigmoid_value.as_ref())
                                {
                                    // Compute sigmoid * (1 - sigmoid)
                                    if let Ok(sigmoid_deriv) =
                                        multiply(sigmoid_value.as_ref(), one_minus_sigmoid.as_ref())
                                    {
                                        // Compute grad_out * sigmoid * (1 - sigmoid)
                                        if let Ok(grad_input) =
                                            multiply(node_grad.as_ref(), sigmoid_deriv.as_ref())
                                        {
                                            let mut input_node = input.node.borrow_mut();
                                            if let Some(input_grad) = &input_node.grad {
                                                // Accumulate gradients
                                                if let Ok(sum) =
                                                    add(input_grad.as_ref(), grad_input.as_ref())
                                                {
                                                    input_node.grad =
                                                        Some(box_to_rc_array_protocol(sum));
                                                }
                                            } else {
                                                input_node.grad =
                                                    Some(box_to_rc_array_protocol(grad_input));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                "mean"
                    // For mean: grad_input = grad_out / n (where n is the number of elements)
                    if inputs.len() == 1 => {
                        let input = &inputs[0];

                        if input.requiresgrad() {
                            // Get the number of elements
                            let input_value = input.value();
                            if let Some(inputarray) = input_value
                                .as_any()
                                .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
                            {
                                let n_elements = inputarray.as_array().len() as f64;

                                // Compute grad_input = grad_out / n
                                if let Ok(grad_input) =
                                    multiply_by_scalar(node_grad.as_ref(), 1.0 / n_elements)
                                {
                                    // Broadcast the gradient to match input shape
                                    if let Ok(broadcasted_grad) = broadcast_to(
                                        grad_input.as_ref(),
                                        inputarray.as_array().shape(),
                                    ) {
                                        let mut input_node = input.node.borrow_mut();
                                        if let Some(input_grad) = &input_node.grad {
                                            // Accumulate gradients
                                            if let Ok(sum) =
                                                add(input_grad.as_ref(), broadcasted_grad.as_ref())
                                            {
                                                input_node.grad =
                                                    Some(box_to_rc_array_protocol(sum));
                                            }
                                        } else {
                                            input_node.grad =
                                                Some(box_to_rc_array_protocol(broadcasted_grad));
                                        }
                                    }
                                }
                            }
                        }
                    }
                _ => {
                    // Other operations would be implemented here
                }
            }
        }

        Ok(())
    }

    /// Detach this tensor from the computation graph.
    pub fn detach(&self) -> Self {
        GradientTensor::new(self.value(), false)
    }
}

/// Implementations of gradient-aware operations
/// Addition operation with gradient tracking.
#[allow(dead_code)]
pub fn grad_add(a: &GradientTensor, b: &GradientTensor) -> CoreResult<GradientTensor> {
    let a_value = a.value();
    let b_value = b.value();

    // Perform addition
    let result = add(a_value.as_ref(), b_value.as_ref())?;

    // Create a new gradient tensor - explicitly convert Box to Rc
    let result_rc: Rc<dyn ArrayProtocol> = box_to_rc_array_protocol(result);
    Ok(GradientTensor::from_op(
        result_rc,
        "add".to_string(),
        vec![a.clone(), b.clone()],
    ))
}

/// Element-wise multiplication with gradient tracking.
#[allow(dead_code)]
pub fn grad_multiply(a: &GradientTensor, b: &GradientTensor) -> CoreResult<GradientTensor> {
    let a_value = a.value();
    let b_value = b.value();

    // Perform multiplication
    let result = multiply(a_value.as_ref(), b_value.as_ref())?;

    // Create a new gradient tensor - explicitly convert Box to Rc
    let result_rc: Rc<dyn ArrayProtocol> = box_to_rc_array_protocol(result);
    Ok(GradientTensor::from_op(
        result_rc,
        "multiply".to_string(),
        vec![a.clone(), b.clone()],
    ))
}

/// Matrix multiplication with gradient tracking.
#[allow(dead_code)]
pub fn grad_matmul(a: &GradientTensor, b: &GradientTensor) -> CoreResult<GradientTensor> {
    let a_value = a.value();
    let b_value = b.value();

    // Perform matrix multiplication
    let result = matmul(a_value.as_ref(), b_value.as_ref())?;

    // Create a new gradient tensor - explicitly convert Box to Rc
    let result_rc: Rc<dyn ArrayProtocol> = box_to_rc_array_protocol(result);
    Ok(GradientTensor::from_op(
        result_rc,
        "matmul".to_string(),
        vec![a.clone(), b.clone()],
    ))
}

/// Subtraction with gradient tracking.
#[allow(dead_code)]
pub fn grad_subtract(a: &GradientTensor, b: &GradientTensor) -> CoreResult<GradientTensor> {
    let a_value = a.value();
    let b_value = b.value();

    // Perform subtraction
    let result = subtract(a_value.as_ref(), b_value.as_ref())?;

    // Create a new gradient tensor - explicitly convert Box to Rc
    let result_rc: Rc<dyn ArrayProtocol> = box_to_rc_array_protocol(result);
    Ok(GradientTensor::from_op(
        result_rc,
        "subtract".to_string(),
        vec![a.clone(), b.clone()],
    ))
}

/// Division with gradient tracking.
#[allow(dead_code)]
pub fn grad_divide(a: &GradientTensor, b: &GradientTensor) -> CoreResult<GradientTensor> {
    let a_value = a.value();
    let b_value = b.value();

    // Perform division
    let result = divide(a_value.as_ref(), b_value.as_ref())?;

    // Create a new gradient tensor - explicitly convert Box to Rc
    let result_rc: Rc<dyn ArrayProtocol> = box_to_rc_array_protocol(result);
    Ok(GradientTensor::from_op(
        result_rc,
        "divide".to_string(),
        vec![a.clone(), b.clone()],
    ))
}

/// Sigmoid activation with gradient tracking.
#[allow(dead_code)]
pub fn grad_sigmoid(a: &GradientTensor) -> CoreResult<GradientTensor> {
    let a_value = a.value();

    // Perform sigmoid: 1 / (1 + exp(-x))
    if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
    {
        let array = a_array.as_array();
        let result = array.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "sigmoid".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f32, IxDyn>>()
    {
        let array = a_array.as_array();
        let result = array.mapv(|x| 1.0f32 / (1.0f32 + (-x).exp()));
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "sigmoid".to_string(),
            vec![a.clone()],
        ))
    } else {
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "sigmoid not implemented for this array type".to_string(),
        )))
    }
}

/// Mean reduction with gradient tracking.
#[allow(dead_code)]
pub fn grad_mean(a: &GradientTensor) -> CoreResult<GradientTensor> {
    let a_value = a.value();

    // Perform mean reduction
    if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f64, IxDyn>>()
    {
        let array = a_array.as_array();
        let mean_value = array.mean_or(0.0);
        let result = ArrayD::<f64>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<f32, IxDyn>>()
    {
        let array = a_array.as_array();
        let mean_value = array.mean_or(0.0f32);
        let result = ArrayD::<f32>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<i32, IxDyn>>()
    {
        let array = a_array.as_array();
        let mean_value = if array.is_empty() {
            0.0f64
        } else {
            array.iter().map(|&x| x as f64).sum::<f64>() / array.len() as f64
        };
        let result = ArrayD::<f64>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<i64, IxDyn>>()
    {
        let array = a_array.as_array();
        let mean_value = if array.is_empty() {
            0.0f64
        } else {
            array.iter().map(|&x| x as f64).sum::<f64>() / array.len() as f64
        };
        let result = ArrayD::<f64>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value.as_any().downcast_ref::<NdarrayWrapper<u8, IxDyn>>() {
        let array = a_array.as_array();
        let mean_value = if array.is_empty() {
            0.0f64
        } else {
            array.iter().map(|&x| x as f64).sum::<f64>() / array.len() as f64
        };
        let result = ArrayD::<f64>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<u16, IxDyn>>()
    {
        let array = a_array.as_array();
        let mean_value = if array.is_empty() {
            0.0f64
        } else {
            array.iter().map(|&x| x as f64).sum::<f64>() / array.len() as f64
        };
        let result = ArrayD::<f64>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<u32, IxDyn>>()
    {
        let array = a_array.as_array();
        let mean_value = if array.is_empty() {
            0.0f64
        } else {
            array.iter().map(|&x| x as f64).sum::<f64>() / array.len() as f64
        };
        let result = ArrayD::<f64>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else if let Some(a_array) = a_value
        .as_any()
        .downcast_ref::<NdarrayWrapper<u64, IxDyn>>()
    {
        let array = a_array.as_array();
        let mean_value = if array.is_empty() {
            0.0f64
        } else {
            array.iter().map(|&x| x as f64).sum::<f64>() / array.len() as f64
        };
        let result = ArrayD::<f64>::from_elem(IxDyn(&[1]), mean_value);
        let result_wrapped = NdarrayWrapper::new(result);
        let result_rc: Rc<dyn ArrayProtocol> = Rc::new(result_wrapped);
        Ok(GradientTensor::from_op(
            result_rc,
            "mean".to_string(),
            vec![a.clone()],
        ))
    } else {
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "mean not implemented for this array type".to_string(),
        )))
    }
}

/// Gradient-aware variable that can be optimized.
pub struct Variable {
    /// The gradient tensor.
    tensor: GradientTensor,

    /// Name for the variable.
    name: String,
}

impl Variable {
    /// Create a new variable from an array.
    pub fn new<T, D>(name: &str, array: Array<T, D>) -> Self
    where
        T: Clone + Send + Sync + 'static,
        D: Dimension + Send + Sync + 'static,
    {
        let tensor = GradientTensor::from_array(array, true);
        Self {
            tensor,
            name: name.to_string(),
        }
    }

    /// Get the gradient tensor.
    pub const fn tensor(&self) -> &GradientTensor {
        &self.tensor
    }

    /// Get the value of the variable.
    pub fn value(&self) -> Rc<dyn ArrayProtocol> {
        self.tensor.value()
    }

    /// Get the gradient of the variable.
    pub fn grad_2(&self) -> Option<Rc<dyn ArrayProtocol>> {
        self.tensor.grad_2()
    }

    /// Get the name of the variable.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the gradient of the variable
    pub fn set_gradient(&mut self, gradient: Box<dyn ArrayProtocol>) -> CoreResult<()> {
        // Convert Box to Rc
        let gradient_rc = self.box_to_rc(gradient);

        // Set gradient on the tensor node
        self.tensor.node.borrow_mut().grad = Some(gradient_rc);
        Ok(())
    }

    /// Set the value of the variable (for updating during optimization).
    pub fn set_value(&mut self, newvalue: Box<dyn ArrayProtocol>) {
        let newvalue_rc = self.box_to_rc(newvalue);
        self.tensor.set_value(newvalue_rc);
    }

    /// Helper to convert Box<dyn ArrayProtocol> to Rc<dyn ArrayProtocol>
    fn box_to_rc(&self, boxed: Box<dyn ArrayProtocol>) -> Rc<dyn ArrayProtocol> {
        // See the module-level `boxed_to_rc`: `Rc::from` preserves the exact
        // concrete type via std's `Rc<T>: From<Box<T>>` impl, rather than
        // downcast-enumerating only `NdarrayWrapper<f64, IxDyn>` and silently
        // substituting a fabricated 1x1 zero placeholder for anything else
        // (e.g. an update or gradient with a different dimension type).
        Rc::from(boxed)
    }
}

/// Trait for optimizers that update variables.
pub trait Optimizer {
    /// Step the optimizer to update variables.
    fn step(&mut self) -> CoreResult<()>;

    /// Zero all gradients.
    fn zero_grad(&mut self);

    /// Add a variable to the optimizer.
    fn add_variable(&mut self, var: Variable);

    /// Get all variables managed by the optimizer.
    fn variables(&self) -> &[Variable];

    /// Accumulate gradients for momentum-based optimizers
    fn accumulate_gradients(&mut self, gradients: &GradientDict) -> CoreResult<()> {
        // Default implementation: update variable gradients
        for (param_name, gradient) in gradients.iter() {
            // Find the variable with matching name and update its gradient
            for var in self.variables_mut() {
                if var.name() == param_name {
                    var.set_gradient(gradient.clone())?;
                    break;
                }
            }
        }
        Ok(())
    }

    /// Get mutable reference to variables (for default implementation)
    fn variables_mut(&mut self) -> &mut [Variable] {
        // Default implementation returns empty slice
        // Implementations should override this if they support accumulate_gradients
        &mut []
    }
}

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    /// Variables to optimize.
    variables: Vec<Variable>,

    /// Learning rate.
    learningrate: f64,

    /// Momentum factor.
    momentum: f64,

    /// Velocity for momentum.
    velocity: Vec<Option<Box<dyn ArrayProtocol>>>,
}

impl SGD {
    /// Create a new SGD optimizer.
    pub fn new(learningrate: f64, momentum: Option<f64>) -> Self {
        Self {
            variables: Vec::new(),
            learningrate,
            momentum: momentum.unwrap_or(0.0),
            velocity: Vec::new(),
        }
    }

    /// Set the learning rate.
    pub fn set_learningrate(&mut self, learningrate: f64) {
        self.learningrate = learningrate;
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> CoreResult<()> {
        for (i, var) in self.variables.iter_mut().enumerate() {
            if let Some(grad) = var.grad_2() {
                let var_value = var.value();

                // Compute update with momentum
                let update = if self.momentum > 0.0 {
                    if i >= self.velocity.len() {
                        self.velocity.resize_with(i + 1, || None);
                    }

                    if let Some(vel) = &self.velocity[i] {
                        // v = momentum * v + lr * grad
                        let scaled_grad = multiply_by_scalar(grad.as_ref(), self.learningrate)?;
                        let scaled_vel = multiply_by_scalar(vel.as_ref(), self.momentum)?;
                        let update = add(scaled_vel.as_ref(), scaled_grad.as_ref())?;
                        self.velocity[i] = Some(update.clone());
                        update
                    } else {
                        // First iteration, just use lr * grad
                        let update = multiply_by_scalar(grad.as_ref(), self.learningrate)?;
                        self.velocity[i] = Some(update.clone());
                        update
                    }
                } else {
                    // No momentum, just use lr * grad
                    multiply_by_scalar(grad.as_ref(), self.learningrate)?
                };

                // Update variable: var = var - update
                let updated_value = subtract_arrays(var_value.as_ref(), update.as_ref())?;
                var.set_value(updated_value);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for var in &self.variables {
            var.tensor.node.borrow_mut().grad = None;
        }
    }

    fn add_variable(&mut self, var: Variable) {
        self.variables.push(var);
        self.velocity.push(None);
    }

    fn variables(&self) -> &[Variable] {
        &self.variables
    }

    fn variables_mut(&mut self) -> &mut [Variable] {
        &mut self.variables
    }
}

/// Adam optimizer.
pub struct Adam {
    /// Variables to optimize.
    variables: Vec<Variable>,

    /// Learning rate.
    learningrate: f64,

    /// Beta1 parameter (for first moment).
    beta1: f64,

    /// Beta2 parameter (for second moment).
    beta2: f64,

    /// Epsilon for numerical stability.
    epsilon: f64,

    /// First moment estimates.
    m: Vec<Option<Box<dyn ArrayProtocol>>>,

    /// Second moment estimates.
    v: Vec<Option<Box<dyn ArrayProtocol>>>,

    /// Iteration counter.
    t: usize,
}

impl Adam {
    /// Create a new Adam optimizer.
    pub fn new(
        learningrate: f64,
        beta1: Option<f64>,
        beta2: Option<f64>,
        epsilon: Option<f64>,
    ) -> Self {
        Self {
            variables: Vec::new(),
            learningrate,
            beta1: beta1.unwrap_or(0.9),
            beta2: beta2.unwrap_or(0.999),
            epsilon: epsilon.unwrap_or(1e-8),
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> CoreResult<()> {
        self.t += 1;

        for (i, var) in self.variables.iter_mut().enumerate() {
            if let Some(grad) = var.grad_2() {
                let var_value = var.value();

                // Ensure we have space for state variables
                if i >= self.m.len() {
                    self.m.resize_with(i + 1, || None);
                    self.v.resize_with(i + 1, || None);
                }

                // Update biased first moment estimate
                let m = if let Some(m_prev) = &self.m[i] {
                    // m = beta1 * m + (1 - beta1) * grad
                    let scaled_m = multiply_by_scalar(m_prev.as_ref(), self.beta1)?;
                    let scaled_grad = multiply_by_scalar(grad.as_ref(), 1.0 - self.beta1)?;
                    add(scaled_m.as_ref(), scaled_grad.as_ref())?
                } else {
                    // First iteration, just use (1 - beta1) * grad
                    multiply_by_scalar(grad.as_ref(), 1.0 - self.beta1)?
                };

                // Update biased second moment estimate
                let v = if let Some(v_prev) = &self.v[i] {
                    // v = beta2 * v + (1 - beta2) * grad^2
                    let scaled_v = multiply_by_scalar(v_prev.as_ref(), self.beta2)?;
                    let grad_squared = multiply(grad.as_ref(), grad.as_ref())?;
                    let scaled_grad_sq =
                        multiply_by_scalar(grad_squared.as_ref(), 1.0 - self.beta2)?;
                    add(scaled_v.as_ref(), scaled_grad_sq.as_ref())?
                } else {
                    // First iteration, just use (1 - beta2) * grad^2
                    let grad_squared = multiply(grad.as_ref(), grad.as_ref())?;
                    multiply_by_scalar(grad_squared.as_ref(), 1.0 - self.beta2)?
                };

                // Store state variables - no need to convert since we're already using Box
                self.m[i] = Some(m.clone());
                self.v[i] = Some(v.clone());

                // Compute bias-corrected estimates
                let m_hat =
                    multiply_by_scalar(m.as_ref(), 1.0 / (1.0 - self.beta1.powi(self.t as i32)))?;
                let v_hat =
                    multiply_by_scalar(v.as_ref(), 1.0 / (1.0 - self.beta2.powi(self.t as i32)))?;

                // Compute update: lr * m_hat / (sqrt(v_hat) + epsilon)
                let v_hat_sqrt = sqrt(v_hat.as_ref())?;
                let v_hat_sqrt_eps = add_scalar(v_hat_sqrt.as_ref(), self.epsilon)?;
                let update_dir = divide(m_hat.as_ref(), v_hat_sqrt_eps.as_ref())?;
                let update = multiply_by_scalar(update_dir.as_ref(), self.learningrate)?;

                // Update variable: var = var - update
                let updated_value = subtract_arrays(var_value.as_ref(), update.as_ref())?;
                var.set_value(updated_value);
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for var in &self.variables {
            var.tensor.node.borrow_mut().grad = None;
        }
    }

    fn add_variable(&mut self, var: Variable) {
        self.variables.push(var);
        self.m.push(None);
        self.v.push(None);
    }

    fn variables(&self) -> &[Variable] {
        &self.variables
    }

    fn variables_mut(&mut self) -> &mut [Variable] {
        &mut self.variables
    }
}

// Helper functions for optimizers

/// Extract an owned, dynamically-dimensioned copy of the array wrapped by an
/// `ArrayProtocol` value, regardless of whether it is concretely stored as
/// `Ix1`, `Ix2`, or `IxDyn`. Callers that only special-case `IxDyn` (as this
/// file's optimizer helpers historically did) silently fail on the very
/// common `Array2`/`Array1`-backed case — e.g. a `Variable` built from a 2-D
/// weight matrix — even though the arithmetic itself has no real dimension
/// restriction.
fn downcast_to_ixdyn<T: Clone + Send + Sync + 'static>(a: &dyn ArrayProtocol) -> Option<ArrayD<T>> {
    if let Some(w) = a.as_any().downcast_ref::<NdarrayWrapper<T, IxDyn>>() {
        return Some(w.as_array().clone());
    }
    if let Some(w) = a.as_any().downcast_ref::<NdarrayWrapper<T, Ix2>>() {
        return Some(w.as_array().clone().into_dyn());
    }
    if let Some(w) = a.as_any().downcast_ref::<NdarrayWrapper<T, Ix1>>() {
        return Some(w.as_array().clone().into_dyn());
    }
    None
}

/// Multiply an array by a scalar.
fn multiply_by_scalar(a: &dyn ArrayProtocol, scalar: f64) -> CoreResult<Box<dyn ArrayProtocol>> {
    if let Some(inputarray) = downcast_to_ixdyn::<f64>(a) {
        let result = inputarray.mapv(|x| x * scalar);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(inputarray) = downcast_to_ixdyn::<f32>(a) {
        let result = inputarray.mapv(|x| x * scalar as f32);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(inputarray) = downcast_to_ixdyn::<i32>(a) {
        let result = inputarray.mapv(|x| (x as f64 * scalar) as i32);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(inputarray) = downcast_to_ixdyn::<i64>(a) {
        let result = inputarray.mapv(|x| (x as f64 * scalar) as i64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(inputarray) = downcast_to_ixdyn::<u8>(a) {
        let result = inputarray.mapv(|x| (x as f64 * scalar) as u8);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(inputarray) = downcast_to_ixdyn::<u16>(a) {
        let result = inputarray.mapv(|x| (x as f64 * scalar) as u16);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(inputarray) = downcast_to_ixdyn::<u32>(a) {
        let result = inputarray.mapv(|x| (x as f64 * scalar) as u32);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(inputarray) = downcast_to_ixdyn::<u64>(a) {
        let result = inputarray.mapv(|x| (x as f64 * scalar) as u64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else {
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "multiply_by_scalar not implemented for this array type".to_string(),
        )))
    }
}

/// Subtract one array from another, returning a new array.
fn subtract_arrays(
    a: &dyn ArrayProtocol,
    b: &dyn ArrayProtocol,
) -> CoreResult<Box<dyn ArrayProtocol>> {
    // Perform element-wise subtraction and return a new array
    if let (Some(a_arr), Some(b_arr)) = (downcast_to_ixdyn::<f64>(a), downcast_to_ixdyn::<f64>(b)) {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_arr), Some(b_arr)) =
        (downcast_to_ixdyn::<f32>(a), downcast_to_ixdyn::<f32>(b))
    {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_arr), Some(b_arr)) =
        (downcast_to_ixdyn::<i32>(a), downcast_to_ixdyn::<i32>(b))
    {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_arr), Some(b_arr)) =
        (downcast_to_ixdyn::<i64>(a), downcast_to_ixdyn::<i64>(b))
    {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_arr), Some(b_arr)) =
        (downcast_to_ixdyn::<u8>(a), downcast_to_ixdyn::<u8>(b))
    {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_arr), Some(b_arr)) =
        (downcast_to_ixdyn::<u16>(a), downcast_to_ixdyn::<u16>(b))
    {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_arr), Some(b_arr)) =
        (downcast_to_ixdyn::<u32>(a), downcast_to_ixdyn::<u32>(b))
    {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_arr), Some(b_arr)) =
        (downcast_to_ixdyn::<u64>(a), downcast_to_ixdyn::<u64>(b))
    {
        Ok(Box::new(NdarrayWrapper::new(a_arr - b_arr)) as Box<dyn ArrayProtocol>)
    } else {
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "subtract_arrays not implemented for these array types".to_string(),
        )))
    }
}

/// Element-wise square root.
fn sqrt(a: &dyn ArrayProtocol) -> CoreResult<Box<dyn ArrayProtocol>> {
    if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x.sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x.sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| (x as f64).sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| (x as f64).sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u8, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| (x as f64).sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u16, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| (x as f64).sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u32, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| (x as f64).sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u64, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| (x as f64).sqrt());
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else {
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "sqrt not implemented for this array type".to_string(),
        )))
    }
}

/// Add a scalar to an array.
fn add_scalar(a: &dyn ArrayProtocol, scalar: f64) -> CoreResult<Box<dyn ArrayProtocol>> {
    if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar as f32);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar as i32);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar as i64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u8, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar as u8);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u16, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar as u16);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u32, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar as u32);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let Some(a_array) = a.as_any().downcast_ref::<NdarrayWrapper<u64, IxDyn>>() {
        let result = a_array.as_array().mapv(|x| x + scalar as u64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else {
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "add_scalar not implemented for this array type".to_string(),
        )))
    }
}

/// Element-wise division.
fn divide(a: &dyn ArrayProtocol, b: &dyn ArrayProtocol) -> CoreResult<Box<dyn ArrayProtocol>> {
    if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<f64, IxDyn>>(),
    ) {
        let result = a_array.as_array() / b_array.as_array();
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<f32, IxDyn>>(),
    ) {
        let result = a_array.as_array() / b_array.as_array();
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<i32, IxDyn>>(),
    ) {
        let result = ::ndarray::Zip::from(a_array.as_array())
            .and(b_array.as_array())
            .map_collect(|&av, &bv| av as f64 / bv as f64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<i64, IxDyn>>(),
    ) {
        let result = ::ndarray::Zip::from(a_array.as_array())
            .and(b_array.as_array())
            .map_collect(|&av, &bv| av as f64 / bv as f64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<u8, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<u8, IxDyn>>(),
    ) {
        let result = ::ndarray::Zip::from(a_array.as_array())
            .and(b_array.as_array())
            .map_collect(|&av, &bv| av as f64 / bv as f64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<u16, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<u16, IxDyn>>(),
    ) {
        let result = ::ndarray::Zip::from(a_array.as_array())
            .and(b_array.as_array())
            .map_collect(|&av, &bv| av as f64 / bv as f64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<u32, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<u32, IxDyn>>(),
    ) {
        let result = ::ndarray::Zip::from(a_array.as_array())
            .and(b_array.as_array())
            .map_collect(|&av, &bv| av as f64 / bv as f64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else if let (Some(a_array), Some(b_array)) = (
        a.as_any().downcast_ref::<NdarrayWrapper<u64, IxDyn>>(),
        b.as_any().downcast_ref::<NdarrayWrapper<u64, IxDyn>>(),
    ) {
        let result = ::ndarray::Zip::from(a_array.as_array())
            .and(b_array.as_array())
            .map_collect(|&av, &bv| av as f64 / bv as f64);
        Ok(Box::new(NdarrayWrapper::new(result)) as Box<dyn ArrayProtocol>)
    } else {
        Err(CoreError::NotImplementedError(ErrorContext::new(
            "divide not implemented for these array types".to_string(),
        )))
    }
}

#[cfg(test)]
#[path = "grad_tests.rs"]
mod tests;
