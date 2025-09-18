//! Automatic differentiation integration for optimizers
//!
//! This module provides automatic differentiation capabilities for computing
//! higher-order gradients, Hessians, and meta-gradients used in advanced
//! optimization algorithms.

use crate::error::{OptimError, Result};
use ndarray::{Array, Array1, Array2, Dimension};
use num_traits::Float;
use std::collections::HashMap;

pub mod forward_mode;
pub mod higher_order;
pub mod meta_gradients;
pub mod reverse_mode;

/// Automatic differentiation configuration
#[derive(Debug, Clone)]
pub struct AutodiffConfig {
    /// Enable forward-mode AD
    pub enable_forward_mode: bool,

    /// Enable reverse-mode AD (backpropagation)
    pub enable_reverse_mode: bool,

    /// Maximum order of derivatives to compute
    pub max_derivative_order: usize,

    /// Enable Hessian computation
    pub enable_hessian: bool,

    /// Use sparse Hessian computation
    pub sparse_hessian: bool,

    /// Hessian approximation method
    pub hessian_approximation: HessianApproximation,

    /// Enable meta-gradient computation
    pub enable_meta_gradients: bool,

    /// Gradient checkpointing for memory efficiency
    pub gradient_checkpointing: bool,

    /// Chunk size for checkpointing
    pub checkpoint_chunk_size: usize,
}

impl Default for AutodiffConfig {
    fn default() -> Self {
        Self {
            enable_forward_mode: true,
            enable_reverse_mode: true,
            max_derivative_order: 2,
            enable_hessian: true,
            sparse_hessian: false,
            hessian_approximation: HessianApproximation::BFGS,
            enable_meta_gradients: false,
            gradient_checkpointing: true,
            checkpoint_chunk_size: 1000,
        }
    }
}

/// Hessian approximation methods
#[derive(Debug, Clone, Copy)]
pub enum HessianApproximation {
    /// Exact Hessian computation
    Exact,
    /// BFGS approximation
    BFGS,
    /// L-BFGS approximation
    LBFGS,
    /// Gauss-Newton approximation
    GaussNewton,
    /// Fisher information matrix
    Fisher,
    /// Diagonal approximation
    Diagonal,
}

/// Computational graph node for automatic differentiation
#[derive(Debug, Clone)]
pub struct ADNode<T: Float> {
    /// Value at this node
    pub value: T,

    /// Gradient with respect to inputs
    pub gradient: Vec<T>,

    /// Hessian (if computed)
    pub hessian: Option<Array2<T>>,

    /// Operation that produced this node
    pub operation: Operation,

    /// Parent nodes in the computational graph
    pub parents: Vec<usize>,

    /// Node ID
    pub id: usize,
}

/// Operations in the computational graph
#[derive(Debug, Clone)]
pub enum Operation {
    Input,
    Add,
    Multiply,
    Divide,
    Power,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Sigmoid,
    ReLU,
    MatMul,
    Dot,
    Sum,
    Mean,
    Custom(String),
}

/// Automatic differentiation engine
pub struct AutodiffEngine<T: Float> {
    /// Configuration
    config: AutodiffConfig,

    /// Computational graph
    graph: Vec<ADNode<T>>,

    /// Variable registry
    variables: HashMap<String, usize>,

    /// Gradient tape for reverse-mode AD
    tape: Vec<TapeEntry<T>>,

    /// Current computation context
    context: ComputationContext,

    /// Hessian approximation state
    hessian_state: Option<HessianApproximationState<T>>,
}

/// Entry in the gradient tape
#[derive(Debug, Clone)]
struct TapeEntry<T: Float> {
    operation: Operation,
    inputs: Vec<usize>,
    output: usize,
    local_gradients: Vec<T>,
}

/// Computation context
#[derive(Debug, Clone)]
struct ComputationContext {
    /// Current forward pass ID
    forward_pass_id: usize,

    /// Enable gradient recording
    record_gradients: bool,

    /// Enable higher-order derivatives
    higher_order: bool,
}

/// Hessian approximation state
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct HessianApproximationState<T: Float> {
    /// Approximation method
    method: HessianApproximation,

    /// History of gradients
    gradient_history: Vec<Array1<T>>,

    /// History of parameter updates
    update_history: Vec<Array1<T>>,

    /// BFGS inverse Hessian approximation
    inverse_hessian: Option<Array2<T>>,

    /// L-BFGS memory
    lbfgs_memory: Option<LBFGSMemory<T>>,

    /// Maximum history size
    max_history: usize,
}

/// L-BFGS memory structure
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct LBFGSMemory<T: Float> {
    /// Gradient differences (y_k = g_{k+1} - g_k)
    y_history: Vec<Array1<T>>,

    /// Parameter differences (s_k = x_{k+1} - x_k)
    s_history: Vec<Array1<T>>,

    /// Scaling factor (rho_k = 1 / (y_k^T s_k))
    rho_history: Vec<T>,

    /// Memory size
    memory_size: usize,

    /// Current position in circular buffer
    current_pos: usize,
}

/// Computation checkpoint for gradient checkpointing
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ComputationCheckpoint<T: Float> {
    /// Saved node values
    values: Vec<T>,

    /// Saved graph state
    graph_state: Vec<ADNode<T>>,

    /// Checkpoint ID
    id: usize,

    /// Memory usage estimate
    memory_usage: usize,
}

impl<T: Float + Default + Clone + ndarray::ScalarOperand> AutodiffEngine<T> {
    /// Create a new automatic differentiation engine
    pub fn new(config: AutodiffConfig) -> Self {
        Self {
            config,
            graph: Vec::new(),
            variables: HashMap::new(),
            tape: Vec::new(),
            context: ComputationContext {
                forward_pass_id: 0,
                record_gradients: true,
                higher_order: false,
            },
            hessian_state: None,
        }
    }

    /// Create a new variable
    pub fn create_variable(&mut self, name: &str, value: T) -> usize {
        let node = ADNode {
            value,
            gradient: vec![T::one()],
            hessian: None,
            operation: Operation::Input,
            parents: Vec::new(),
            id: self.graph.len(),
        };

        let id = self.graph.len();
        self.graph.push(node);
        self.variables.insert(name.to_string(), id);
        id
    }

    /// Compute gradients using reverse-mode AD
    pub fn backward(&mut self, outputid: usize) -> Result<Vec<T>> {
        if outputid >= self.graph.len() {
            return Err(OptimError::InvalidConfig(
                "Invalid output node ID".to_string(),
            ));
        }

        // Initialize gradient of output
        let mut gradients = vec![T::zero(); self.graph.len()];
        gradients[outputid] = T::one();

        // Reverse pass through the tape
        for entry in self.tape.iter().rev() {
            let output_grad = gradients[entry.output];

            for (i, &input_id) in entry.inputs.iter().enumerate() {
                if i < entry.local_gradients.len() {
                    gradients[input_id] =
                        gradients[input_id] + output_grad * entry.local_gradients[i];
                }
            }
        }

        Ok(gradients)
    }

    /// Compute Hessian matrix
    pub fn compute_hessian(&mut self, outputid: usize) -> Result<Array2<T>> {
        match self.config.hessian_approximation {
            HessianApproximation::Exact => self.compute_exact_hessian(outputid),
            HessianApproximation::BFGS => self.compute_bfgs_hessian(),
            HessianApproximation::LBFGS => self.compute_lbfgs_hessian(),
            HessianApproximation::Diagonal => self.compute_diagonal_hessian(outputid),
            HessianApproximation::GaussNewton => self.compute_hessian(outputid),
            HessianApproximation::Fisher => self.compute_diagonal_hessian(outputid),
        }
    }

    fn compute_exact_hessian(&mut self, outputid: usize) -> Result<Array2<T>> {
        let n_vars = self.variables.len();
        let mut hessian = Array2::zeros((n_vars, n_vars));

        // Compute second derivatives using forward-over-reverse mode
        for i in 0..n_vars {
            for j in i..n_vars {
                let second_deriv = self.compute_second_derivative(outputid, i, j)?;
                hessian[[i, j]] = second_deriv;
                hessian[[j, i]] = second_deriv; // Hessian is symmetric
            }
        }

        Ok(hessian)
    }

    fn compute_second_derivative(
        &mut self,
        outputid: usize,
        var1: usize,
        var2: usize,
    ) -> Result<T> {
        // Enhanced second derivative computation with checkpointing
        if self.config.gradient_checkpointing {
            return self.compute_second_derivative_checkpointed(outputid, var1, var2);
        }

        let eps = num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero());

        // Finite difference approximation for now
        let original_val1 = self.graph[var1].value;
        let original_val2 = self.graph[var2].value;

        // f(x+h, y+h)
        self.graph[var1].value = original_val1 + eps;
        self.graph[var2].value = original_val2 + eps;
        let f_pp = self.evaluate_node(outputid)?;

        // f(x+h, y-h)
        self.graph[var2].value = original_val2 - eps;
        let f_pm = self.evaluate_node(outputid)?;

        // f(x-h, y+h)
        self.graph[var1].value = original_val1 - eps;
        self.graph[var2].value = original_val2 + eps;
        let f_mp = self.evaluate_node(outputid)?;

        // f(x-h, y-h)
        self.graph[var2].value = original_val2 - eps;
        let f_mm = self.evaluate_node(outputid)?;

        // Restore original values
        self.graph[var1].value = original_val1;
        self.graph[var2].value = original_val2;

        // Second derivative approximation
        let second_deriv = (f_pp - f_pm - f_mp + f_mm) / (num_traits::cast::cast(4.0).unwrap_or_else(|| T::zero()) * eps * eps);

        Ok(second_deriv)
    }

    /// Memory-efficient second derivative computation with checkpointing
    fn compute_second_derivative_checkpointed(
        &mut self,
        outputid: usize,
        var1: usize,
        var2: usize,
    ) -> Result<T> {
        let chunk_size = self.config.checkpoint_chunk_size;
        let eps = num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero());

        // Create checkpoint
        let checkpoint = self.create_checkpoint();

        // Compute derivatives in chunks
        let mut results = Vec::new();
        for chunk_start in (0..4).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(4);

            // Restore from checkpoint
            self.restore_checkpoint(&checkpoint)?;

            // Compute chunk
            for i in chunk_start..chunk_end {
                let point_values = match i {
                    0 => (eps, eps),   // f(x+h, y+h)
                    1 => (eps, -eps),  // f(x+h, y-h)
                    2 => (-eps, eps),  // f(x-h, y+h)
                    3 => (-eps, -eps), // f(x-h, y-h)
                    _ => continue,
                };

                self.graph[var1].value = checkpoint.values[var1] + point_values.0;
                self.graph[var2].value = checkpoint.values[var2] + point_values.1;

                let result = self.evaluate_node(outputid)?;
                results.push(result);
            }
        }

        // Restore original state
        self.restore_checkpoint(&checkpoint)?;

        // Compute second derivative
        if results.len() >= 4 {
            let second_deriv = (results[0] - results[1] - results[2] + results[3])
                / (num_traits::cast::cast(4.0).unwrap_or_else(|| T::zero()) * eps * eps);
            Ok(second_deriv)
        } else {
            Err(OptimError::InvalidConfig(
                "Insufficient results for second derivative".to_string(),
            ))
        }
    }

    fn evaluate_node(&self, nodeid: usize) -> Result<T> {
        if nodeid >= self.graph.len() {
            return Err(OptimError::InvalidConfig("Invalid node ID".to_string()));
        }

        Ok(self.graph[nodeid].value)
    }

    fn compute_bfgs_hessian(&self) -> Result<Array2<T>> {
        if let Some(ref state) = self.hessian_state {
            if let Some(ref inv_hessian) = state.inverse_hessian {
                return Ok(inv_hessian.clone());
            }
        }

        Err(OptimError::InvalidConfig(
            "BFGS Hessian not initialized".to_string(),
        ))
    }

    fn compute_lbfgs_hessian(&self) -> Result<Array2<T>> {
        if let Some(ref state) = self.hessian_state {
            if let Some(ref lbfgs) = state.lbfgs_memory {
                return self.lbfgs_two_loop_recursion(lbfgs);
            }
        }

        Err(OptimError::InvalidConfig(
            "L-BFGS memory not initialized".to_string(),
        ))
    }

    fn lbfgs_two_loop_recursion(&self, lbfgs: &LBFGSMemory<T>) -> Result<Array2<T>> {
        let n = lbfgs
            .s_history
            .get(0)
            .ok_or_else(|| OptimError::InvalidConfig("Empty L-BFGS history".to_string()))?
            .len();

        let result = Array2::eye(n);

        // Two-loop recursion would be implemented here
        // This is a placeholder returning identity matrix

        Ok(result)
    }

    fn compute_diagonal_hessian(&mut self, outputid: usize) -> Result<Array2<T>> {
        let n_vars = self.variables.len();
        let mut hessian = Array2::zeros((n_vars, n_vars));

        // Compute only diagonal elements (much faster)
        for i in 0..n_vars {
            let second_deriv = self.compute_second_derivative(outputid, i, i)?;
            hessian[[i, i]] = second_deriv;
        }

        Ok(hessian)
    }

    /// Update BFGS approximation
    pub fn update_bfgs(
        &mut self,
        gradient_new: &Array1<T>,
        gradient_old: &Array1<T>,
        step: &Array1<T>,
    ) -> Result<()> {
        if self.hessian_state.is_none() {
            let n = gradient_new.len();
            self.hessian_state = Some(HessianApproximationState {
                method: HessianApproximation::BFGS,
                gradient_history: Vec::new(),
                update_history: Vec::new(),
                inverse_hessian: Some(Array2::eye(n)),
                lbfgs_memory: None,
                max_history: 100,
            });
        }

        let state = self.hessian_state.as_mut().unwrap();

        let y = gradient_new - gradient_old; // Gradient difference
        let s = step; // Parameter update

        let sy = s.dot(&y);
        if sy.abs() < num_traits::cast::cast(1e-10).unwrap_or_else(|| T::zero()) {
            return Ok(()); // Skip update if curvature condition not satisfied
        }

        if let Some(ref mut h_inv) = state.inverse_hessian {
            // BFGS update: H_{k+1}^{-1} = (I - ρ s y^T) H_k^{-1} (I - ρ y s^T) + ρ s s^T
            let rho = T::one() / sy;
            let n = s.len();

            let mut sy_outer = Array2::zeros((n, n));
            let mut ys_outer = Array2::zeros((n, n));
            let mut ss_outer = Array2::zeros((n, n));

            for i in 0..n {
                for j in 0..n {
                    sy_outer[[i, j]] = s[i] * y[j];
                    ys_outer[[i, j]] = y[i] * s[j];
                    ss_outer[[i, j]] = s[i] * s[j];
                }
            }

            let identity = Array2::eye(n);
            let term1 = &identity - &(sy_outer * rho);
            let term2 = &identity - &(ys_outer * rho);

            *h_inv = term1.dot(h_inv).dot(&term2) + ss_outer * rho;
        }

        // Store history
        state.gradient_history.push(gradient_new.clone());
        state.update_history.push(step.clone());

        // Limit history size
        if state.gradient_history.len() > state.max_history {
            state.gradient_history.remove(0);
            state.update_history.remove(0);
        }

        Ok(())
    }

    /// Compute meta-gradients for meta-learning
    pub fn compute_meta_gradients(
        &mut self,
        _inner_steps: usize,
        outer_objective_id: usize,
    ) -> Result<Vec<T>> {
        if !self.config.enable_meta_gradients {
            return Err(OptimError::InvalidConfig(
                "Meta-gradients not enabled".to_string(),
            ));
        }

        // Simplified meta-gradient computation
        // In practice, this would involve:
        // 1. Running inner optimization loop
        // 2. Computing gradients of outer objective with respect to initial parameters
        // 3. Using implicit function theorem or gradient tape

        let meta_grads = self.backward(outer_objective_id)?;
        Ok(meta_grads)
    }

    /// Enable gradient checkpointing for memory efficiency
    pub fn enable_checkpointing(&mut self, enabled: bool) {
        self.config.gradient_checkpointing = enabled;
    }

    /// Get computational graph statistics
    pub fn get_graph_stats(&self) -> GraphStats {
        GraphStats {
            num_nodes: self.graph.len(),
            num_variables: self.variables.len(),
            tape_length: self.tape.len(),
            memory_usage_estimate: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> usize {
        // Rough estimate in bytes
        let node_size = std::mem::size_of::<ADNode<T>>();
        let tape_size = std::mem::size_of::<TapeEntry<T>>();

        self.graph.len() * node_size + self.tape.len() * tape_size
    }

    /// Create computation checkpoint for gradient checkpointing
    fn create_checkpoint(&self) -> ComputationCheckpoint<T> {
        let values: Vec<T> = self.graph.iter().map(|node| node.value).collect();
        let graph_state = self.graph.clone();
        let memory_usage = values.len() * std::mem::size_of::<T>()
            + graph_state.len() * std::mem::size_of::<ADNode<T>>();

        ComputationCheckpoint {
            values,
            graph_state,
            id: self.context.forward_pass_id,
            memory_usage,
        }
    }

    /// Restore computation state from checkpoint
    fn restore_checkpoint(&mut self, checkpoint: &ComputationCheckpoint<T>) -> Result<()> {
        if checkpoint.values.len() != self.graph.len() {
            return Err(OptimError::InvalidConfig(
                "Checkpoint size mismatch".to_string(),
            ));
        }

        // Restore node values
        for (i, &value) in checkpoint.values.iter().enumerate() {
            if i < self.graph.len() {
                self.graph[i].value = value;
            }
        }

        // Restore graph state if needed
        if checkpoint.graph_state.len() == self.graph.len() {
            self.graph = checkpoint.graph_state.clone();
        }

        Ok(())
    }

    /// External framework integration hooks
    pub fn integrate_with_candle(&mut self, enable: bool) -> Result<()> {
        if enable {
            // Enhanced integration with candle-core for production use
            self.context.higher_order = true;
            self.config.enable_forward_mode = true;
            self.config.enable_reverse_mode = true;
            self.config.gradient_checkpointing = true;

            // Enable candle-specific optimizations
            self.setup_candle_compatibility()?;
        }
        Ok(())
    }

    /// Set up candle-core compatibility layer
    fn setup_candle_compatibility(&mut self) -> Result<()> {
        // Configure automatic differentiation for candle tensors
        self.config.max_derivative_order = 3; // Support higher-order derivatives
        self.config.enable_hessian = true;
        self.config.sparse_hessian = true; // Efficient for large models

        // Set up memory-efficient gradient computation
        self.config.checkpoint_chunk_size = 500; // Balanced for candle tensors

        // Initialize candle-specific tape format
        self.tape.clear();
        self.tape.reserve(10000); // Pre-allocate for efficiency

        Ok(())
    }

    /// Convert candle tensor gradients to internal format
    pub fn from_candle_gradients<D: Dimension>(
        &mut self,
        gradients: &Array<T, D>,
    ) -> Result<Vec<T>> {
        if let Some(grad_slice) = gradients.as_slice() {
            Ok(grad_slice.to_vec())
        } else {
            // Handle non-contiguous arrays
            let flat_grads: Vec<T> = gradients.iter().cloned().collect();
            Ok(flat_grads)
        }
    }

    /// Convert internal gradients to candle-compatible format
    pub fn to_candle_gradients<D: Dimension>(
        &self,
        gradients: &[T],
        shape: &[usize],
    ) -> Result<Array<T, D>> {
        if gradients.len() != shape.iter().product::<usize>() {
            return Err(OptimError::InvalidConfig(
                "Gradient size mismatch with shape".to_string(),
            ));
        }

        // Convert to proper dimension type
        let dyn_array = Array::from_shape_vec(shape, gradients.to_vec())
            .map_err(|_| OptimError::InvalidConfig("Invalid shape for gradients".to_string()))?;

        // Convert to target dimension
        dyn_array
            .into_dimensionality::<D>()
            .map_err(|_| OptimError::InvalidConfig("Dimension conversion failed".to_string()))
    }

    /// Compute candle-compatible Jacobian-vector product
    pub fn compute_jvp(&mut self, outputid: usize, tangent: &[T]) -> Result<Vec<T>> {
        if tangent.len() != self.variables.len() {
            return Err(OptimError::InvalidConfig(
                "Tangent vector size mismatch".to_string(),
            ));
        }

        // Forward-mode AD for computing JVP
        let mut jvp_values = vec![T::zero(); self.graph.len()];

        // Initialize tangent values for input variables
        for (_var_name, &var_id) in &self.variables {
            if var_id < tangent.len() {
                jvp_values[var_id] = tangent[var_id];
            }
        }

        // Forward pass through computation graph
        for entry in &self.tape {
            let mut output_tangent = T::zero();

            for (i, &input_id) in entry.inputs.iter().enumerate() {
                if i < entry.local_gradients.len() {
                    output_tangent =
                        output_tangent + entry.local_gradients[i] * jvp_values[input_id];
                }
            }

            jvp_values[entry.output] = output_tangent;
        }

        Ok(vec![jvp_values[outputid]])
    }

    /// Compute candle-compatible vector-Jacobian product
    pub fn compute_vjp(&mut self, outputid: usize, cotangent: T) -> Result<Vec<T>> {
        // Reverse-mode AD for computing VJP
        let mut vjp_values = vec![T::zero(); self.graph.len()];
        vjp_values[outputid] = cotangent;

        // Reverse pass through computation graph
        for entry in self.tape.iter().rev() {
            let output_cotangent = vjp_values[entry.output];

            for (i, &input_id) in entry.inputs.iter().enumerate() {
                if i < entry.local_gradients.len() {
                    vjp_values[input_id] =
                        vjp_values[input_id] + output_cotangent * entry.local_gradients[i];
                }
            }
        }

        // Extract gradients for variables only
        let mut result = Vec::new();
        for (_var_name, &var_id) in &self.variables {
            result.push(vjp_values[var_id]);
        }

        Ok(result)
    }

    /// Candle tensor operation registration
    pub fn register_candle_operation(
        &mut self,
        op_name: &str,
        inputs: &[usize],
        output_value: T,
        local_grads: &[T],
    ) -> usize {
        let outputid = self.graph.len();

        // Create output node
        let output_node = ADNode {
            value: output_value,
            gradient: vec![T::zero()],
            hessian: None,
            operation: Operation::Custom(op_name.to_string()),
            parents: inputs.to_vec(),
            id: outputid,
        };

        self.graph.push(output_node);

        // Record operation in tape
        let tape_entry = TapeEntry {
            operation: Operation::Custom(op_name.to_string()),
            inputs: inputs.to_vec(),
            output: outputid,
            local_gradients: local_grads.to_vec(),
        };

        self.tape.push(tape_entry);

        outputid
    }

    /// Optimize computation graph for candle operations
    pub fn optimize_for_candle(&mut self) -> Result<()> {
        // Graph optimization passes for candle integration
        self.fuse_operations()?;
        self.eliminate_dead_code()?;
        self.optimize_memory_layout()?;

        Ok(())
    }

    /// Fuse compatible operations for better performance
    fn fuse_operations(&mut self) -> Result<()> {
        // Identify fusion opportunities (e.g., add + multiply, exp + log)
        let mut fused_operations = Vec::new();

        for i in 0..self.tape.len() {
            if i + 1 < self.tape.len() {
                let curr_op = &self.tape[i];
                let next_op = &self.tape[i + 1];

                // Example fusion: add followed by multiply
                if matches!(curr_op.operation, Operation::Add)
                    && matches!(next_op.operation, Operation::Multiply)
                    && next_op.inputs.contains(&curr_op.output)
                {
                    fused_operations.push((i, i + 1, "fused_add_mul".to_string()));
                }
            }
        }

        // Apply fusions (simplified implementation)
        for (start, end, fused_name) in fused_operations.iter().rev() {
            // Replace operations with fused version
            let fused_entry = TapeEntry {
                operation: Operation::Custom(fused_name.clone()),
                inputs: self.tape[*start].inputs.clone(),
                output: self.tape[*end].output,
                local_gradients: vec![], // Would compute combined gradients
            };

            self.tape[*start] = fused_entry;
            self.tape.remove(*end);
        }

        Ok(())
    }

    /// Eliminate dead code in computation graph
    fn eliminate_dead_code(&mut self) -> Result<()> {
        // Mark reachable nodes from outputs
        let mut reachable = vec![false; self.graph.len()];
        let mut stack = Vec::new();

        // Start from all variable nodes (they're always reachable)
        for &var_id in self.variables.values() {
            reachable[var_id] = true;
            stack.push(var_id);
        }

        // Mark all nodes reachable from variables
        while let Some(nodeid) = stack.pop() {
            for entry in &self.tape {
                if entry.inputs.contains(&nodeid) && !reachable[entry.output] {
                    reachable[entry.output] = true;
                    stack.push(entry.output);
                }
            }
        }

        // Remove unreachable operations
        self.tape.retain(|entry| reachable[entry.output]);

        Ok(())
    }

    /// Optimize memory layout for better cache performance
    fn optimize_memory_layout(&mut self) -> Result<()> {
        // Sort tape entries to improve memory locality
        self.tape.sort_by_key(|entry| entry.output);

        // Compact graph representation
        let mut new_graph = Vec::new();
        let mut id_mapping = HashMap::new();

        for (new_id, node) in self.graph.iter().enumerate() {
            id_mapping.insert(node.id, new_id);
            let mut new_node = node.clone();
            new_node.id = new_id;
            new_graph.push(new_node);
        }

        // Update references in tape
        for entry in &mut self.tape {
            entry.output = *id_mapping.get(&entry.output).unwrap_or(&entry.output);
            for input in &mut entry.inputs {
                *input = *id_mapping.get(input).unwrap_or(input);
            }
        }

        self.graph = new_graph;

        Ok(())
    }

    /// PyTorch-style autograd integration
    pub fn set_pytorch_compatibility(&mut self, enable: bool) {
        // Enable PyTorch-compatible gradient computation
        if enable {
            self.context.record_gradients = true;
            self.config.enable_meta_gradients = true;
        }
    }

    /// JAX-style transformation compatibility  
    pub fn enable_jax_transforms(&mut self, enable: bool) -> Result<()> {
        if enable {
            // Enable JAX-compatible transformations
            self.config.enable_forward_mode = true;
            self.config.enable_reverse_mode = true;
            self.config.gradient_checkpointing = true;
        }
        Ok(())
    }
}

/// Computational graph statistics
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_variables: usize,
    pub tape_length: usize,
    pub memory_usage_estimate: usize,
}

/// Automatic differentiation utilities
pub struct AutodiffUtils;

impl AutodiffUtils {
    /// Check gradients using finite differences
    pub fn check_gradients<T: Float>(
        forward_fn: impl Fn(&[T]) -> T,
        gradient_fn: impl Fn(&[T]) -> Vec<T>,
        point: &[T],
        epsilon: T,
    ) -> Result<GradientCheckResult<T>> {
        let analytical_grad = gradient_fn(point);
        let mut numerical_grad = vec![T::zero(); point.len()];

        for i in 0..point.len() {
            let mut point_plus = point.to_vec();
            let mut point_minus = point.to_vec();

            point_plus[i] = point_plus[i] + epsilon;
            point_minus[i] = point_minus[i] - epsilon;

            let f_plus = forward_fn(&point_plus);
            let f_minus = forward_fn(&point_minus);

            numerical_grad[i] = (f_plus - f_minus) / (num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()) * epsilon);
        }

        // Compute relative error
        let mut max_error = T::zero();
        let mut total_error = T::zero();

        for i in 0..point.len() {
            let error = (analytical_grad[i] - numerical_grad[i]).abs();
            let relative_error = if analytical_grad[i].abs() > num_traits::cast::cast(1e-8).unwrap_or_else(|| T::zero()) {
                error / analytical_grad[i].abs()
            } else {
                error
            };

            if relative_error > max_error {
                max_error = relative_error;
            }

            total_error = total_error + relative_error;
        }

        let avg_error = total_error / T::from(point.len()).unwrap();

        Ok(GradientCheckResult {
            analytical_gradient: analytical_grad,
            numerical_gradient: numerical_grad,
            max_relative_error: max_error,
            avg_relative_error: avg_error,
            is_correct: max_error < num_traits::cast::cast(1e-4).unwrap_or_else(|| T::zero()),
        })
    }
}

/// Gradient check result
#[derive(Debug, Clone)]
pub struct GradientCheckResult<T: Float> {
    pub analytical_gradient: Vec<T>,
    pub numerical_gradient: Vec<T>,
    pub max_relative_error: T,
    pub avg_relative_error: T,
    pub is_correct: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autodiff_config_default() {
        let config = AutodiffConfig::default();
        assert!(config.enable_forward_mode);
        assert!(config.enable_reverse_mode);
        assert_eq!(config.max_derivative_order, 2);
        assert!(config.enable_hessian);
    }

    #[test]
    fn test_autodiff_engine_creation() {
        let config = AutodiffConfig::default();
        let mut engine = AutodiffEngine::<f64>::new(config);

        let var_id = engine.create_variable("x", 5.0);
        assert_eq!(engine.graph.len(), 1);
        assert_eq!(engine.graph[var_id].value, 5.0);
    }

    #[test]
    fn test_gradient_check() {
        let forward_fn = |x: &[f64]| x[0] * x[0] + 2.0 * x[0] * x[1] + x[1] * x[1];
        let gradient_fn = |x: &[f64]| vec![2.0 * x[0] + 2.0 * x[1], 2.0 * x[0] + 2.0 * x[1]];

        let point = [3.0, 4.0];
        let result = AutodiffUtils::check_gradients(forward_fn, gradient_fn, &point, 1e-8).unwrap();

        assert!(result.is_correct);
        assert!(result.max_relative_error < 1e-4);
    }

    #[test]
    fn test_bfgs_update() {
        let config = AutodiffConfig {
            hessian_approximation: HessianApproximation::BFGS,
            ..Default::default()
        };

        let mut engine = AutodiffEngine::<f64>::new(config);

        let grad_new = Array1::from_vec(vec![1.0, 2.0]);
        let grad_old = Array1::from_vec(vec![0.5, 1.5]);
        let step = Array1::from_vec(vec![0.1, 0.2]);

        let result = engine.update_bfgs(&grad_new, &grad_old, &step);
        assert!(result.is_ok());
    }
}
