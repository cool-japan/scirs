//! Gradient tape module for automatic differentiation.
//!
//! This module provides a gradient tape implementation that records operations
//! and allows automatic computation of gradients in reverse mode.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_core::numeric::Float;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use crate::error::{AutogradError, Result};
use crate::graph::{Graph, Node};
use crate::tensor::Tensor;

/// GradientTape records operations for automatic differentiation.
///
/// The tape keeps track of operations and allows computing gradients
/// with respect to input variables through a reverse mode autodiff.
pub struct GradientTape<F: Float + Debug> {
    /// The computational graph
    graph: Mutex<Graph<F>>,

    /// Set of tensor IDs being watched
    watched_tensors: Mutex<HashSet<usize>>,

    /// Whether the tape is currently recording operations
    is_recording: Mutex<bool>,

    /// Whether to persist gradients after they are computed
    persistent: bool,
}

impl<F: Float + Debug + Send + Sync + 'static> GradientTape<F> {
    /// Create a new gradient tape.
    ///
    /// # Arguments
    ///
    /// * `persistent` - Whether to persist gradients after they are computed
    ///
    /// # Returns
    ///
    /// A new GradientTape instance
    pub fn new(persistent: bool) -> Self {
        Self {
            graph: Mutex::new(Graph::new()),
            watched_tensors: Mutex::new(HashSet::new()),
            is_recording: Mutex::new(true),
            persistent,
        }
    }

    /// Start recording operations on the tape.
    pub fn record(&self) {
        let mut is_recording = self.is_recording.lock().unwrap();
        *is_recording = true;
    }

    /// Stop recording operations on the tape.
    pub fn stop_recording(&self) {
        let mut is_recording = self.is_recording.lock().unwrap();
        *is_recording = false;
    }

    /// Check if the tape is currently recording operations.
    ///
    /// # Returns
    ///
    /// True if the tape is recording, false otherwise
    pub fn is_recording(&self) -> bool {
        *self.is_recording.lock().unwrap()
    }

    /// Add a tensor to the list of watched tensors.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to watch
    pub fn watch(&self, tensor: &Tensor<F>) {
        let mut watched_tensors = self.watched_tensors.lock().unwrap();
        watched_tensors.insert(tensor.id);
    }

    /// Remove a tensor from the list of watched tensors.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to stop watching
    pub fn unwatch(&self, tensor: &Tensor<F>) {
        let mut watched_tensors = self.watched_tensors.lock().unwrap();
        watched_tensors.remove(&tensor.id);
    }

    /// Check if a tensor is being watched.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to check
    ///
    /// # Returns
    ///
    /// True if the tensor is being watched, false otherwise
    pub fn is_watched(&self, tensor: &Tensor<F>) -> bool {
        let watched_tensors = self.watched_tensors.lock().unwrap();
        watched_tensors.contains(&tensor.id)
    }

    /// Record an operation on the tape.
    ///
    /// # Arguments
    ///
    /// * `node` - The operation node
    /// * `result_tensor` - The tensor produced by the operation
    pub fn record_operation(&self, node: Arc<Node<F>>, resulttensor: &Tensor<F>) {
        if !self.is_recording() {
            return;
        }

        let mut graph = self.graph.lock().unwrap();
        graph.add_node(node, result_tensor.id);
    }

    /// Compute gradients with respect to target tensors.
    ///
    /// # Arguments
    ///
    /// * `target` - The output tensor to compute gradients from
    /// * `sources` - The input tensors to compute gradients with respect to
    ///
    /// # Returns
    ///
    /// A HashMap mapping tensor IDs to their gradients
    pub fn gradient(
        &self,
        target: &mut Tensor<F>,
        sources: &[&Tensor<F>],
    ) -> Result<HashMap<usize, Array<F, IxDyn>>> {
        if !target.requires_grad {
            return Err(AutogradError::OperationError(
                "Cannot compute gradients with respect to a tensor that doesn't require gradients"
                    .to_string(),
            ));
        }

        // Initialize gradients
        let mut gradients = HashMap::new();

        // Filter sources to only include watched tensors
        let source_ids: HashSet<usize> = sources
            .iter()
            .filter(|t| self.is_watched(t))
            .map(|t| t.id)
            .collect();

        if source_ids.is_empty() {
            return Err(AutogradError::OperationError(
                "No source tensors are being watched".to_string(),
            ));
        }

        // Compute backward pass starting from the target
        target.backward(None)?;

        // Extract gradients for the sources
        for &tensor in sources {
            if let Some(ref grad) = tensor.grad {
                gradients.insert(tensor.id, grad.clone());
            }
        }

        // Clear gradients if not persistent
        if !self.persistent {
            // In a real implementation, we would clear all gradients
            // This is a placeholder for that operation
        }

        Ok(gradients)
    }

    /// Reset the tape, clearing all recorded operations and watched tensors.
    pub fn reset(&self) {
        let mut graph = self.graph.lock().unwrap();
        let mut watched_tensors = self.watched_tensors.lock().unwrap();

        *graph = Graph::new();
        watched_tensors.clear();
    }
}

/// Context manager for gradient computation.
///
/// This struct provides a convenient way to use a gradient tape
/// with automatic resource management.
pub struct GradientContext<F: Float + Debug + Send + Sync + 'static> {
    /// The underlying gradient tape
    pub tape: Arc<GradientTape<F>>,
}

impl<F: Float + Debug + Send + Sync + 'static> GradientContext<F> {
    /// Create a new gradient context.
    ///
    /// # Arguments
    ///
    /// * `persistent` - Whether to persist gradients after they are computed
    ///
    /// # Returns
    ///
    /// A new GradientContext instance
    pub fn new(persistent: bool) -> Self {
        Self {
            tape: Arc::new(GradientTape::new(_persistent)),
        }
    }

    /// Watch a tensor for gradient computation.
    ///
    /// # Arguments
    ///
    /// * `tensor` - The tensor to watch
    pub fn watch(&self, tensor: &Tensor<F>) {
        self.tape.watch(tensor);
    }

    /// Compute gradients with respect to target tensors.
    ///
    /// # Arguments
    ///
    /// * `target` - The output tensor to compute gradients from
    /// * `sources` - The input tensors to compute gradients with respect to
    ///
    /// # Returns
    ///
    /// A HashMap mapping tensor IDs to their gradients
    pub fn gradient(
        &self,
        target: &mut Tensor<F>,
        sources: &[&Tensor<F>],
    ) -> Result<HashMap<usize, Array<F, IxDyn>>> {
        self.tape.gradient(target, sources)
    }
}

impl<F: Float + Debug + Send + Sync + 'static> Drop for GradientContext<F> {
    fn drop(&mut self) {
        // Clean up resources when the context is dropped
        self.tape.reset();
    }
}
