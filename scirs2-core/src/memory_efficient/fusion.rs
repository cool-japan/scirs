use crate::error::{CoreError, ErrorContext, ErrorLocation};
use once_cell::sync::Lazy;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

/// Type alias for a fused operation
type FusedOpArc = Arc<dyn FusedOp>;

/// Type alias for the fusion registry storage
type FusionRegistryMap = HashMap<TypeId, Vec<FusedOpArc>>;

// Global registry of fused operations
static FUSION_REGISTRY: Lazy<Mutex<FusionRegistryMap>> = Lazy::new(|| Mutex::new(HashMap::new()));

/// A trait for operations that can be fused together for better performance
pub trait FusedOp: Send + Sync {
    /// Returns the unique name of this operation
    fn name(&self) -> &str;

    /// Returns the type ID of the input to this operation
    fn input_type(&self) -> TypeId;

    /// Returns the type ID of the output from this operation
    fn output_type(&self) -> TypeId;

    /// Checks if this operation can be fused with another operation
    fn can_fuse_with(&self, other: &dyn FusedOp) -> bool;

    /// Creates a new operation that is the fusion of this operation with another
    fn fuse_with(&self, other: &dyn FusedOp) -> Arc<dyn FusedOp>;

    /// Applies this operation to an input (as Any)
    fn apply(&self, input: &dyn Any) -> Result<Box<dyn Any>, CoreError>;

    /// Clone this operation
    fn clone_op(&self) -> Arc<dyn FusedOp>;
}

/// A structure for chaining multiple operations together and optimizing the execution
#[derive(Clone)]
pub struct OpFusion {
    /// The sequence of operations to apply
    ops: Vec<Arc<dyn FusedOp>>,
    /// The input type ID
    input_type: TypeId,
    /// The output type ID
    output_type: TypeId,
}

impl fmt::Debug for OpFusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpFusion")
            .field("num_ops", &self.ops.len())
            .finish()
    }
}

impl OpFusion {
    /// Create a new operation fusion
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            input_type: TypeId::of::<()>(),
            output_type: TypeId::of::<()>(),
        }
    }

    /// Add an operation to the fusion chain
    pub fn add_op(&mut self, op: Arc<dyn FusedOp>) -> Result<&mut Self, CoreError> {
        if self.ops.is_empty() {
            self.input_type = op.input_type();
            self.output_type = op.output_type();
        } else if op.input_type() != self.output_type {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Operation input type does not match previous output type")
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let output_type = op.output_type();
        self.ops.push(op);
        self.output_type = output_type;
        Ok(self)
    }

    /// Optimize the operation chain by fusing operations where possible
    pub fn optimize(&mut self) -> Result<&mut Self, CoreError> {
        if self.ops.len() <= 1 {
            return Ok(self);
        }

        let mut optimized = Vec::new();
        let mut current_op = self.ops[0].clone_op();

        for i in 1..self.ops.len() {
            let next_op = &self.ops[i];

            if current_op.can_fuse_with(next_op.as_ref()) {
                current_op = current_op.fuse_with(next_op.as_ref());
            } else {
                optimized.push(current_op);
                current_op = next_op.clone_op();
            }
        }

        optimized.push(current_op);
        self.ops = optimized;

        Ok(self)
    }

    /// Apply the operation chain to an input value
    pub fn apply<A: 'static>(&self, input: A) -> Result<Box<dyn Any>, CoreError> {
        if TypeId::of::<A>() != self.input_type {
            return Err(CoreError::ValidationError(
                ErrorContext::new("Input type does not match expected type")
                    .with_location(ErrorLocation::new(file!(), line!())),
            ));
        }

        let mut result: Box<dyn Any> = Box::new(input);

        for op in &self.ops {
            result = op.apply(result.as_ref())?;
        }

        Ok(result)
    }

    /// Get the number of operations in the chain
    pub fn num_ops(&self) -> usize {
        self.ops.len()
    }

    /// Check if the operation chain is empty
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

impl Default for OpFusion {
    fn default() -> Self {
        Self::new()
    }
}

/// Register a fused operation in the global registry
pub fn register_fusion<T: 'static>(op: Arc<dyn FusedOp>) -> Result<(), CoreError> {
    let type_id = TypeId::of::<T>();

    let mut registry = FUSION_REGISTRY.lock().unwrap();
    let ops = registry.entry(type_id).or_default();
    ops.push(op);

    Ok(())
}

/// Get all registered fused operations for a type
#[allow(dead_code)]
pub fn get_fusions<T: 'static>() -> Vec<Arc<dyn FusedOp>> {
    let type_id = TypeId::of::<T>();

    let registry = FUSION_REGISTRY.lock().unwrap();
    match registry.get(&type_id) {
        Some(ops) => ops.clone(),
        None => Vec::new(),
    }
}
