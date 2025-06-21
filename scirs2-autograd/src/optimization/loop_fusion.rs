//! Loop fusion optimization for element-wise operations
//!
//! This module implements automatic fusion of consecutive element-wise operations
//! to reduce memory bandwidth requirements and improve cache efficiency.

use crate::graph::{Graph, TensorID};
use crate::op::OpError;
use crate::Float;
use ndarray::{Array, IxDyn, Zip};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// Type alias for kernel function
type KernelFunction<F> =
    Box<dyn Fn(&[&Array<F, IxDyn>]) -> Result<Array<F, IxDyn>, OpError> + Send + Sync>;
/// Type alias for kernel compilation result
type KernelResult<F> = Result<KernelFunction<F>, OpError>;

/// Types of operations that can be fused
#[derive(Debug, Clone, PartialEq)]
pub enum FusableOperation<F: Float> {
    /// Element-wise addition
    Add,
    /// Element-wise subtraction
    Sub,
    /// Element-wise multiplication
    Mul,
    /// Element-wise division
    Div,
    /// Unary function application
    UnaryFunc(UnaryFunction<F>),
    /// Binary function application with scalar
    ScalarOp(F, BinaryFunction),
}

/// Unary functions that can be fused
#[derive(Debug, Clone, PartialEq)]
pub enum UnaryFunction<F: Float> {
    /// ReLU activation
    ReLU,
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// Square function
    Square,
    /// Square root function
    Sqrt,
    /// Exponential function
    Exp,
    /// Natural logarithm
    Log,
    /// Absolute value
    Abs,
    /// Custom function
    Custom(fn(F) -> F),
}

/// Binary functions with scalars
#[derive(Debug, Clone, PartialEq)]
pub enum BinaryFunction {
    /// Addition with scalar
    AddScalar,
    /// Multiplication with scalar
    MulScalar,
    /// Power function
    Pow,
}

/// A sequence of fusable operations
#[derive(Debug, Clone)]
pub struct FusionChain<F: Float> {
    /// Operations in the chain
    operations: Vec<FusableOperation<F>>,
    /// Input tensor shapes
    input_shapes: Vec<Vec<usize>>,
    /// Output shape
    output_shape: Vec<usize>,
    /// Estimated performance benefit
    performance_benefit: f64,
}

impl<F: Float> Default for FusionChain<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> FusionChain<F> {
    /// Create a new fusion chain
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            input_shapes: Vec::new(),
            output_shape: Vec::new(),
            performance_benefit: 0.0,
        }
    }

    /// Add an operation to the chain
    pub fn add_operation(&mut self, op: FusableOperation<F>, input_shape: Vec<usize>) {
        // For the first operation, set output shape before estimating benefit
        if self.output_shape.is_empty() {
            self.output_shape = input_shape.clone();
        }

        // Estimate performance benefit before moving op
        self.performance_benefit += self.estimate_benefit(&op);

        self.operations.push(op);
        self.input_shapes.push(input_shape.clone());

        // For element-wise operations, output shape matches input shape
        self.output_shape = input_shape;
    }

    /// Estimate performance benefit of adding this operation
    fn estimate_benefit(&self, op: &FusableOperation<F>) -> f64 {
        // Each fused operation saves one memory round-trip
        let elements = self.output_shape.iter().product::<usize>() as f64;

        match op {
            FusableOperation::Add
            | FusableOperation::Sub
            | FusableOperation::Mul
            | FusableOperation::Div => {
                // Binary operations: save memory bandwidth
                elements * 0.5
            }
            FusableOperation::UnaryFunc(_) => {
                // Unary operations: save memory + computation overhead
                elements * 0.3
            }
            FusableOperation::ScalarOp(_, _) => {
                // Scalar operations: very efficient when fused
                elements * 0.7
            }
        }
    }

    /// Check if this chain is worth fusing
    pub fn is_worthwhile(&self) -> bool {
        self.operations.len() >= 2 && self.performance_benefit > 1000.0
    }

    /// Get the number of operations in the chain
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

/// Loop fusion optimizer
pub struct LoopFusionOptimizer<F: Float> {
    /// Detected fusion chains
    fusion_chains: Vec<FusionChain<F>>,
    /// Mapping from original operations to fused operations
    fusion_mapping: HashMap<TensorID, usize>,
    /// Performance statistics
    stats: FusionStats<F>,
}

impl<F: Float> Default for LoopFusionOptimizer<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> LoopFusionOptimizer<F> {
    /// Create a new loop fusion optimizer
    pub fn new() -> Self {
        Self {
            fusion_chains: Vec::new(),
            fusion_mapping: HashMap::new(),
            stats: FusionStats::default(),
        }
    }

    /// Analyze a computation graph for fusion opportunities
    pub fn analyze_graph(&mut self, graph: &Graph<F>) -> Result<(), OpError> {
        self.fusion_chains.clear();
        self.fusion_mapping.clear();

        // Find all element-wise operations
        let element_wise_ops = self.find_element_wise_operations(graph);

        // Group operations into fusion chains
        let chains = self.identify_fusion_chains(&element_wise_ops, graph);

        // Filter chains that are worth fusing
        for chain in chains {
            if chain.is_worthwhile() {
                self.fusion_chains.push(chain);
            }
        }

        self.stats.chains_identified = self.fusion_chains.len();
        self.stats.total_operations_fused =
            self.fusion_chains.iter().map(|chain| chain.len()).sum();

        Ok(())
    }

    /// Find all element-wise operations in the graph
    fn find_element_wise_operations(&self, _graph: &Graph<F>) -> Vec<TensorID> {
        // This is a simplified implementation - in practice, would need to
        // inspect the actual graph structure and operation types
        Vec::new()
    }

    /// Identify sequences of operations that can be fused
    fn identify_fusion_chains(
        &self,
        operations: &[TensorID],
        graph: &Graph<F>,
    ) -> Vec<FusionChain<F>> {
        let mut chains = Vec::new();
        let mut visited = HashSet::new();

        for &op_idx in operations {
            if visited.contains(&op_idx) {
                continue;
            }

            let chain = self.build_fusion_chain(op_idx, graph, &mut visited);
            if !chain.is_empty() {
                chains.push(chain);
            }
        }

        chains
    }

    /// Build a fusion chain starting from a given operation
    fn build_fusion_chain(
        &self,
        start_op: TensorID,
        graph: &Graph<F>,
        visited: &mut HashSet<TensorID>,
    ) -> FusionChain<F> {
        let mut chain = FusionChain::new();
        let mut current_op = start_op;

        loop {
            if visited.contains(&current_op) {
                break;
            }

            // Check if current operation is fusable
            if let Some(fusable_op) = self.classify_operation(current_op, graph) {
                visited.insert(current_op);

                // For this example, assume shape is [100] - in practice would extract from graph
                chain.add_operation(fusable_op, vec![100]);

                // Find next operation in chain
                if let Some(next_op) = self.find_next_fusable_operation(current_op, graph) {
                    current_op = next_op;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        chain
    }

    /// Classify an operation as fusable or not
    fn classify_operation(
        &self,
        _op_idx: TensorID,
        _graph: &Graph<F>,
    ) -> Option<FusableOperation<F>> {
        // In practice, would inspect the actual operation type
        // For this example, return a sample operation
        Some(FusableOperation::Add)
    }

    /// Find the next operation that can be fused with the current one
    fn find_next_fusable_operation(
        &self,
        _current_op: TensorID,
        _graph: &Graph<F>,
    ) -> Option<TensorID> {
        // In practice, would traverse graph dependencies
        None
    }

    /// Apply fusion optimizations to create fused kernels
    pub fn apply_fusion(&self) -> Result<Vec<FusedKernel<F>>, OpError> {
        let mut fused_kernels = Vec::new();

        for chain in &self.fusion_chains {
            let kernel = self.create_fused_kernel(chain)?;
            fused_kernels.push(kernel);
        }

        Ok(fused_kernels)
    }

    /// Create a fused kernel from a fusion chain
    fn create_fused_kernel(&self, chain: &FusionChain<F>) -> Result<FusedKernel<F>, OpError> {
        FusedKernel::from_chain(chain.clone())
    }

    /// Get fusion statistics
    pub fn get_stats(&self) -> &FusionStats<F> {
        &self.stats
    }
}

/// A fused kernel that combines multiple element-wise operations
pub struct FusedKernel<F: Float> {
    /// The fusion chain this kernel implements
    chain: FusionChain<F>,
    /// Compiled kernel function
    kernel_func: KernelFunction<F>,
}

impl<F: Float> FusedKernel<F> {
    /// Create a fused kernel from a fusion chain
    pub fn from_chain(chain: FusionChain<F>) -> Result<Self, OpError> {
        let kernel_func = Self::compile_kernel(&chain)?;

        Ok(Self { chain, kernel_func })
    }

    /// Compile the fusion chain into an executable kernel
    fn compile_kernel(chain: &FusionChain<F>) -> KernelResult<F> {
        let operations = chain.operations.clone();
        let _output_shape = chain.output_shape.clone();

        Ok(Box::new(
            move |inputs: &[&Array<F, IxDyn>]| -> Result<Array<F, IxDyn>, OpError> {
                if inputs.is_empty() {
                    return Err(OpError::RuntimeError(
                        "No input arrays provided".to_string(),
                    ));
                }

                let input = inputs[0];
                let mut result = Array::zeros(input.raw_dim());

                // Execute fused operations in a single loop
                Zip::from(&mut result)
                    .and(input)
                    .par_for_each(|output, &input_val| {
                        let mut value = input_val;

                        // Apply all operations in sequence
                        for op in &operations {
                            value = match op {
                                FusableOperation::Add => {
                                    // For binary ops, would need second input
                                    // For now, just pass through
                                    value
                                }
                                FusableOperation::Mul => value,
                                FusableOperation::UnaryFunc(func) => {
                                    Self::apply_unary_function(value, func)
                                }
                                FusableOperation::ScalarOp(scalar, func) => {
                                    Self::apply_scalar_operation(value, *scalar, func)
                                }
                                _ => value,
                            };
                        }

                        *output = value;
                    });

                Ok(result)
            },
        ))
    }

    /// Apply a unary function
    pub fn apply_unary_function(value: F, func: &UnaryFunction<F>) -> F {
        match func {
            UnaryFunction::ReLU => {
                if value > F::zero() {
                    value
                } else {
                    F::zero()
                }
            }
            UnaryFunction::Sigmoid => {
                let one = F::one();
                one / (one + (-value).exp())
            }
            UnaryFunction::Tanh => value.tanh(),
            UnaryFunction::Square => value * value,
            UnaryFunction::Sqrt => value.sqrt(),
            UnaryFunction::Exp => value.exp(),
            UnaryFunction::Log => value.ln(),
            UnaryFunction::Abs => value.abs(),
            UnaryFunction::Custom(f) => f(value),
        }
    }

    /// Apply a scalar operation
    pub fn apply_scalar_operation(value: F, scalar: F, func: &BinaryFunction) -> F {
        match func {
            BinaryFunction::AddScalar => value + scalar,
            BinaryFunction::MulScalar => value * scalar,
            BinaryFunction::Pow => value.powf(scalar),
        }
    }

    /// Execute the fused kernel
    pub fn execute(&self, inputs: &[&Array<F, IxDyn>]) -> Result<Array<F, IxDyn>, OpError> {
        (self.kernel_func)(inputs)
    }

    /// Get the fusion chain
    pub fn get_chain(&self) -> &FusionChain<F> {
        &self.chain
    }

    /// Estimate performance improvement
    pub fn estimate_speedup(&self) -> f64 {
        // Estimate based on reduced memory traffic
        let num_ops = self.chain.len() as f64;
        let memory_reduction = (num_ops - 1.0) / num_ops;

        // Conservative estimate: 1.2x to 3x speedup depending on chain length
        1.0 + memory_reduction * 2.0
    }
}

/// Statistics for fusion optimization
#[derive(Debug, Clone)]
pub struct FusionStats<F: crate::Float> {
    /// Number of fusion chains identified
    pub chains_identified: usize,
    /// Total operations that were fused
    pub total_operations_fused: usize,
    /// Estimated memory bandwidth reduction (percentage)
    pub memory_bandwidth_reduction: f64,
    /// Estimated performance improvement (speedup factor)
    pub estimated_speedup: f64,
    /// Phantom data for type parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: crate::Float> FusionStats<F> {
    /// Calculate memory bandwidth reduction
    pub fn calculate_memory_reduction(&mut self, original_ops: usize) {
        if original_ops > 0 {
            self.memory_bandwidth_reduction =
                (original_ops - self.chains_identified) as f64 / original_ops as f64 * 100.0;
        }
    }

    /// Calculate estimated speedup
    pub fn calculate_speedup(&mut self, kernels: &[FusedKernel<F>]) {
        if !kernels.is_empty() {
            self.estimated_speedup = kernels
                .iter()
                .map(|kernel| kernel.estimate_speedup())
                .sum::<f64>()
                / kernels.len() as f64;
        }
    }
}

impl<F: crate::Float> Default for FusionStats<F> {
    fn default() -> Self {
        Self {
            chains_identified: 0,
            total_operations_fused: 0,
            memory_bandwidth_reduction: 0.0,
            estimated_speedup: 0.0,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// High-level loop fusion manager
pub struct LoopFusionManager<F: Float> {
    /// The optimizer instance
    optimizer: LoopFusionOptimizer<F>,
    /// Compiled kernels
    kernels: Vec<FusedKernel<F>>,
    /// Configuration
    config: FusionConfig,
}

impl<F: Float> Default for LoopFusionManager<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float> LoopFusionManager<F> {
    /// Create a new fusion manager
    pub fn new() -> Self {
        Self {
            optimizer: LoopFusionOptimizer::new(),
            kernels: Vec::new(),
            config: FusionConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: FusionConfig) -> Self {
        Self {
            optimizer: LoopFusionOptimizer::new(),
            kernels: Vec::new(),
            config,
        }
    }

    /// Optimize a computation graph with loop fusion
    pub fn optimize_graph(&mut self, graph: &Graph<F>) -> Result<(), OpError> {
        if !self.config.enable_fusion {
            return Ok(());
        }

        // Analyze the graph for fusion opportunities
        self.optimizer.analyze_graph(graph)?;

        // Create fused kernels
        self.kernels = self.optimizer.apply_fusion()?;

        Ok(())
    }

    /// Execute fused operations
    pub fn execute_fused_operation(
        &self,
        kernel_id: usize,
        inputs: &[&Array<F, IxDyn>],
    ) -> Result<Array<F, IxDyn>, OpError> {
        if kernel_id >= self.kernels.len() {
            return Err(OpError::RuntimeError("Invalid kernel ID".to_string()));
        }

        self.kernels[kernel_id].execute(inputs)
    }

    /// Get optimization statistics
    pub fn get_stats(&self) -> &FusionStats<F> {
        self.optimizer.get_stats()
    }

    /// Get the number of fused kernels
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }

    /// Check if fusion is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enable_fusion
    }
}

/// Configuration for loop fusion optimization
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Enable loop fusion optimization
    pub enable_fusion: bool,
    /// Minimum chain length to consider for fusion
    pub min_chain_length: usize,
    /// Maximum chain length to avoid compilation overhead
    pub max_chain_length: usize,
    /// Minimum tensor size to consider for fusion
    pub min_tensor_size: usize,
    /// Enable parallel execution within fused kernels
    pub enable_parallel_fusion: bool,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_fusion: true,
            min_chain_length: 2,
            max_chain_length: 10,
            min_tensor_size: 1000,
            enable_parallel_fusion: true,
        }
    }
}

/// Global fusion manager instance
static FUSION_MANAGER: std::sync::OnceLock<Arc<Mutex<LoopFusionManager<f32>>>> =
    std::sync::OnceLock::new();

/// Initialize the global fusion manager
pub fn init_fusion_manager() -> Arc<Mutex<LoopFusionManager<f32>>> {
    FUSION_MANAGER
        .get_or_init(|| Arc::new(Mutex::new(LoopFusionManager::new())))
        .clone()
}

/// Configure global fusion settings
pub fn configure_fusion(config: FusionConfig) -> Result<(), OpError> {
    let manager = init_fusion_manager();
    let mut manager_guard = manager
        .lock()
        .map_err(|_| OpError::RuntimeError("Lock error".to_string()))?;
    *manager_guard = LoopFusionManager::with_config(config);
    Ok(())
}

/// Enable or disable loop fusion globally
pub fn set_fusion_enabled(enabled: bool) -> Result<(), OpError> {
    let config = FusionConfig {
        enable_fusion: enabled,
        ..Default::default()
    };
    configure_fusion(config)
}

/// Check if loop fusion is enabled
pub fn is_fusion_enabled() -> bool {
    let manager = init_fusion_manager();
    let result = match manager.lock() {
        Ok(manager_guard) => manager_guard.is_enabled(),
        Err(_) => false,
    };
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use ndarray::Array1;

    #[test]
    fn test_fusion_chain_creation() {
        let mut chain = FusionChain::<f32>::new();
        assert!(chain.is_empty());

        // Use larger tensors to ensure fusion is worthwhile (>1000 benefit)
        chain.add_operation(FusableOperation::Add, vec![10000]);
        chain.add_operation(
            FusableOperation::UnaryFunc(UnaryFunction::ReLU),
            vec![10000],
        );

        assert_eq!(chain.len(), 2);
        assert!(chain.is_worthwhile());
    }

    #[test]
    fn test_unary_functions() {
        let value = 2.0f32;

        assert_eq!(
            FusedKernel::<f32>::apply_unary_function(value, &UnaryFunction::Square),
            4.0
        );
        assert_eq!(
            FusedKernel::<f32>::apply_unary_function(-1.0, &UnaryFunction::ReLU),
            0.0
        );
        assert_eq!(
            FusedKernel::<f32>::apply_unary_function(1.0, &UnaryFunction::ReLU),
            1.0
        );
        assert_eq!(
            FusedKernel::<f32>::apply_unary_function(4.0, &UnaryFunction::Sqrt),
            2.0
        );
        assert_eq!(
            FusedKernel::<f32>::apply_unary_function(-2.0, &UnaryFunction::Abs),
            2.0
        );
    }

    #[test]
    fn test_scalar_operations() {
        let value = 3.0f32;
        let scalar = 2.0f32;

        assert_eq!(
            FusedKernel::<f32>::apply_scalar_operation(value, scalar, &BinaryFunction::AddScalar),
            5.0
        );
        assert_eq!(
            FusedKernel::<f32>::apply_scalar_operation(value, scalar, &BinaryFunction::MulScalar),
            6.0
        );
        assert_eq!(
            FusedKernel::<f32>::apply_scalar_operation(value, scalar, &BinaryFunction::Pow),
            9.0
        );
    }

    #[test]
    fn test_fused_kernel_creation() {
        let mut chain = FusionChain::new();
        chain.add_operation(FusableOperation::UnaryFunc(UnaryFunction::Square), vec![5]);
        chain.add_operation(
            FusableOperation::ScalarOp(2.0, BinaryFunction::MulScalar),
            vec![5],
        );

        let kernel = FusedKernel::from_chain(chain).unwrap();

        let input = Array::from_shape_vec(IxDyn(&[5]), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let result = kernel.execute(&[&input]).unwrap();

        // Should be (x^2) * 2 = [2, 8, 18, 32, 50]
        let expected = vec![2.0, 8.0, 18.0, 32.0, 50.0];
        assert_eq!(result.as_slice().unwrap(), &expected);
    }

    #[test]
    fn test_fusion_config() {
        let config = FusionConfig {
            enable_fusion: false,
            min_chain_length: 3,
            max_chain_length: 15,
            min_tensor_size: 5000,
            enable_parallel_fusion: false,
        };

        let manager: LoopFusionManager<f32> = LoopFusionManager::with_config(config.clone());
        assert!(!manager.is_enabled());
    }

    #[test]
    fn test_fusion_stats() {
        let mut stats: FusionStats<f32> = FusionStats {
            chains_identified: 5,
            total_operations_fused: 20,
            ..Default::default()
        };

        stats.calculate_memory_reduction(25);
        assert_eq!(stats.memory_bandwidth_reduction, 80.0);
    }

    #[test]
    fn test_global_fusion_manager() {
        set_fusion_enabled(true).unwrap();
        assert!(is_fusion_enabled());

        set_fusion_enabled(false).unwrap();
        assert!(!is_fusion_enabled());
    }

    #[test]
    fn test_complex_fused_chain() {
        let mut chain = FusionChain::new();

        // Create a complex chain: x -> x^2 -> ReLU -> *3 -> +1
        chain.add_operation(FusableOperation::UnaryFunc(UnaryFunction::Square), vec![4]);
        chain.add_operation(FusableOperation::UnaryFunc(UnaryFunction::ReLU), vec![4]);
        chain.add_operation(
            FusableOperation::ScalarOp(3.0, BinaryFunction::MulScalar),
            vec![4],
        );
        chain.add_operation(
            FusableOperation::ScalarOp(1.0, BinaryFunction::AddScalar),
            vec![4],
        );

        let kernel = FusedKernel::from_chain(chain).unwrap();

        let input = Array::from_shape_vec(IxDyn(&[4]), vec![-2.0, -1.0, 1.0, 2.0]).unwrap();
        let result = kernel.execute(&[&input]).unwrap();

        // Expected: x^2 -> ReLU -> *3 -> +1
        // [-2, -1, 1, 2] -> [4, 1, 1, 4] -> [4, 1, 1, 4] -> [12, 3, 3, 12] -> [13, 4, 4, 13]
        let expected = vec![13.0, 4.0, 4.0, 13.0];
        assert_eq!(result.as_slice().unwrap(), &expected);

        let speedup = kernel.estimate_speedup();
        assert!(speedup > 1.0 && speedup <= 4.0);
    }
}
