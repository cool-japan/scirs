//! Gradient-based neural architecture search (DARTS-style methods)
//!
//! This module implements differentiable architecture search methods that use
//! gradient-based optimization to discover optimal optimizer architectures.

#![allow(dead_code)]

use ndarray::{Array1, Array2, Array3};
use num_traits::Float;
use std::collections::{HashMap, HashSet};

use crate::error::{OptimError, Result};

/// Configuration for gradient-based NAS (DARTS-style)
#[derive(Debug, Clone)]
pub struct GradientBasedNASConfig<T: Float> {
    /// Learning rate for architecture parameters
    pub arch_learning_rate: T,
    
    /// Learning rate for model weights
    pub weight_learning_rate: T,
    
    /// Weight decay for architecture parameters
    pub arch_weight_decay: T,
    
    /// Weight decay for model weights
    pub weight_decay: T,
    
    /// Number of epochs for architecture search
    pub search_epochs: usize,
    
    /// Number of warm-up epochs before architecture optimization
    pub warmup_epochs: usize,
    
    /// Temperature for Gumbel softmax
    pub temperature: T,
    
    /// Temperature annealing rate
    pub temperature_decay: T,
    
    /// Minimum temperature
    pub min_temperature: T,
    
    /// Architecture parameter regularization strength
    pub arch_regularization: T,
    
    /// Early stopping patience
    pub early_stopping_patience: usize,
    
    /// Progressive pruning threshold
    pub pruning_threshold: T,
    
    /// Number of candidate operations per edge
    pub num_candidate_ops: usize,
}

/// DARTS-style architecture searcher
#[derive(Debug)]
pub struct DARTSSearcher<T: Float> {
    /// Configuration
    config: GradientBasedNASConfig<T>,
    
    /// Architecture parameters (alpha values)
    architecture_parameters: Array3<T>,
    
    /// Search space definition
    search_space: SearchSpace<T>,
    
    /// Mixed operations for each edge
    mixed_operations: HashMap<EdgeId, MixedOperation<T>>,
    
    /// Architecture history
    architecture_history: Vec<ArchitectureCandidate<T>>,
    
    /// Training statistics
    training_stats: DARTSTrainingStats<T>,
    
    /// Current epoch
    current_epoch: usize,
    
    /// Best architecture found
    best_architecture: Option<ArchitectureCandidate<T>>,
    
    /// Progressive pruning tracker
    pruning_tracker: PruningTracker<T>,
}

/// Search space for DARTS
#[derive(Debug, Clone)]
pub struct SearchSpace<T: Float> {
    /// Nodes in the computational graph
    pub nodes: Vec<SearchNode>,
    
    /// Edges connecting nodes
    pub edges: Vec<SearchEdge>,
    
    /// Available operations
    pub operations: Vec<CandidateOperation<T>>,
    
    /// Edge-to-operations mapping
    pub edge_operations: HashMap<EdgeId, Vec<usize>>,
    
    /// Search space constraints
    pub constraints: SearchSpaceConstraints<T>,
}

/// Node in the search space
#[derive(Debug, Clone)]
pub struct SearchNode {
    /// Node identifier
    pub id: NodeId,
    
    /// Node type
    pub node_type: NodeType,
    
    /// Input dimensions
    pub input_dims: Vec<usize>,
    
    /// Output dimensions
    pub output_dims: Vec<usize>,
    
    /// Node-specific parameters
    pub parameters: HashMap<String, f64>,
}

/// Edge in the search space
#[derive(Debug, Clone)]
pub struct SearchEdge {
    /// Edge identifier
    pub id: EdgeId,
    
    /// Source node
    pub from_node: NodeId,
    
    /// Target node
    pub to_node: NodeId,
    
    /// Edge weight
    pub weight: f64,
    
    /// Edge constraints
    pub constraints: EdgeConstraints,
}

/// Types of nodes in search space
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Intermediate,
    Output,
    Skip,
}

/// Edge constraints
#[derive(Debug, Clone)]
pub struct EdgeConstraints {
    /// Maximum operations per edge
    pub max_operations: usize,
    
    /// Allowed operation types
    pub allowed_ops: HashSet<String>,
    
    /// Resource constraints
    pub resource_limits: ResourceLimits,
}

/// Resource limits for edges
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum parameters
    pub max_parameters: usize,
    
    /// Maximum FLOPs
    pub max_flops: u64,
    
    /// Maximum memory
    pub max_memory: usize,
}

/// Candidate operations for DARTS
#[derive(Debug, Clone)]
pub struct CandidateOperation<T: Float> {
    /// Operation identifier
    pub id: String,
    
    /// Operation type
    pub op_type: OperationType,
    
    /// Operation parameters
    pub parameters: HashMap<String, T>,
    
    /// Computational cost
    pub cost: ComputationalCost<T>,
    
    /// Operation implementation
    pub implementation: OperationImpl<T>,
}

/// Types of operations
#[derive(Debug, Clone)]
pub enum OperationType {
    /// Convolution operation
    Conv { kernel_size: usize, stride: usize },
    
    /// Dense/Linear operation
    Dense { units: usize },
    
    /// Pooling operation
    Pool { pool_type: PoolType, kernel_size: usize },
    
    /// Skip connection
    Skip,
    
    /// Zero operation (no connection)
    Zero,
    
    /// Batch normalization
    BatchNorm,
    
    /// Activation function
    Activation { activation: ActivationType },
    
    /// Attention mechanism
    Attention { heads: usize, dim: usize },
    
    /// Custom optimizer operation
    OptimizerOp { optimizer_type: String },
}

/// Pool types
#[derive(Debug, Clone, Copy)]
pub enum PoolType {
    Max,
    Average,
    Global,
}

/// Activation types
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    Tanh,
    Sigmoid,
    GELU,
    Swish,
}

/// Computational cost of operations
#[derive(Debug, Clone)]
pub struct ComputationalCost<T: Float> {
    /// Parameter count
    pub parameters: usize,
    
    /// FLOPs estimate
    pub flops: u64,
    
    /// Memory usage
    pub memory: usize,
    
    /// Relative cost weight
    pub cost_weight: T,
}

/// Operation implementation
#[derive(Debug, Clone)]
pub struct OperationImpl<T: Float> {
    /// Forward pass implementation
    pub forward_fn: String, // Function name/identifier
    
    /// Backward pass implementation
    pub backward_fn: String,
    
    /// Parameter initialization
    pub init_params: HashMap<String, T>,
}

/// Mixed operation combining multiple candidates
#[derive(Debug)]
pub struct MixedOperation<T: Float> {
    /// Edge this operation belongs to
    pub edge_id: EdgeId,
    
    /// Candidate operations
    pub candidates: Vec<CandidateOperation<T>>,
    
    /// Architecture weights (alpha)
    pub arch_weights: Array1<T>,
    
    /// Softmax probabilities
    pub probabilities: Array1<T>,
    
    /// Gumbel noise for sampling
    pub gumbel_noise: Option<Array1<T>>,
    
    /// Current temperature
    pub temperature: T,
}

/// Architecture candidate
#[derive(Debug, Clone)]
pub struct ArchitectureCandidate<T: Float> {
    /// Architecture identifier
    pub id: String,
    
    /// Selected operations for each edge
    pub operations: HashMap<EdgeId, usize>,
    
    /// Architecture parameters
    pub arch_params: Array3<T>,
    
    /// Performance metrics
    pub performance: PerformanceMetrics<T>,
    
    /// Architecture complexity
    pub complexity: ArchitectureComplexity<T>,
    
    /// Generation method
    pub generation_method: GenerationMethod,
}

/// Performance metrics for architectures
#[derive(Debug, Clone)]
pub struct PerformanceMetrics<T: Float> {
    /// Validation accuracy/performance
    pub validation_performance: T,
    
    /// Training performance
    pub training_performance: T,
    
    /// Convergence speed
    pub convergence_speed: T,
    
    /// Training stability
    pub stability: T,
    
    /// Generalization score
    pub generalization: T,
}

/// Architecture complexity metrics
#[derive(Debug, Clone)]
pub struct ArchitectureComplexity<T: Float> {
    /// Total parameters
    pub total_parameters: usize,
    
    /// Total FLOPs
    pub total_flops: u64,
    
    /// Memory usage
    pub memory_usage: usize,
    
    /// Architectural depth
    pub depth: usize,
    
    /// Connectivity complexity
    pub connectivity: T,
    
    /// Operation diversity
    pub diversity: T,
}

/// How architecture was generated
#[derive(Debug, Clone, Copy)]
pub enum GenerationMethod {
    /// Derived from softmax
    Derived,
    /// Sampled with Gumbel
    Sampled,
    /// Manually specified
    Manual,
}

/// Training statistics for DARTS
#[derive(Debug, Clone)]
pub struct DARTSTrainingStats<T: Float> {
    /// Architecture parameter gradients
    pub arch_grad_norms: Vec<T>,
    
    /// Weight gradients
    pub weight_grad_norms: Vec<T>,
    
    /// Architecture losses
    pub arch_losses: Vec<T>,
    
    /// Weight losses
    pub weight_losses: Vec<T>,
    
    /// Validation accuracies
    pub validation_accuracies: Vec<T>,
    
    /// Temperature history
    pub temperature_history: Vec<T>,
    
    /// Entropy of architecture distribution
    pub entropy_history: Vec<T>,
}

/// Progressive pruning tracker
#[derive(Debug)]
pub struct PruningTracker<T: Float> {
    /// Operations pruned per epoch
    pub pruned_operations: HashMap<usize, Vec<(EdgeId, usize)>>,
    
    /// Pruning scores for operations
    pub pruning_scores: HashMap<EdgeId, Array1<T>>,
    
    /// Pruning thresholds over time
    pub threshold_history: Vec<T>,
    
    /// Number of remaining operations per edge
    pub remaining_ops: HashMap<EdgeId, usize>,
}

/// Search space constraints
#[derive(Debug, Clone)]
pub struct SearchSpaceConstraints<T: Float> {
    /// Maximum total parameters
    pub max_total_parameters: usize,
    
    /// Maximum depth
    pub max_depth: usize,
    
    /// Maximum operations per edge
    pub max_ops_per_edge: usize,
    
    /// Resource budget
    pub resource_budget: T,
}

/// Type aliases for clarity
pub type NodeId = String;
pub type EdgeId = String;

impl<T: Float + Default + Clone> DARTSSearcher<T> {
    /// Create new DARTS searcher
    pub fn new(config: GradientBasedNASConfig<T>, search_space: SearchSpace<T>) -> Result<Self> {
        let num_edges = search_space.edges.len();
        let num_ops = config.num_candidate_ops;
        
        // Initialize architecture parameters
        let architecture_parameters = Array3::from_shape_fn((num_edges, num_ops, 1), |_| {
            T::from(rand::random::<f64>() * 0.1).unwrap()
        });
        
        // Initialize mixed operations
        let mut mixed_operations = HashMap::new();
        for edge in &search_space.edges {
            if let Some(op_indices) = search_space.edge_operations.get(&edge.id) {
                let candidates = op_indices.iter()
                    .map(|&idx| search_space.operations[idx].clone())
                    .collect();
                
                let arch_weights = Array1::zeros(candidates.len());
                let mixed_op = MixedOperation {
                    edge_id: edge.id.clone(),
                    candidates,
                    arch_weights,
                    probabilities: Array1::zeros(op_indices.len()),
                    gumbel_noise: None,
                    temperature: config.temperature,
                };
                mixed_operations.insert(edge.id.clone(), mixed_op);
            }
        }
        
        Ok(Self {
            config,
            architecture_parameters,
            search_space,
            mixed_operations,
            architecture_history: Vec::new(),
            training_stats: DARTSTrainingStats::new(),
            current_epoch: 0,
            best_architecture: None,
            pruning_tracker: PruningTracker::new(),
        })
    }
    
    /// Run DARTS architecture search
    pub fn search(&mut self, objective_fn: &dyn Fn(&ArchitectureCandidate<T>) -> Result<T>) -> Result<ArchitectureCandidate<T>> {
        println!("Starting DARTS architecture search for {} epochs", self.config.search_epochs);
        
        for epoch in 0..self.config.search_epochs {
            self.current_epoch = epoch;
            
            // Warm-up phase: only train weights
            if epoch < self.config.warmup_epochs {
                self.warmup_step(objective_fn)?;
            } else {
                // Joint optimization of architecture and weights
                self.joint_optimization_step(objective_fn)?;
            }
            
            // Update temperature
            self.update_temperature();
            
            // Progressive pruning
            if epoch > self.config.warmup_epochs {
                self.progressive_pruning()?;
            }
            
            // Derive current architecture
            let current_arch = self.derive_architecture()?;
            let performance = objective_fn(&current_arch)?;
            
            // Update best architecture
            if let Some(ref best) = self.best_architecture {
                if performance > best.performance.validation_performance {
                    self.best_architecture = Some(current_arch.clone());
                }
            } else {
                self.best_architecture = Some(current_arch.clone());
            }
            
            // Record statistics
            self.record_epoch_stats(performance)?;
            
            // Early stopping check
            if self.should_early_stop() {
                println!("Early stopping at epoch {} due to convergence", epoch);
                break;
            }
            
            // Progress logging
            if epoch % 10 == 0 {
                println!("DARTS Epoch {}: Performance = {:.4}, Temperature = {:.4}",
                    epoch,
                    performance.to_f64().unwrap_or(0.0),
                    self.config.temperature.to_f64().unwrap_or(1.0)
                );
            }
        }
        
        self.best_architecture.clone().ok_or_else(|| 
            OptimError::SearchFailed("No valid architecture found".to_string())
        )
    }
    
    /// Warm-up training step (weights only)
    fn warmup_step(&mut self, objective_fn: &dyn Fn(&ArchitectureCandidate<T>) -> Result<T>) -> Result<()> {
        // Sample architecture from current distribution
        let sampled_arch = self.sample_architecture()?;
        
        // Evaluate architecture
        let performance = objective_fn(&sampled_arch)?;
        
        // Update weight parameters (simplified)
        self.update_weights(performance)?;
        
        Ok(())
    }
    
    /// Joint optimization step (architecture + weights)
    fn joint_optimization_step(&mut self, objective_fn: &dyn Fn(&ArchitectureCandidate<T>) -> Result<T>) -> Result<()> {
        // Step 1: Update architecture parameters
        let arch_gradient = self.compute_architecture_gradient(objective_fn)?;
        self.update_architecture_parameters(arch_gradient)?;
        
        // Step 2: Update weight parameters
        let weight_gradient = self.compute_weight_gradient(objective_fn)?;
        self.update_weight_parameters(weight_gradient)?;
        
        Ok(())
    }
    
    /// Compute gradient for architecture parameters
    fn compute_architecture_gradient(&self, objective_fn: &dyn Fn(&ArchitectureCandidate<T>) -> Result<T>) -> Result<Array3<T>> {
        let mut gradient = Array3::zeros(self.architecture_parameters.dim());
        
        // Finite difference approximation (simplified)
        let epsilon = T::from(1e-4).unwrap();
        
        for i in 0..gradient.shape()[0] {
            for j in 0..gradient.shape()[1] {
                for k in 0..gradient.shape()[2] {
                    // Forward difference
                    let mut params_plus = self.architecture_parameters.clone();
                    params_plus[[i, j, k]] = params_plus[[i, j, k]] + epsilon;
                    
                    let arch_plus = self.params_to_architecture(&params_plus)?;
                    let perf_plus = objective_fn(&arch_plus)?;
                    
                    // Backward difference
                    let mut params_minus = self.architecture_parameters.clone();
                    params_minus[[i, j, k]] = params_minus[[i, j, k]] - epsilon;
                    
                    let arch_minus = self.params_to_architecture(&params_minus)?;
                    let perf_minus = objective_fn(&arch_minus)?;
                    
                    // Gradient approximation
                    gradient[[i, j, k]] = (perf_plus - perf_minus) / (T::from(2.0).unwrap() * epsilon);
                }
            }
        }
        
        Ok(gradient)
    }
    
    /// Compute gradient for weight parameters
    fn compute_weight_gradient(&self, objective_fn: &dyn Fn(&ArchitectureCandidate<T>) -> Result<T>) -> Result<T> {
        // Simplified weight gradient computation
        let current_arch = self.derive_architecture()?;
        let performance = objective_fn(&current_arch)?;
        
        // Return scalar gradient (in practice would be tensor)
        Ok(performance * T::from(0.01).unwrap())
    }
    
    /// Update architecture parameters with gradient
    fn update_architecture_parameters(&mut self, gradient: Array3<T>) -> Result<()> {
        let learning_rate = self.config.arch_learning_rate;
        let weight_decay = self.config.arch_weight_decay;
        
        for ((i, j, k), &grad) in gradient.indexed_iter() {
            let current_param = self.architecture_parameters[[i, j, k]];
            
            // Gradient descent with weight decay
            let update = learning_rate * grad + weight_decay * current_param;
            self.architecture_parameters[[i, j, k]] = current_param - update;
        }
        
        Ok(())
    }
    
    /// Update weight parameters
    fn update_weight_parameters(&mut self, gradient: T) -> Result<()> {
        // Simplified weight parameter update
        // In practice, this would update actual model weights
        Ok(())
    }
    
    /// Update mixed operations based on current parameters
    fn update_mixed_operations(&mut self) -> Result<()> {
        for (edge_id, mixed_op) in &mut self.mixed_operations {
            // Update architecture weights from parameters
            if let Some(edge_idx) = self.get_edge_index(edge_id) {
                for i in 0..mixed_op.arch_weights.len() {
                    mixed_op.arch_weights[i] = self.architecture_parameters[[edge_idx, i, 0]];
                }
                
                // Compute softmax probabilities
                self.compute_softmax_probabilities(mixed_op)?;
            }
        }
        
        Ok(())
    }
    
    /// Compute softmax probabilities for mixed operation
    fn compute_softmax_probabilities(&self, mixed_op: &mut MixedOperation<T>) -> Result<()> {
        let temperature = mixed_op.temperature;
        let weights = &mixed_op.arch_weights;
        
        // Apply temperature scaling
        let scaled_weights = weights.mapv(|w| w / temperature);
        
        // Compute softmax
        let max_weight = scaled_weights.iter().cloned().fold(T::from(f64::NEG_INFINITY).unwrap(), T::max);
        let exp_weights = scaled_weights.mapv(|w| (w - max_weight).exp());
        let sum_exp = exp_weights.sum();
        
        mixed_op.probabilities = exp_weights / sum_exp;
        
        Ok(())
    }
    
    /// Sample architecture from current distribution
    fn sample_architecture(&mut self) -> Result<ArchitectureCandidate<T>> {
        self.update_mixed_operations()?;
        
        let mut operations = HashMap::new();
        let mut arch_params = self.architecture_parameters.clone();
        
        for (edge_id, mixed_op) in &mut self.mixed_operations {
            // Gumbel-Softmax sampling
            let gumbel_noise = self.sample_gumbel_noise(mixed_op.probabilities.len())?;
            mixed_op.gumbel_noise = Some(gumbel_noise.clone());
            
            let gumbel_softmax = &mixed_op.probabilities + &gumbel_noise;
            let argmax = gumbel_softmax.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            operations.insert(edge_id.clone(), argmax);
        }
        
        Ok(ArchitectureCandidate {
            id: format!("sampled_epoch_{}", self.current_epoch),
            operations,
            arch_params,
            performance: PerformanceMetrics::default(),
            complexity: self.compute_architecture_complexity(&HashMap::new())?,
            generation_method: GenerationMethod::Sampled,
        })
    }
    
    /// Derive discrete architecture from continuous parameters
    fn derive_architecture(&self) -> Result<ArchitectureCandidate<T>> {
        let mut operations = HashMap::new();
        
        for (edge_id, mixed_op) in &self.mixed_operations {
            // Select operation with highest probability
            let best_op_idx = mixed_op.probabilities.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            operations.insert(edge_id.clone(), best_op_idx);
        }
        
        Ok(ArchitectureCandidate {
            id: format!("derived_epoch_{}", self.current_epoch),
            operations: operations.clone(),
            arch_params: self.architecture_parameters.clone(),
            performance: PerformanceMetrics::default(),
            complexity: self.compute_architecture_complexity(&operations)?,
            generation_method: GenerationMethod::Derived,
        })
    }
    
    /// Convert parameters to architecture candidate
    fn params_to_architecture(&self, params: &Array3<T>) -> Result<ArchitectureCandidate<T>> {
        // Create temporary architecture from parameters
        let mut operations = HashMap::new();
        
        for (i, (edge_id, mixed_op)) in self.mixed_operations.iter().enumerate() {
            if i < params.shape()[0] {
                let edge_params = params.slice(ndarray::s![i, .., 0]);
                let best_idx = edge_params.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                operations.insert(edge_id.clone(), best_idx);
            }
        }
        
        Ok(ArchitectureCandidate {
            id: format!("temp_arch_{}", self.current_epoch),
            operations: operations.clone(),
            arch_params: params.clone(),
            performance: PerformanceMetrics::default(),
            complexity: self.compute_architecture_complexity(&operations)?,
            generation_method: GenerationMethod::Manual,
        })
    }
    
    /// Sample Gumbel noise
    fn sample_gumbel_noise(&self, size: usize) -> Result<Array1<T>> {
        use rand::Rng;
        let mut rng = rand::rng();
        
        let noise = Array1::from_shape_fn(size, |_| {
            let u: f64 = rng.gen_range(1e-10..1.0);
            let gumbel = -(-u.ln()).ln();
            T::from(gumbel).unwrap()
        });
        
        Ok(noise)
    }
    
    /// Update temperature for annealing
    fn update_temperature(&mut self) {
        self.config.temperature = (self.config.temperature * self.config.temperature_decay)
            .max(self.config.min_temperature);
        
        // Update temperature in mixed operations
        for mixed_op in self.mixed_operations.values_mut() {
            mixed_op.temperature = self.config.temperature;
        }
    }
    
    /// Progressive pruning of low-weight operations
    fn progressive_pruning(&mut self) -> Result<()> {
        let threshold = self.config.pruning_threshold;
        
        for (edge_id, mixed_op) in &mut self.mixed_operations {
            let mut pruned_ops = Vec::new();
            
            for (i, &prob) in mixed_op.probabilities.iter().enumerate() {
                if prob < threshold && mixed_op.candidates.len() > 1 {
                    pruned_ops.push(i);
                }
            }
            
            // Remove pruned operations (simplified)
            self.pruning_tracker.pruned_operations
                .entry(self.current_epoch)
                .or_insert_with(Vec::new)
                .extend(pruned_ops.iter().map(|&i| (edge_id.clone(), i)));
        }
        
        Ok(())
    }
    
    /// Check if early stopping should occur
    fn should_early_stop(&self) -> bool {
        if self.training_stats.validation_accuracies.len() < self.config.early_stopping_patience {
            return false;
        }
        
        let recent_accs = &self.training_stats.validation_accuracies[
            self.training_stats.validation_accuracies.len().saturating_sub(self.config.early_stopping_patience)..
        ];
        
        let mut is_improving = false;
        for i in 1..recent_accs.len() {
            if recent_accs[i] > recent_accs[i-1] {
                is_improving = true;
                break;
            }
        }
        
        !is_improving
    }
    
    /// Record statistics for current epoch
    fn record_epoch_stats(&mut self, performance: T) -> Result<()> {
        self.training_stats.validation_accuracies.push(performance);
        self.training_stats.temperature_history.push(self.config.temperature);
        
        // Compute entropy of architecture distribution
        let entropy = self.compute_architecture_entropy()?;
        self.training_stats.entropy_history.push(entropy);
        
        Ok(())
    }
    
    /// Compute entropy of architecture distribution
    fn compute_architecture_entropy(&self) -> Result<T> {
        let mut total_entropy = T::zero();
        let mut num_edges = 0;
        
        for mixed_op in self.mixed_operations.values() {
            let mut edge_entropy = T::zero();
            for &p in mixed_op.probabilities.iter() {
                if p > T::zero() {
                    edge_entropy = edge_entropy - p * p.ln();
                }
            }
            total_entropy = total_entropy + edge_entropy;
            num_edges += 1;
        }
        
        Ok(total_entropy / T::from(num_edges as f64).unwrap())
    }
    
    /// Compute architecture complexity
    fn compute_architecture_complexity(&self, operations: &HashMap<EdgeId, usize>) -> Result<ArchitectureComplexity<T>> {
        let mut total_parameters = 0;
        let mut total_flops = 0;
        let mut total_memory = 0;
        let mut operation_types = HashSet::new();
        
        for (edge_id, &op_idx) in operations {
            if let Some(mixed_op) = self.mixed_operations.get(edge_id) {
                if op_idx < mixed_op.candidates.len() {
                    let op = &mixed_op.candidates[op_idx];
                    total_parameters += op.cost.parameters;
                    total_flops += op.cost.flops;
                    total_memory += op.cost.memory;
                    operation_types.insert(op.id.clone());
                }
            }
        }
        
        let diversity = T::from(operation_types.len() as f64).unwrap();
        let depth = operations.len();
        
        Ok(ArchitectureComplexity {
            total_parameters,
            total_flops,
            memory_usage: total_memory,
            depth,
            connectivity: T::from(operations.len() as f64).unwrap(),
            diversity,
        })
    }
    
    /// Get edge index for parameter lookup
    fn get_edge_index(&self, edge_id: &EdgeId) -> Option<usize> {
        self.search_space.edges.iter()
            .position(|edge| &edge.id == edge_id)
    }
    
    /// Update weights (simplified implementation)
    fn update_weights(&mut self, performance: T) -> Result<()> {
        // Simplified weight update - in practice would update actual model weights
        Ok(())
    }
    
    /// Get current best architecture
    pub fn best_architecture(&self) -> Option<&ArchitectureCandidate<T>> {
        self.best_architecture.as_ref()
    }
    
    /// Get training statistics
    pub fn training_statistics(&self) -> &DARTSTrainingStats<T> {
        &self.training_stats
    }
    
    /// Get search space
    pub fn search_space(&self) -> &SearchSpace<T> {
        &self.search_space
    }
}

impl<T: Float + Default + Clone> DARTSTrainingStats<T> {
    fn new() -> Self {
        Self {
            arch_grad_norms: Vec::new(),
            weight_grad_norms: Vec::new(),
            arch_losses: Vec::new(),
            weight_losses: Vec::new(),
            validation_accuracies: Vec::new(),
            temperature_history: Vec::new(),
            entropy_history: Vec::new(),
        }
    }
}

impl<T: Float + Default + Clone> PruningTracker<T> {
    fn new() -> Self {
        Self {
            pruned_operations: HashMap::new(),
            pruning_scores: HashMap::new(),
            threshold_history: Vec::new(),
            remaining_ops: HashMap::new(),
        }
    }
}

impl<T: Float + Default + Clone> Default for PerformanceMetrics<T> {
    fn default() -> Self {
        Self {
            validation_performance: T::zero(),
            training_performance: T::zero(),
            convergence_speed: T::zero(),
            stability: T::zero(),
            generalization: T::zero(),
        }
    }
}

impl<T: Float + Default + Clone> Default for GradientBasedNASConfig<T> {
    fn default() -> Self {
        Self {
            arch_learning_rate: T::from(3e-4).unwrap(),
            weight_learning_rate: T::from(0.025).unwrap(),
            arch_weight_decay: T::from(1e-3).unwrap(),
            weight_decay: T::from(3e-4).unwrap(),
            search_epochs: 50,
            warmup_epochs: 15,
            temperature: T::from(1.0).unwrap(),
            temperature_decay: T::from(0.96).unwrap(),
            min_temperature: T::from(0.1).unwrap(),
            arch_regularization: T::from(1e-3).unwrap(),
            early_stopping_patience: 10,
            pruning_threshold: T::from(0.01).unwrap(),
            num_candidate_ops: 8,
        }
    }
}

impl Default for EdgeConstraints {
    fn default() -> Self {
        Self {
            max_operations: 8,
            allowed_ops: HashSet::new(),
            resource_limits: ResourceLimits {
                max_parameters: 1_000_000,
                max_flops: 1_000_000_000,
                max_memory: 1_000_000,
            },
        }
    }
}