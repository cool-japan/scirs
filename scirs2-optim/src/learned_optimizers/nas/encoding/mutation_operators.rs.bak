//! Mutation operators for neural architecture search
//!
//! This module provides various mutation operators that can modify architectures
//! during the search process to explore the search space effectively.

#![allow(dead_code)]

use ndarray::{Array1, Array2};
use num_traits::Float;
use std::collections::{HashMap, HashSet, VecDeque};

use crate::error::{OptimError, Result};

/// Configuration for mutation operators
#[derive(Debug, Clone)]
pub struct MutationConfig<T: Float> {
    /// Mutation probability
    pub mutation_probability: T,
    
    /// Maximum mutations per architecture
    pub max_mutations_per_arch: usize,
    
    /// Mutation strength (how drastic the mutations are)
    pub mutation_strength: T,
    
    /// Available mutation operators
    pub available_operators: Vec<MutationType>,
    
    /// Operator weights for selection
    pub operator_weights: HashMap<MutationType, T>,
    
    /// Adaptive mutation parameters
    pub adaptive_params: Option<AdaptiveMutationParams<T>>,
    
    /// Constraint preservation settings
    pub preserve_constraints: bool,
    
    /// Mutation history tracking
    pub track_history: bool,
}

/// Types of mutation operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MutationType {
    /// Add a new operation/layer
    AddOperation,
    
    /// Remove an existing operation/layer
    RemoveOperation,
    
    /// Replace an operation with another
    ReplaceOperation,
    
    /// Modify operation parameters
    ModifyParameters,
    
    /// Add a new connection/edge
    AddConnection,
    
    /// Remove an existing connection
    RemoveConnection,
    
    /// Modify connection weights
    ModifyConnection,
    
    /// Change layer depth
    ChangeDepth,
    
    /// Change layer width
    ChangeWidth,
    
    /// Duplicate a block/module
    DuplicateBlock,
    
    /// Remove a block/module
    RemoveBlock,
    
    /// Swap two operations
    SwapOperations,
    
    /// Insert skip connection
    InsertSkip,
    
    /// Remove skip connection
    RemoveSkip,
    
    /// Modify activation function
    ChangeActivation,
    
    /// Modify normalization
    ChangeNormalization,
    
    /// Split an operation into multiple
    SplitOperation,
    
    /// Merge multiple operations
    MergeOperations,
    
    /// Randomize architecture section
    RandomizeSection,
    
    /// Apply macro-level changes
    MacroMutation,
}

/// Adaptive mutation parameters
#[derive(Debug, Clone)]
pub struct AdaptiveMutationParams<T: Float> {
    /// Learning rate for adaptation
    pub adaptation_rate: T,
    
    /// Success-based adaptation
    pub success_threshold: T,
    
    /// Failure penalty
    pub failure_penalty: T,
    
    /// Adaptation window size
    pub window_size: usize,
    
    /// Minimum mutation rate
    pub min_mutation_rate: T,
    
    /// Maximum mutation rate
    pub max_mutation_rate: T,
}

/// Architecture mutation operator
#[derive(Debug)]
pub struct ArchitectureMutator<T: Float> {
    /// Configuration
    config: MutationConfig<T>,
    
    /// Available operations pool
    operation_pool: OperationPool<T>,
    
    /// Mutation operators
    operators: HashMap<MutationType, Box<dyn MutationOperator<T>>>,
    
    /// Mutation history
    mutation_history: MutationHistory<T>,
    
    /// Performance feedback for adaptation
    performance_feedback: PerformanceFeedback<T>,
    
    /// Constraint validator
    constraint_validator: Option<Box<dyn ConstraintValidator<T>>>,
}

/// Pool of available operations for mutations
#[derive(Debug, Clone)]
pub struct OperationPool<T: Float> {
    /// Available operation types
    pub operation_types: Vec<OperationType>,
    
    /// Operation templates with default parameters
    pub operation_templates: HashMap<String, OperationTemplate<T>>,
    
    /// Compatibility matrix
    pub compatibility_matrix: Array2<bool>,
    
    /// Operation frequencies (for biased sampling)
    pub operation_frequencies: HashMap<String, T>,
}

/// Operation types available for mutation
#[derive(Debug, Clone)]
pub struct OperationType {
    /// Type identifier
    pub id: String,
    
    /// Type name
    pub name: String,
    
    /// Operation category
    pub category: OperationCategory,
    
    /// Parameter ranges
    pub parameter_ranges: HashMap<String, ParameterRange>,
    
    /// Computational cost estimate
    pub cost_estimate: CostEstimate,
}

/// Operation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationCategory {
    Convolution, Dense, Normalization, Activation, 
    Pooling, Attention, Recurrent, Custom,
}

/// Parameter range for operation parameters
#[derive(Debug, Clone)]
pub enum ParameterRange {
    /// Integer range
    Integer { min: i64, max: i64, default: i64 },
    
    /// Float range
    Float { min: f64, max: f64, default: f64 },
    
    /// Discrete choices
    Discrete { choices: Vec<String>, default: String },
    
    /// Boolean parameter
    Boolean { default: bool },
}

/// Cost estimate for operations
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Parameter count multiplier
    pub param_multiplier: f64,
    
    /// FLOPS multiplier
    pub flops_multiplier: f64,
    
    /// Memory multiplier
    pub memory_multiplier: f64,
    
    /// Latency multiplier
    pub latency_multiplier: f64,
}

/// Operation template with default configuration
#[derive(Debug, Clone)]
pub struct OperationTemplate<T: Float> {
    /// Template identifier
    pub id: String,
    
    /// Operation type
    pub operation_type: String,
    
    /// Default parameters
    pub default_parameters: HashMap<String, T>,
    
    /// Parameter constraints
    pub constraints: Vec<ParameterConstraint<T>>,
    
    /// Compatibility information
    pub compatibility: CompatibilityInfo,
}

/// Parameter constraints
#[derive(Debug, Clone)]
pub enum ParameterConstraint<T: Float> {
    /// Range constraint
    Range { param: String, min: T, max: T },
    
    /// Dependency constraint
    Dependency { param: String, depends_on: String, condition: String },
    
    /// Mutual exclusion
    MutualExclusion { params: Vec<String> },
    
    /// Custom constraint
    Custom { name: String, description: String },
}

/// Compatibility information
#[derive(Debug, Clone)]
pub struct CompatibilityInfo {
    /// Compatible input types
    pub input_types: Vec<String>,
    
    /// Compatible output types
    pub output_types: Vec<String>,
    
    /// Shape requirements
    pub shape_requirements: Vec<ShapeRequirement>,
    
    /// Incompatible operations
    pub incompatible_with: Vec<String>,
}

/// Shape requirements for operations
#[derive(Debug, Clone)]
pub enum ShapeRequirement {
    /// Exact shape
    Exact(Vec<usize>),
    
    /// Minimum dimensions
    MinDims(usize),
    
    /// Maximum dimensions
    MaxDims(usize),
    
    /// Divisible by
    DivisibleBy { dim: usize, factor: usize },
    
    /// Shape relationship
    Relationship { relation: String, factor: f64 },
}

/// Trait for mutation operators
pub trait MutationOperator<T: Float> {
    /// Apply mutation to architecture
    fn mutate(&self, architecture: &ArchitectureGenotype<T>) -> Result<ArchitectureGenotype<T>>;
    
    /// Check if mutation is applicable
    fn is_applicable(&self, architecture: &ArchitectureGenotype<T>) -> bool;
    
    /// Get mutation metadata
    fn get_metadata(&self) -> MutationMetadata;
    
    /// Get estimated impact
    fn estimate_impact(&self, architecture: &ArchitectureGenotype<T>) -> MutationImpact<T>;
}

/// Architecture genotype representation
#[derive(Debug, Clone)]
pub struct ArchitectureGenotype<T: Float> {
    /// Architecture identifier
    pub id: String,
    
    /// Operations in the architecture
    pub operations: Vec<OperationGene<T>>,
    
    /// Connections between operations
    pub connections: Vec<ConnectionGene<T>>,
    
    /// Global parameters
    pub global_parameters: HashMap<String, T>,
    
    /// Architecture metadata
    pub metadata: GenotypeMetadata,
    
    /// Constraint satisfaction status
    pub is_valid: bool,
}

/// Operation gene representation
#[derive(Debug, Clone)]
pub struct OperationGene<T: Float> {
    /// Gene identifier
    pub id: String,
    
    /// Operation type
    pub operation_type: String,
    
    /// Operation parameters
    pub parameters: HashMap<String, T>,
    
    /// Position in architecture
    pub position: Position,
    
    /// Gene-specific metadata
    pub metadata: GeneMetadata,
}

/// Connection gene representation
#[derive(Debug, Clone)]
pub struct ConnectionGene<T: Float> {
    /// Gene identifier
    pub id: String,
    
    /// Source operation
    pub source: String,
    
    /// Target operation
    pub target: String,
    
    /// Connection type
    pub connection_type: ConnectionType,
    
    /// Connection weight/strength
    pub weight: T,
    
    /// Connection metadata
    pub metadata: GeneMetadata,
}

/// Connection types
#[derive(Debug, Clone, Copy)]
pub enum ConnectionType {
    Standard, Skip, Residual, Dense, Attention,
}

/// Position in architecture
#[derive(Debug, Clone)]
pub struct Position {
    /// Layer index
    pub layer: usize,
    
    /// Position within layer
    pub index: usize,
    
    /// Depth in hierarchy
    pub depth: usize,
}

/// Genotype metadata
#[derive(Debug, Clone)]
pub struct GenotypeMetadata {
    /// Creation timestamp
    pub created_at: u64,
    
    /// Parent architectures
    pub parents: Vec<String>,
    
    /// Generation number
    pub generation: usize,
    
    /// Mutation history
    pub mutations_applied: Vec<String>,
}

/// Gene metadata
#[derive(Debug, Clone)]
pub struct GeneMetadata {
    /// Gene age (generations since creation)
    pub age: usize,
    
    /// Mutation count
    pub mutation_count: usize,
    
    /// Last mutation timestamp
    pub last_mutated: u64,
    
    /// Performance contribution
    pub performance_contribution: Option<f64>,
}

/// Mutation metadata
#[derive(Debug, Clone)]
pub struct MutationMetadata {
    /// Mutation operator name
    pub operator_name: String,
    
    /// Mutation type
    pub mutation_type: MutationType,
    
    /// Difficulty level
    pub difficulty: MutationDifficulty,
    
    /// Expected impact level
    pub expected_impact: ImpactLevel,
}

/// Mutation difficulty levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationDifficulty {
    Easy, Medium, Hard, Expert,
}

/// Impact levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImpactLevel {
    Minimal, Low, Medium, High, Critical,
}

/// Mutation impact estimate
#[derive(Debug, Clone)]
pub struct MutationImpact<T: Float> {
    /// Expected performance change
    pub performance_delta: T,
    
    /// Expected complexity change
    pub complexity_delta: T,
    
    /// Expected resource change
    pub resource_delta: HashMap<String, T>,
    
    /// Confidence in estimates
    pub confidence: T,
    
    /// Risk assessment
    pub risk_level: RiskLevel,
}

/// Risk levels for mutations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    VeryLow, Low, Medium, High, VeryHigh,
}

/// Mutation history tracking
#[derive(Debug)]
pub struct MutationHistory<T: Float> {
    /// Applied mutations log
    pub mutation_log: Vec<MutationRecord<T>>,
    
    /// Success statistics per operator
    pub operator_stats: HashMap<MutationType, OperatorStatistics<T>>,
    
    /// Rollback information
    pub rollback_info: HashMap<String, ArchitectureGenotype<T>>,
    
    /// Performance tracking
    pub performance_tracking: Vec<PerformanceRecord<T>>,
}

/// Mutation record
#[derive(Debug, Clone)]
pub struct MutationRecord<T: Float> {
    /// Record identifier
    pub id: String,
    
    /// Mutation type applied
    pub mutation_type: MutationType,
    
    /// Source architecture
    pub source_architecture: String,
    
    /// Resulting architecture
    pub result_architecture: String,
    
    /// Mutation parameters
    pub parameters: HashMap<String, String>,
    
    /// Performance before mutation
    pub performance_before: Option<T>,
    
    /// Performance after mutation
    pub performance_after: Option<T>,
    
    /// Success indicator
    pub success: Option<bool>,
    
    /// Timestamp
    pub timestamp: u64,
}

/// Statistics for mutation operators
#[derive(Debug, Clone)]
pub struct OperatorStatistics<T: Float> {
    /// Number of applications
    pub applications: usize,
    
    /// Number of successful mutations
    pub successes: usize,
    
    /// Average performance improvement
    pub avg_improvement: T,
    
    /// Success rate
    pub success_rate: T,
    
    /// Average impact magnitude
    pub avg_impact: T,
    
    /// Operator reliability score
    pub reliability_score: T,
}

/// Performance feedback for adaptive mutation
#[derive(Debug)]
pub struct PerformanceFeedback<T: Float> {
    /// Recent performance changes
    pub recent_changes: VecDeque<T>,
    
    /// Moving average performance
    pub moving_average: T,
    
    /// Performance trend
    pub trend: PerformanceTrend,
    
    /// Stagnation detection
    pub stagnation_counter: usize,
    
    /// Best performance seen
    pub best_performance: T,
}

/// Performance trend indicators
#[derive(Debug, Clone, Copy)]
pub enum PerformanceTrend {
    Improving, Declining, Stable, Volatile,
}

/// Performance record
#[derive(Debug, Clone)]
pub struct PerformanceRecord<T: Float> {
    /// Architecture identifier
    pub architecture_id: String,
    
    /// Performance value
    pub performance: T,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Context information
    pub context: HashMap<String, String>,
}

/// Constraint validator trait
pub trait ConstraintValidator<T: Float> {
    /// Validate architecture constraints
    fn validate(&self, architecture: &ArchitectureGenotype<T>) -> Result<bool>;
    
    /// Get constraint violations
    fn get_violations(&self, architecture: &ArchitectureGenotype<T>) -> Vec<String>;
    
    /// Suggest constraint fixes
    fn suggest_fixes(&self, architecture: &ArchitectureGenotype<T>) -> Vec<String>;
}

/// Specific mutation operators

/// Add operation mutation operator
#[derive(Debug)]
pub struct AddOperationMutator<T: Float> {
    /// Operation pool for selection
    operation_pool: OperationPool<T>,
    
    /// Insertion strategy
    insertion_strategy: InsertionStrategy,
    
    /// Maximum operations to add
    max_additions: usize,
}

/// Insertion strategies for new operations
#[derive(Debug, Clone, Copy)]
pub enum InsertionStrategy {
    /// Insert at random position
    Random,
    
    /// Insert at optimal position
    Optimal,
    
    /// Insert at end
    Append,
    
    /// Insert at beginning
    Prepend,
    
    /// Insert based on compatibility
    Compatible,
}

/// Remove operation mutation operator
#[derive(Debug)]
pub struct RemoveOperationMutator<T: Float> {
    /// Minimum operations to keep
    min_operations: usize,
    
    /// Removal strategy
    removal_strategy: RemovalStrategy,
    
    /// Preserve critical operations
    preserve_critical: bool,
    
    /// Critical operation identifiers
    critical_operations: HashSet<String>,
}

/// Removal strategies
#[derive(Debug, Clone, Copy)]
pub enum RemovalStrategy {
    /// Remove random operation
    Random,
    
    /// Remove least important operation
    LeastImportant,
    
    /// Remove redundant operations
    Redundant,
    
    /// Remove based on performance impact
    PerformanceBased,
}

/// Replace operation mutation operator
#[derive(Debug)]
pub struct ReplaceOperationMutator<T: Float> {
    /// Operation pool for replacements
    operation_pool: OperationPool<T>,
    
    /// Replacement strategy
    replacement_strategy: ReplacementStrategy,
    
    /// Compatibility checking
    check_compatibility: bool,
}

/// Replacement strategies
#[derive(Debug, Clone, Copy)]
pub enum ReplacementStrategy {
    /// Replace with random compatible operation
    RandomCompatible,
    
    /// Replace with similar operation
    Similar,
    
    /// Replace with better performing operation
    Performance,
    
    /// Replace with more efficient operation
    Efficiency,
}

/// Parameter modification mutation operator
#[derive(Debug)]
pub struct ParameterMutator<T: Float> {
    /// Mutation strength
    strength: T,
    
    /// Mutation distribution
    distribution: ParameterDistribution<T>,
    
    /// Parameter selection strategy
    selection_strategy: ParameterSelectionStrategy,
}

/// Parameter mutation distributions
#[derive(Debug, Clone)]
pub enum ParameterDistribution<T: Float> {
    /// Gaussian noise
    Gaussian { std: T },
    
    /// Uniform noise
    Uniform { range: T },
    
    /// Exponential scaling
    Exponential { factor: T },
    
    /// Discrete jumps
    Discrete { steps: Vec<T> },
}

/// Parameter selection strategies
#[derive(Debug, Clone, Copy)]
pub enum ParameterSelectionStrategy {
    /// Select random parameter
    Random,
    
    /// Select most sensitive parameter
    MostSensitive,
    
    /// Select all parameters
    All,
    
    /// Select based on importance
    ImportanceBased,
}

impl<T: Float + Default + Clone> ArchitectureMutator<T> {
    /// Create new architecture mutator
    pub fn new(config: MutationConfig<T>) -> Result<Self> {
        let mut mutator = Self {
            config: config.clone(),
            operation_pool: OperationPool::new()?,
            operators: HashMap::new(),
            mutation_history: MutationHistory::new(),
            performance_feedback: PerformanceFeedback::new(),
            constraint_validator: None,
        };
        
        // Initialize mutation operators
        mutator.initialize_operators()?;
        
        Ok(mutator)
    }
    
    /// Initialize mutation operators
    fn initialize_operators(&mut self) -> Result<()> {
        // Add operation mutator
        if self.config.available_operators.contains(&MutationType::AddOperation) {
            let add_mutator = AddOperationMutator {
                operation_pool: self.operation_pool.clone(),
                insertion_strategy: InsertionStrategy::Random,
                max_additions: 3,
            };
            self.operators.insert(MutationType::AddOperation, Box::new(add_mutator));
        }
        
        // Remove operation mutator
        if self.config.available_operators.contains(&MutationType::RemoveOperation) {
            let remove_mutator = RemoveOperationMutator {
                min_operations: 1,
                removal_strategy: RemovalStrategy::Random,
                preserve_critical: true,
                critical_operations: HashSet::new(),
            };
            self.operators.insert(MutationType::RemoveOperation, Box::new(remove_mutator));
        }
        
        // Replace operation mutator
        if self.config.available_operators.contains(&MutationType::ReplaceOperation) {
            let replace_mutator = ReplaceOperationMutator {
                operation_pool: self.operation_pool.clone(),
                replacement_strategy: ReplacementStrategy::RandomCompatible,
                check_compatibility: true,
            };
            self.operators.insert(MutationType::ReplaceOperation, Box::new(replace_mutator));
        }
        
        // Parameter mutator
        if self.config.available_operators.contains(&MutationType::ModifyParameters) {
            let param_mutator = ParameterMutator {
                strength: self.config.mutation_strength,
                distribution: ParameterDistribution::Gaussian { std: T::from(0.1).unwrap() },
                selection_strategy: ParameterSelectionStrategy::Random,
            };
            self.operators.insert(MutationType::ModifyParameters, Box::new(param_mutator));
        }
        
        Ok(())
    }
    
    /// Apply mutation to architecture
    pub fn mutate(&mut self, architecture: &ArchitectureGenotype<T>) -> Result<ArchitectureGenotype<T>> {
        // Select mutation operator
        let mutation_type = self.select_mutation_operator(architecture)?;
        
        // Apply mutation
        if let Some(operator) = self.operators.get(&mutation_type) {
            let mutated_architecture = operator.mutate(architecture)?;
            
            // Validate constraints if validator is available
            let is_valid = if let Some(ref validator) = self.constraint_validator {
                validator.validate(&mutated_architecture)?
            } else {
                true
            };
            
            let mut final_architecture = mutated_architecture;
            final_architecture.is_valid = is_valid;
            
            // Record mutation
            if self.config.track_history {
                self.record_mutation(mutation_type, architecture, &final_architecture)?;
            }
            
            // Update adaptive parameters
            if self.config.adaptive_params.is_some() {
                self.update_adaptive_parameters(mutation_type, is_valid)?;
            }
            
            Ok(final_architecture)
        } else {
            Err(OptimError::UnsupportedOperation(
                format!("Mutation operator {:?} not available", mutation_type)
            ))
        }
    }
    
    /// Select mutation operator based on strategy
    fn select_mutation_operator(&self, architecture: &ArchitectureGenotype<T>) -> Result<MutationType> {
        use rand::Rng;
        let mut rng = rand::rng();
        
        // Filter applicable operators
        let applicable_ops: Vec<MutationType> = self.config.available_operators
            .iter()
            .filter(|&&op_type| {
                if let Some(operator) = self.operators.get(&op_type) {
                    operator.is_applicable(architecture)
                } else {
                    false
                }
            })
            .copied()
            .collect();
        
        if applicable_ops.is_empty() {
            return Err(OptimError::SearchFailed(
                "No applicable mutation operators".to_string()
            ));
        }
        
        // Weighted selection
        if !self.config.operator_weights.is_empty() {
            let total_weight: T = applicable_ops.iter()
                .map(|op| self.config.operator_weights.get(op).unwrap_or(&T::one()))
                .cloned()
                .fold(T::zero(), |acc, w| acc + w);
            
            let mut cumulative_weight = T::zero();
            let random_weight = T::from(rng.gen::<f64>()).unwrap() * total_weight;
            
            for &op_type in &applicable_ops {
                let weight = self.config.operator_weights.get(&op_type).unwrap_or(&T::one());
                cumulative_weight = cumulative_weight + *weight;
                if random_weight <= cumulative_weight {
                    return Ok(op_type);
                }
            }
        }
        
        // Fallback to random selection
        let selected_idx = rng.gen_range(0..applicable_ops.len());
        Ok(applicable_ops[selected_idx])
    }
    
    /// Record mutation in history
    fn record_mutation(&mut self, mutation_type: MutationType, 
                      source: &ArchitectureGenotype<T>,
                      result: &ArchitectureGenotype<T>) -> Result<()> {
        let record = MutationRecord {
            id: format!("mut_{}", rand::random::<u64>()),
            mutation_type,
            source_architecture: source.id.clone(),
            result_architecture: result.id.clone(),
            parameters: HashMap::new(),
            performance_before: None,
            performance_after: None,
            success: Some(result.is_valid),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };
        
        self.mutation_history.mutation_log.push(record);
        
        // Update operator statistics
        let stats = self.mutation_history.operator_stats
            .entry(mutation_type)
            .or_insert_with(|| OperatorStatistics {
                applications: 0,
                successes: 0,
                avg_improvement: T::zero(),
                success_rate: T::zero(),
                avg_impact: T::zero(),
                reliability_score: T::zero(),
            });
        
        stats.applications += 1;
        if result.is_valid {
            stats.successes += 1;
        }
        stats.success_rate = T::from(stats.successes as f64 / stats.applications as f64).unwrap();
        
        Ok(())
    }
    
    /// Update adaptive mutation parameters
    fn update_adaptive_parameters(&mut self, mutation_type: MutationType, success: bool) -> Result<()> {
        if let Some(ref params) = self.config.adaptive_params {
            // Update operator weight based on success
            let current_weight = self.config.operator_weights
                .get(&mutation_type)
                .copied()
                .unwrap_or(T::one());
            
            let new_weight = if success {
                current_weight * (T::one() + params.adaptation_rate)
            } else {
                current_weight * (T::one() - params.failure_penalty)
            };
            
            let clamped_weight = new_weight.max(T::from(0.1).unwrap()).min(T::from(10.0).unwrap());
            self.config.operator_weights.insert(mutation_type, clamped_weight);
        }
        
        Ok(())
    }
    
    /// Apply multiple mutations to architecture
    pub fn mutate_multiple(&mut self, architecture: &ArchitectureGenotype<T>, 
                          num_mutations: usize) -> Result<ArchitectureGenotype<T>> {
        let mut current_arch = architecture.clone();
        
        for _ in 0..num_mutations.min(self.config.max_mutations_per_arch) {
            current_arch = self.mutate(&current_arch)?;
            
            // Stop early if mutation fails
            if !current_arch.is_valid {
                break;
            }
        }
        
        Ok(current_arch)
    }
    
    /// Get mutation statistics
    pub fn get_statistics(&self) -> &HashMap<MutationType, OperatorStatistics<T>> {
        &self.mutation_history.operator_stats
    }
    
    /// Get mutation history
    pub fn get_mutation_history(&self) -> &[MutationRecord<T>] {
        &self.mutation_history.mutation_log
    }
    
    /// Set constraint validator
    pub fn set_constraint_validator(&mut self, validator: Box<dyn ConstraintValidator<T>>) {
        self.constraint_validator = Some(validator);
    }
    
    /// Update performance feedback
    pub fn update_performance_feedback(&mut self, architecture_id: &str, performance: T) -> Result<()> {
        self.performance_feedback.recent_changes.push_back(performance);
        
        // Maintain window size
        let window_size = if let Some(ref params) = self.config.adaptive_params {
            params.window_size
        } else {
            10
        };
        
        while self.performance_feedback.recent_changes.len() > window_size {
            self.performance_feedback.recent_changes.pop_front();
        }
        
        // Update moving average
        let sum: T = self.performance_feedback.recent_changes.iter().cloned().sum();
        let count = T::from(self.performance_feedback.recent_changes.len() as f64).unwrap();
        self.performance_feedback.moving_average = sum / count;
        
        // Update best performance
        self.performance_feedback.best_performance = 
            self.performance_feedback.best_performance.max(performance);
        
        Ok(())
    }
}

// Implementations for specific mutation operators

impl<T: Float + Default + Clone> MutationOperator<T> for AddOperationMutator<T> {
    fn mutate(&self, architecture: &ArchitectureGenotype<T>) -> Result<ArchitectureGenotype<T>> {
        use rand::Rng;
        let mut rng = rand::rng();
        
        let mut mutated = architecture.clone();
        mutated.id = format!("{}_{}", architecture.id, "add_op");
        
        // Select operation to add
        let op_types: Vec<&OperationType> = self.operation_pool.operation_types.iter().collect();
        if op_types.is_empty() {
            return Err(OptimError::SearchFailed("No operations available to add".to_string()));
        }
        
        let selected_op = &op_types[rng.gen_range(0..op_types.len())];
        
        // Create new operation gene
        let new_op = OperationGene {
            id: format!("op_{}", rand::random::<u64>()),
            operation_type: selected_op.id.clone(),
            parameters: HashMap::new(), // Would be populated from template
            position: Position {
                layer: rng.gen_range(0..mutated.operations.len() + 1),
                index: 0,
                depth: 0,
            },
            metadata: GeneMetadata {
                age: 0,
                mutation_count: 0,
                last_mutated: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
                performance_contribution: None,
            },
        };
        
        // Insert operation
        match self.insertion_strategy {
            InsertionStrategy::Random => {
                let pos = rng.gen_range(0..=mutated.operations.len());
                mutated.operations.insert(pos, new_op);
            }
            InsertionStrategy::Append => {
                mutated.operations.push(new_op);
            }
            _ => {
                mutated.operations.push(new_op); // Fallback
            }
        }
        
        Ok(mutated)
    }
    
    fn is_applicable(&self, architecture: &ArchitectureGenotype<T>) -> bool {
        architecture.operations.len() < 100 // Arbitrary limit
    }
    
    fn get_metadata(&self) -> MutationMetadata {
        MutationMetadata {
            operator_name: "AddOperation".to_string(),
            mutation_type: MutationType::AddOperation,
            difficulty: MutationDifficulty::Easy,
            expected_impact: ImpactLevel::Medium,
        }
    }
    
    fn estimate_impact(&self, _architecture: &ArchitectureGenotype<T>) -> MutationImpact<T> {
        MutationImpact {
            performance_delta: T::from(0.05).unwrap(), // Small positive impact
            complexity_delta: T::from(0.1).unwrap(),   // Increased complexity
            resource_delta: {
                let mut resources = HashMap::new();
                resources.insert("memory".to_string(), T::from(0.1).unwrap());
                resources.insert("flops".to_string(), T::from(0.15).unwrap());
                resources
            },
            confidence: T::from(0.7).unwrap(),
            risk_level: RiskLevel::Low,
        }
    }
}

impl<T: Float + Default + Clone> MutationOperator<T> for RemoveOperationMutator<T> {
    fn mutate(&self, architecture: &ArchitectureGenotype<T>) -> Result<ArchitectureGenotype<T>> {
        use rand::Rng;
        let mut rng = rand::rng();
        
        if architecture.operations.len() <= self.min_operations {
            return Err(OptimError::InvalidInput(
                "Cannot remove operation: minimum operations reached".to_string()
            ));
        }
        
        let mut mutated = architecture.clone();
        mutated.id = format!("{}_{}", architecture.id, "remove_op");
        
        // Select operation to remove
        let removal_idx = match self.removal_strategy {
            RemovalStrategy::Random => {
                rng.gen_range(0..mutated.operations.len())
            }
            _ => {
                // Fallback to random
                rng.gen_range(0..mutated.operations.len())
            }
        };
        
        // Check if operation is critical
        let op_to_remove = &mutated.operations[removal_idx];
        if self.preserve_critical && self.critical_operations.contains(&op_to_remove.id) {
            return Err(OptimError::InvalidInput(
                "Cannot remove critical operation".to_string()
            ));
        }
        
        // Remove operation
        mutated.operations.remove(removal_idx);
        
        // Remove associated connections
        mutated.connections.retain(|conn| {
            conn.source != op_to_remove.id && conn.target != op_to_remove.id
        });
        
        Ok(mutated)
    }
    
    fn is_applicable(&self, architecture: &ArchitectureGenotype<T>) -> bool {
        architecture.operations.len() > self.min_operations
    }
    
    fn get_metadata(&self) -> MutationMetadata {
        MutationMetadata {
            operator_name: "RemoveOperation".to_string(),
            mutation_type: MutationType::RemoveOperation,
            difficulty: MutationDifficulty::Medium,
            expected_impact: ImpactLevel::High,
        }
    }
    
    fn estimate_impact(&self, _architecture: &ArchitectureGenotype<T>) -> MutationImpact<T> {
        MutationImpact {
            performance_delta: T::from(-0.1).unwrap(), // Potential negative impact
            complexity_delta: T::from(-0.2).unwrap(),  // Reduced complexity
            resource_delta: {
                let mut resources = HashMap::new();
                resources.insert("memory".to_string(), T::from(-0.1).unwrap());
                resources.insert("flops".to_string(), T::from(-0.15).unwrap());
                resources
            },
            confidence: T::from(0.6).unwrap(),
            risk_level: RiskLevel::Medium,
        }
    }
}

impl<T: Float + Default + Clone> MutationOperator<T> for ParameterMutator<T> {
    fn mutate(&self, architecture: &ArchitectureGenotype<T>) -> Result<ArchitectureGenotype<T>> {
        use rand::Rng;
        let mut rng = rand::rng();
        
        let mut mutated = architecture.clone();
        mutated.id = format!("{}_{}", architecture.id, "param_mut");
        
        // Select operation to mutate
        if mutated.operations.is_empty() {
            return Ok(mutated);
        }
        
        let op_idx = rng.gen_range(0..mutated.operations.len());
        let operation = &mut mutated.operations[op_idx];
        
        // Select parameter to mutate
        let param_keys: Vec<String> = operation.parameters.keys().cloned().collect();
        if param_keys.is_empty() {
            return Ok(mutated);
        }
        
        let param_key = &param_keys[rng.gen_range(0..param_keys.len())];
        
        // Apply mutation based on distribution
        if let Some(current_value) = operation.parameters.get(param_key) {
            let new_value = match &self.distribution {
                ParameterDistribution::Gaussian { std } => {
                    let noise = T::from(rng.gen::<f64>() - 0.5).unwrap() * *std * T::from(2.0).unwrap();
                    *current_value + noise
                }
                ParameterDistribution::Uniform { range } => {
                    let noise = T::from(rng.gen::<f64>() - 0.5).unwrap() * *range * T::from(2.0).unwrap();
                    *current_value + noise
                }
                _ => *current_value, // Fallback
            };
            
            operation.parameters.insert(param_key.clone(), new_value);
            operation.metadata.mutation_count += 1;
            operation.metadata.last_mutated = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
        }
        
        Ok(mutated)
    }
    
    fn is_applicable(&self, architecture: &ArchitectureGenotype<T>) -> bool {
        !architecture.operations.is_empty() &&
        architecture.operations.iter().any(|op| !op.parameters.is_empty())
    }
    
    fn get_metadata(&self) -> MutationMetadata {
        MutationMetadata {
            operator_name: "ParameterMutation".to_string(),
            mutation_type: MutationType::ModifyParameters,
            difficulty: MutationDifficulty::Easy,
            expected_impact: ImpactLevel::Low,
        }
    }
    
    fn estimate_impact(&self, _architecture: &ArchitectureGenotype<T>) -> MutationImpact<T> {
        MutationImpact {
            performance_delta: T::zero(), // Neutral expected impact
            complexity_delta: T::zero(),  // No complexity change
            resource_delta: HashMap::new(),
            confidence: T::from(0.8).unwrap(),
            risk_level: RiskLevel::VeryLow,
        }
    }
}

// Helper implementations

impl<T: Float + Default + Clone> OperationPool<T> {
    fn new() -> Result<Self> {
        Ok(Self {
            operation_types: vec![
                OperationType {
                    id: "conv2d".to_string(),
                    name: "2D Convolution".to_string(),
                    category: OperationCategory::Convolution,
                    parameter_ranges: HashMap::new(),
                    cost_estimate: CostEstimate {
                        param_multiplier: 1.0,
                        flops_multiplier: 2.0,
                        memory_multiplier: 1.5,
                        latency_multiplier: 1.0,
                    },
                },
                OperationType {
                    id: "dense".to_string(),
                    name: "Dense Layer".to_string(),
                    category: OperationCategory::Dense,
                    parameter_ranges: HashMap::new(),
                    cost_estimate: CostEstimate {
                        param_multiplier: 1.5,
                        flops_multiplier: 1.0,
                        memory_multiplier: 1.0,
                        latency_multiplier: 0.8,
                    },
                },
            ],
            operation_templates: HashMap::new(),
            compatibility_matrix: Array2::eye(2),
            operation_frequencies: HashMap::new(),
        })
    }
}

impl<T: Float + Default + Clone> MutationHistory<T> {
    fn new() -> Self {
        Self {
            mutation_log: Vec::new(),
            operator_stats: HashMap::new(),
            rollback_info: HashMap::new(),
            performance_tracking: Vec::new(),
        }
    }
}

impl<T: Float + Default + Clone> PerformanceFeedback<T> {
    fn new() -> Self {
        Self {
            recent_changes: VecDeque::new(),
            moving_average: T::zero(),
            trend: PerformanceTrend::Stable,
            stagnation_counter: 0,
            best_performance: T::from(f64::NEG_INFINITY).unwrap(),
        }
    }
}

impl<T: Float + Default + Clone> Default for MutationConfig<T> {
    fn default() -> Self {
        let mut operator_weights = HashMap::new();
        operator_weights.insert(MutationType::AddOperation, T::from(0.3).unwrap());
        operator_weights.insert(MutationType::RemoveOperation, T::from(0.2).unwrap());
        operator_weights.insert(MutationType::ReplaceOperation, T::from(0.25).unwrap());
        operator_weights.insert(MutationType::ModifyParameters, T::from(0.25).unwrap());
        
        Self {
            mutation_probability: T::from(0.1).unwrap(),
            max_mutations_per_arch: 3,
            mutation_strength: T::from(0.5).unwrap(),
            available_operators: vec![
                MutationType::AddOperation,
                MutationType::RemoveOperation,
                MutationType::ReplaceOperation,
                MutationType::ModifyParameters,
            ],
            operator_weights,
            adaptive_params: Some(AdaptiveMutationParams {
                adaptation_rate: T::from(0.1).unwrap(),
                success_threshold: T::from(0.6).unwrap(),
                failure_penalty: T::from(0.1).unwrap(),
                window_size: 10,
                min_mutation_rate: T::from(0.01).unwrap(),
                max_mutation_rate: T::from(0.5).unwrap(),
            }),
            preserve_constraints: true,
            track_history: true,
        }
    }
}