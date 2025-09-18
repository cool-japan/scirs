//! Architecture space definitions for neural optimizer search
//!
//! Defines the search space of possible optimizer architectures,
//! including components, connections, and constraints.

use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete optimizer architecture representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerArchitecture<T: Float> {
    /// List of optimizer components
    pub components: Vec<OptimizerComponent<T>>,

    /// Connections between components
    pub connections: Vec<ComponentConnection>,

    /// Architecture metadata
    pub metadata: HashMap<String, String>,
}

/// Individual optimizer component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerComponent<T: Float> {
    /// Type of the component
    pub component_type: ComponentType,

    /// Hyperparameters for this component
    pub hyperparameters: HashMap<String, T>,

    /// Input/output connections
    pub connections: Vec<ComponentConnection>,
}

/// Types of optimizer components
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentType {
    // Basic optimizers
    SGD,
    Adam,
    AdaGrad,
    RMSprop,
    AdamW,

    // Advanced optimizers
    LAMB,
    LARS,
    Lion,
    RAdam,
    Lookahead,
    SAM,
    LBFGS,
    SparseAdam,
    GroupedAdam,

    // Meta-learning components
    MAML,
    Reptile,
    MetaSGD,

    // Learning rate schedulers
    ConstantLR,
    ExponentialLR,
    StepLR,
    CosineAnnealingLR,
    OneCycleLR,
    CyclicLR,

    // Regularization components
    L1Regularizer,
    L2Regularizer,
    ElasticNetRegularizer,
    DropoutRegularizer,
    GradientClipping,
    WeightDecay,

    // Adaptive components
    AdaptiveLR,
    AdaptiveMomentum,
    AdaptiveRegularization,

    // Neural components
    LSTMOptimizer,
    TransformerOptimizer,
    AttentionOptimizer,

    // Custom components
    Custom(String),
}

/// Connection between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentConnection {
    /// Source component index
    pub from: usize,

    /// Target component index
    pub to: usize,

    /// Connection type
    pub connection_type: ConnectionType,

    /// Connection weight (if applicable)
    pub weight: f64,

    /// Connection metadata
    pub metadata: HashMap<String, String>,
}

/// Types of connections between components
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Sequential connection (output of one feeds into next)
    Sequential,

    /// Parallel connection (components process same input)
    Parallel,

    /// Skip connection (bypassing intermediate components)
    Skip,

    /// Residual connection (adding input to output)
    Residual,

    /// Attention connection (weighted combination)
    Attention,

    /// Gating connection (conditional processing)
    Gating,

    /// Feedback connection (output feeds back as input)
    Feedback,

    /// Custom connection
    Custom(String),
}

/// Connection pattern for entire architecture
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionPattern {
    /// Simple sequential chain
    Sequential,

    /// Parallel branches that merge
    Parallel,

    /// Skip connections throughout
    Skip,

    /// Dense connections (all-to-all)
    Dense,

    /// Hierarchical structure
    Hierarchical,

    /// Graph-based structure
    Graph,

    /// Custom pattern
    Custom,
}

/// Search space configuration for architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Available component types
    pub component_types: Vec<ComponentType>,

    /// Hyperparameter ranges for each component type
    pub hyperparameter_ranges: HashMap<ComponentType, ComponentHyperparameters>,

    /// Available connection patterns
    pub connection_patterns: Vec<ConnectionPattern>,

    /// Architecture constraints
    pub constraints: ArchitectureConstraints,

    /// Search space metadata
    pub metadata: SearchSpaceMetadata,
}

/// Hyperparameter configuration for a component type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHyperparameters {
    /// Continuous parameters (min, max)
    pub continuous: HashMap<String, (f64, f64)>,

    /// Integer parameters (min, max)
    pub integer: HashMap<String, (i32, i32)>,

    /// Categorical parameters
    pub categorical: HashMap<String, Vec<String>>,

    /// Boolean parameters
    pub boolean: Vec<String>,

    /// Log-uniform parameters (min, max)
    pub log_uniform: HashMap<String, (f64, f64)>,

    /// Dependent parameters (parameter -> condition)
    pub dependencies: HashMap<String, ParameterDependency>,
}

/// Parameter dependency specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDependency {
    /// Parameter this depends on
    pub depends_on: String,

    /// Condition for dependency
    pub condition: DependencyCondition,

    /// Valid values when condition is met
    pub valid_values: ParameterValues,
}

/// Dependency conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DependencyCondition {
    /// Equals specific value
    Equals(String),

    /// Greater than threshold
    GreaterThan(f64),

    /// Less than threshold
    LessThan(f64),

    /// In specific range
    InRange(f64, f64),

    /// In set of values
    InSet(Vec<String>),
}

/// Parameter values for dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValues {
    /// Continuous range
    Continuous(f64, f64),

    /// Integer range
    Integer(i32, i32),

    /// Categorical options
    Categorical(Vec<String>),

    /// Boolean value
    Boolean(bool),
}

/// Architecture constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureConstraints {
    /// Maximum number of components
    pub max_components: usize,

    /// Minimum number of components
    pub min_components: usize,

    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,

    /// Maximum computation time per step (ms)
    pub max_computation_time_ms: f64,

    /// Maximum model parameters
    pub max_parameters: usize,

    /// Compatibility constraints
    pub compatibility: Vec<CompatibilityConstraint>,

    /// Performance constraints
    pub performance: PerformanceConstraints,
}

/// Compatibility constraint between components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityConstraint {
    /// First component type
    pub component1: ComponentType,

    /// Second component type
    pub component2: ComponentType,

    /// Compatibility type
    pub compatibility_type: CompatibilityType,

    /// Additional conditions
    pub conditions: Vec<String>,
}

/// Types of compatibility between components
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityType {
    /// Components are fully compatible
    Compatible,

    /// Components are incompatible
    Incompatible,

    /// Components are conditionally compatible
    Conditional,

    /// Components have synergistic effects
    Synergistic,

    /// Components have conflicting effects
    Conflicting,
}

/// Performance constraints for architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConstraints {
    /// Minimum convergence speed
    pub min_convergence_speed: f64,

    /// Maximum training time (hours)
    pub max_training_time_hours: f64,

    /// Minimum final performance
    pub min_final_performance: f64,

    /// Maximum performance variance
    pub max_performance_variance: f64,

    /// Robustness requirements
    pub robustness_requirements: RobustnessRequirements,
}

/// Robustness requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessRequirements {
    /// Hyperparameter sensitivity threshold
    pub hyperparameter_sensitivity: f64,

    /// Noise tolerance level
    pub noise_tolerance: f64,

    /// Distribution shift robustness
    pub distribution_shift_robustness: f64,

    /// Initialization sensitivity
    pub initialization_sensitivity: f64,
}

/// Search space metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpaceMetadata {
    /// Search space name
    pub name: String,

    /// Version
    pub version: String,

    /// Description
    pub description: String,

    /// Target domains
    pub target_domains: Vec<OptimizationDomain>,

    /// Complexity level
    pub complexity_level: ComplexityLevel,

    /// Creation timestamp
    pub created_at: String,

    /// Author information
    pub author: String,
}

/// Optimization domains
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationDomain {
    ComputerVision,
    NaturalLanguageProcessing,
    ReinforcementLearning,
    RecommendationSystems,
    TimeSeriesForecasting,
    GraphAnalytics,
    ScientificComputing,
    FinancialModeling,
    RoboticsControl,
    GeneralMachineLearning,
}

/// Complexity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
    ExpertLevel,
}

/// Architecture validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether architecture is valid
    pub is_valid: bool,

    /// Validation errors
    pub errors: Vec<ValidationError>,

    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,

    /// Estimated resource usage
    pub resource_estimate: ResourceEstimate,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Error type
    pub error_type: ValidationErrorType,

    /// Error message
    pub message: String,

    /// Component index (if applicable)
    pub component_index: Option<usize>,

    /// Parameter name (if applicable)
    pub parameter_name: Option<String>,
}

/// Types of validation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationErrorType {
    InvalidParameter,
    IncompatibleComponents,
    ConstraintViolation,
    InvalidConnection,
    ResourceExceeded,
    DependencyViolation,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Warning type
    pub warning_type: ValidationWarningType,

    /// Warning message
    pub message: String,

    /// Severity level
    pub severity: WarningSeverity,
}

/// Types of validation warnings
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationWarningType {
    SuboptimalConfiguration,
    PerformanceImpact,
    ResourceInefficiency,
    CompatibilityIssue,
    ExperimentalFeature,
}

/// Warning severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

/// Resource usage estimate
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Estimated memory usage (MB)
    pub memory_mb: f64,

    /// Estimated computation time per step (ms)
    pub computation_time_ms: f64,

    /// Estimated model parameters
    pub model_parameters: usize,

    /// Estimated training time (hours)
    pub training_time_hours: f64,

    /// Estimated energy consumption (kWh)
    pub energy_consumption_kwh: f64,
}

/// Architecture factory for creating predefined architectures
pub struct ArchitectureFactory;

impl ArchitectureFactory {
    /// Create a simple SGD architecture
    pub fn create_sgd_architecture<T: Float>() -> OptimizerArchitecture<T> {
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), num_traits::cast::cast(0.01).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("momentum".to_string(), num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("weight_decay".to_string(), num_traits::cast::cast(0.0001).unwrap_or_else(|| T::zero()));

        OptimizerArchitecture {
            components: vec![OptimizerComponent {
                component_type: ComponentType::SGD,
                hyperparameters,
                connections: Vec::new(),
            }],
            connections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create an Adam architecture
    pub fn create_adam_architecture<T: Float>() -> OptimizerArchitecture<T> {
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("beta1".to_string(), num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("beta2".to_string(), num_traits::cast::cast(0.999).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("epsilon".to_string(), num_traits::cast::cast(1e-7).unwrap_or_else(|| T::zero())); // Within (1e-10, 1e-6) range
        hyperparameters.insert("weight_decay".to_string(), num_traits::cast::cast(1e-5).unwrap_or_else(|| T::zero())); // Within (1e-8, 1e-2) range

        OptimizerArchitecture {
            components: vec![OptimizerComponent {
                component_type: ComponentType::Adam,
                hyperparameters,
                connections: Vec::new(),
            }],
            connections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a complex hybrid architecture
    pub fn create_hybrid_architecture<T: Float>() -> OptimizerArchitecture<T> {
        let mut components = Vec::new();

        // Adam optimizer component
        let mut adam_params = HashMap::new();
        adam_params.insert("learning_rate".to_string(), num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()));
        adam_params.insert("beta1".to_string(), num_traits::cast::cast(0.9).unwrap_or_else(|| T::zero()));
        adam_params.insert("beta2".to_string(), num_traits::cast::cast(0.999).unwrap_or_else(|| T::zero()));

        components.push(OptimizerComponent {
            component_type: ComponentType::Adam,
            hyperparameters: adam_params,
            connections: Vec::new(),
        });

        // Cosine annealing scheduler
        let mut scheduler_params = HashMap::new();
        scheduler_params.insert("t_max".to_string(), num_traits::cast::cast(100.0).unwrap_or_else(|| T::zero()));
        scheduler_params.insert("eta_min".to_string(), num_traits::cast::cast(1e-6).unwrap_or_else(|| T::zero()));

        components.push(OptimizerComponent {
            component_type: ComponentType::CosineAnnealingLR,
            hyperparameters: scheduler_params,
            connections: Vec::new(),
        });

        // L2 regularization
        let mut reg_params = HashMap::new();
        reg_params.insert("lambda".to_string(), num_traits::cast::cast(0.0001).unwrap_or_else(|| T::zero()));

        components.push(OptimizerComponent {
            component_type: ComponentType::L2Regularizer,
            hyperparameters: reg_params,
            connections: Vec::new(),
        });

        // Sequential connections
        let connections = vec![
            ComponentConnection {
                from: 0,
                to: 1,
                connection_type: ConnectionType::Sequential,
                weight: 1.0,
                metadata: HashMap::new(),
            },
            ComponentConnection {
                from: 1,
                to: 2,
                connection_type: ConnectionType::Sequential,
                weight: 1.0,
                metadata: HashMap::new(),
            },
        ];

        OptimizerArchitecture {
            components,
            connections,
            metadata: HashMap::new(),
        }
    }

    /// Create an LSTM-based neural optimizer architecture
    pub fn create_lstm_optimizer_architecture<T: Float>() -> OptimizerArchitecture<T> {
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("hidden_size".to_string(), num_traits::cast::cast(256.0).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("num_layers".to_string(), num_traits::cast::cast(2.0).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("dropout".to_string(), num_traits::cast::cast(0.1).unwrap_or_else(|| T::zero()));
        hyperparameters.insert("meta_learning_rate".to_string(), num_traits::cast::cast(0.001).unwrap_or_else(|| T::zero()));

        OptimizerArchitecture {
            components: vec![OptimizerComponent {
                component_type: ComponentType::LSTMOptimizer,
                hyperparameters,
                connections: Vec::new(),
            }],
            connections: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Architecture validator for checking validity and constraints
pub struct ArchitectureValidator {
    search_space: SearchSpace,
}

impl ArchitectureValidator {
    /// Create new validator with search space
    pub fn new(_searchspace: SearchSpace) -> Self {
        Self {
            search_space: _searchspace,
        }
    }

    /// Validate an architecture
    pub fn validate<T: Float>(&self, architecture: &OptimizerArchitecture<T>) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check component count constraints
        if architecture.components.len() > self.search_space.constraints.max_components {
            errors.push(ValidationError {
                error_type: ValidationErrorType::ConstraintViolation,
                message: format!(
                    "Too many components: {} > {}",
                    architecture.components.len(),
                    self.search_space.constraints.max_components
                ),
                component_index: None,
                parameter_name: None,
            });
        }

        if architecture.components.len() < self.search_space.constraints.min_components {
            errors.push(ValidationError {
                error_type: ValidationErrorType::ConstraintViolation,
                message: format!(
                    "Too few components: {} < {}",
                    architecture.components.len(),
                    self.search_space.constraints.min_components
                ),
                component_index: None,
                parameter_name: None,
            });
        }

        // Validate each component
        for (i, component) in architecture.components.iter().enumerate() {
            self.validate_component(component, i, &mut errors, &mut warnings);
        }

        // Validate connections
        self.validate_connections(architecture, &mut errors, &mut warnings);

        // Check compatibility constraints
        self.validate_compatibility(architecture, &mut errors, &mut warnings);

        // Estimate resources
        let resource_estimate = self.estimate_resources(architecture);

        // Check resource constraints
        if resource_estimate.memory_mb > self.search_space.constraints.max_memory_mb as f64 {
            errors.push(ValidationError {
                error_type: ValidationErrorType::ResourceExceeded,
                message: format!(
                    "Memory usage exceeds limit: {:.2} MB > {} MB",
                    resource_estimate.memory_mb, self.search_space.constraints.max_memory_mb
                ),
                component_index: None,
                parameter_name: None,
            });
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
            resource_estimate,
        }
    }

    fn validate_component<T: Float>(
        &self,
        component: &OptimizerComponent<T>,
        index: usize,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        // Check if component type is in search space
        if !self
            .search_space
            .component_types
            .contains(&component.component_type)
        {
            errors.push(ValidationError {
                error_type: ValidationErrorType::InvalidParameter,
                message: format!(
                    "Component type {:?} not in search space",
                    component.component_type
                ),
                component_index: Some(index),
                parameter_name: None,
            });
            return;
        }

        // Get hyperparameter specification for this component type
        if let Some(param_spec) = self
            .search_space
            .hyperparameter_ranges
            .get(&component.component_type)
        {
            self.validate_hyperparameters(component, param_spec, index, errors, warnings);
        }
    }

    fn validate_hyperparameters<T: Float>(
        &self,
        component: &OptimizerComponent<T>,
        param_spec: &ComponentHyperparameters,
        component_index: usize,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationWarning>,
    ) {
        for (param_name, param_value) in &component.hyperparameters {
            let value = param_value.to_f64().unwrap_or(0.0);

            // Check continuous parameters
            if let Some((min, max)) = param_spec.continuous.get(param_name) {
                if value < *min || value > *max {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::InvalidParameter,
                        message: format!(
                            "Parameter {} value {} out of range [{}, {}]",
                            param_name, value, min, max
                        ),
                        component_index: Some(component_index),
                        parameter_name: Some(param_name.clone()),
                    });
                }
            }

            // Check log-uniform parameters
            if let Some((min, max)) = param_spec.log_uniform.get(param_name) {
                if value <= 0.0 || value < *min || value > *max {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::InvalidParameter,
                        message: format!(
                            "Log-uniform parameter {} value {} out of range [{}, {}]",
                            param_name, value, min, max
                        ),
                        component_index: Some(component_index),
                        parameter_name: Some(param_name.clone()),
                    });
                }
            }

            // Check integer parameters
            if let Some((min, max)) = param_spec.integer.get(param_name) {
                let int_value = value as i32;
                if int_value < *min || int_value > *max {
                    errors.push(ValidationError {
                        error_type: ValidationErrorType::InvalidParameter,
                        message: format!(
                            "Integer parameter {} value {} out of range [{}, {}]",
                            param_name, int_value, min, max
                        ),
                        component_index: Some(component_index),
                        parameter_name: Some(param_name.clone()),
                    });
                }
            }
        }
    }

    fn validate_connections<T: Float>(
        &self,
        architecture: &OptimizerArchitecture<T>,
        errors: &mut Vec<ValidationError>,
        _warnings: &mut Vec<ValidationWarning>,
    ) {
        for connection in &architecture.connections {
            if connection.from >= architecture.components.len() {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidConnection,
                    message: format!("Connection source index {} out of bounds", connection.from),
                    component_index: None,
                    parameter_name: None,
                });
            }

            if connection.to >= architecture.components.len() {
                errors.push(ValidationError {
                    error_type: ValidationErrorType::InvalidConnection,
                    message: format!("Connection target index {} out of bounds", connection.to),
                    component_index: None,
                    parameter_name: None,
                });
            }
        }
    }

    fn validate_compatibility<T: Float>(
        &self,
        architecture: &OptimizerArchitecture<T>,
        errors: &mut Vec<ValidationError>,
        warnings: &mut Vec<ValidationWarning>,
    ) {
        for constraint in &self.search_space.constraints.compatibility {
            let component1_exists = architecture
                .components
                .iter()
                .any(|c| c.component_type == constraint.component1);
            let component2_exists = architecture
                .components
                .iter()
                .any(|c| c.component_type == constraint.component2);

            if component1_exists && component2_exists {
                match constraint.compatibility_type {
                    CompatibilityType::Incompatible => {
                        errors.push(ValidationError {
                            error_type: ValidationErrorType::IncompatibleComponents,
                            message: format!(
                                "Incompatible components: {:?} and {:?}",
                                constraint.component1, constraint.component2
                            ),
                            component_index: None,
                            parameter_name: None,
                        });
                    }
                    CompatibilityType::Conflicting => {
                        warnings.push(ValidationWarning {
                            warning_type: ValidationWarningType::CompatibilityIssue,
                            message: format!(
                                "Conflicting components: {:?} and {:?}",
                                constraint.component1, constraint.component2
                            ),
                            severity: WarningSeverity::High,
                        });
                    }
                    _ => {} // Other types don't generate errors/warnings
                }
            }
        }
    }

    fn estimate_resources<T: Float>(
        &self,
        architecture: &OptimizerArchitecture<T>,
    ) -> ResourceEstimate {
        let mut memory_mb = 0.0;
        let mut computation_time_ms = 0.0;
        let mut model_parameters = 0;

        for component in &architecture.components {
            match component.component_type {
                ComponentType::LSTMOptimizer => {
                    let hidden_size = component
                        .hyperparameters
                        .get("hidden_size")
                        .map(|v| v.to_f64().unwrap_or(256.0))
                        .unwrap_or(256.0);
                    let num_layers = component
                        .hyperparameters
                        .get("num_layers")
                        .map(|v| v.to_f64().unwrap_or(2.0))
                        .unwrap_or(2.0);

                    memory_mb += hidden_size * num_layers * 4.0 / 1024.0 / 1024.0; // Rough estimate
                    computation_time_ms += hidden_size * num_layers * 0.01;
                    model_parameters += (hidden_size * hidden_size * 4.0 * num_layers) as usize;
                }
                ComponentType::TransformerOptimizer => {
                    memory_mb += 512.0; // Base transformer memory
                    computation_time_ms += 10.0;
                    model_parameters += 1_000_000; // Rough estimate
                }
                _ => {
                    memory_mb += 10.0; // Base component memory
                    computation_time_ms += 1.0;
                    model_parameters += 1000;
                }
            }
        }

        ResourceEstimate {
            memory_mb,
            computation_time_ms,
            model_parameters,
            training_time_hours: computation_time_ms / 3600.0 / 1000.0 * 1000.0, // Rough estimate
            energy_consumption_kwh: computation_time_ms / 3600.0 / 1000.0 * 0.1, // Very rough estimate
        }
    }
}

/// Default search space for common optimization scenarios
impl Default for SearchSpace {
    fn default() -> Self {
        let component_types = vec![
            ComponentType::SGD,
            ComponentType::Adam,
            ComponentType::AdaGrad,
            ComponentType::RMSprop,
            ComponentType::AdamW,
            ComponentType::LAMB,
            ComponentType::LARS,
            ComponentType::Lion,
            ComponentType::ConstantLR,
            ComponentType::ExponentialLR,
            ComponentType::CosineAnnealingLR,
            ComponentType::L1Regularizer,
            ComponentType::L2Regularizer,
            ComponentType::GradientClipping,
        ];

        let mut hyperparameter_ranges = HashMap::new();

        // SGD hyperparameters
        let mut sgd_params = ComponentHyperparameters {
            continuous: HashMap::new(),
            integer: HashMap::new(),
            categorical: HashMap::new(),
            boolean: Vec::new(),
            log_uniform: HashMap::new(),
            dependencies: HashMap::new(),
        };
        sgd_params
            .log_uniform
            .insert("learning_rate".to_string(), (1e-6, 1e-1));
        sgd_params
            .continuous
            .insert("momentum".to_string(), (0.0, 0.99));
        sgd_params
            .log_uniform
            .insert("weight_decay".to_string(), (1e-8, 1e-2));
        hyperparameter_ranges.insert(ComponentType::SGD, sgd_params);

        // Adam hyperparameters
        let mut adam_params = ComponentHyperparameters {
            continuous: HashMap::new(),
            integer: HashMap::new(),
            categorical: HashMap::new(),
            boolean: Vec::new(),
            log_uniform: HashMap::new(),
            dependencies: HashMap::new(),
        };
        adam_params
            .log_uniform
            .insert("learning_rate".to_string(), (1e-6, 1e-1));
        adam_params
            .continuous
            .insert("beta1".to_string(), (0.8, 0.999));
        adam_params
            .continuous
            .insert("beta2".to_string(), (0.9, 0.9999));
        adam_params
            .log_uniform
            .insert("epsilon".to_string(), (1e-10, 1e-6));
        adam_params
            .log_uniform
            .insert("weight_decay".to_string(), (1e-8, 1e-2));
        hyperparameter_ranges.insert(ComponentType::Adam, adam_params);

        Self {
            component_types,
            hyperparameter_ranges,
            connection_patterns: vec![
                ConnectionPattern::Sequential,
                ConnectionPattern::Parallel,
                ConnectionPattern::Skip,
            ],
            constraints: ArchitectureConstraints {
                max_components: 10,
                min_components: 1,
                max_memory_mb: 8192,
                max_computation_time_ms: 1000.0,
                max_parameters: 10_000_000,
                compatibility: Vec::new(),
                performance: PerformanceConstraints {
                    min_convergence_speed: 0.01,
                    max_training_time_hours: 24.0,
                    min_final_performance: 0.5,
                    max_performance_variance: 0.1,
                    robustness_requirements: RobustnessRequirements {
                        hyperparameter_sensitivity: 0.1,
                        noise_tolerance: 0.05,
                        distribution_shift_robustness: 0.1,
                        initialization_sensitivity: 0.05,
                    },
                },
            },
            metadata: SearchSpaceMetadata {
                name: "Default Optimizer Search Space".to_string(),
                version: "1.0.0".to_string(),
                description: "Standard search space for optimizer architectures".to_string(),
                target_domains: vec![OptimizationDomain::GeneralMachineLearning],
                complexity_level: ComplexityLevel::Moderate,
                created_at: "2024-01-01T00:00:00Z".to_string(),
                author: "SciRS2 Team".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_architecture_creation() {
        let arch = ArchitectureFactory::create_sgd_architecture::<f64>();
        assert_eq!(arch.components.len(), 1);
        assert_eq!(arch.components[0].component_type, ComponentType::SGD);
        assert!(arch.components[0]
            .hyperparameters
            .contains_key("learning_rate"));
    }

    #[test]
    fn test_adam_architecture_creation() {
        let arch = ArchitectureFactory::create_adam_architecture::<f64>();
        assert_eq!(arch.components.len(), 1);
        assert_eq!(arch.components[0].component_type, ComponentType::Adam);
        assert!(arch.components[0].hyperparameters.contains_key("beta1"));
        assert!(arch.components[0].hyperparameters.contains_key("beta2"));
    }

    #[test]
    fn test_hybrid_architecture_creation() {
        let arch = ArchitectureFactory::create_hybrid_architecture::<f64>();
        assert_eq!(arch.components.len(), 3);
        assert_eq!(arch.connections.len(), 2);
    }

    #[test]
    fn test_architecture_validation() {
        let search_space = SearchSpace::default();
        let validator = ArchitectureValidator::new(search_space);
        let arch = ArchitectureFactory::create_adam_architecture::<f64>();

        let result = validator.validate(&arch);
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_resource_estimation() {
        let search_space = SearchSpace::default();
        let validator = ArchitectureValidator::new(search_space);
        let arch = ArchitectureFactory::create_lstm_optimizer_architecture::<f64>();

        let result = validator.validate(&arch);
        assert!(result.resource_estimate.memory_mb > 0.0);
        assert!(result.resource_estimate.model_parameters > 0);
    }
}
