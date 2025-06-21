//! Neural network integration for optimizers
//!
//! This module provides interfaces and utilities for integrating optimizers with neural networks,
//! including generic parameter optimization, lazy registration, and architecture-aware optimizations.

use crate::error::{OptimError, Result};
use ndarray::{Array, Dimension, ScalarOperand};
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

/// Type alias for layer identifiers
pub type LayerId = String;

/// Type alias for parameter identifiers
pub type ParamId = String;

/// Parameter metadata for neural network parameters
#[derive(Debug, Clone)]
pub struct ParameterMetadata {
    /// Layer name this parameter belongs to
    pub layer_name: LayerId,
    /// Parameter name within the layer
    pub param_name: ParamId,
    /// Parameter shape
    pub shape: Vec<usize>,
    /// Whether parameter requires gradients
    pub requires_grad: bool,
    /// Parameter type (weights, bias, etc.)
    pub param_type: ParameterType,
    /// Sharing group for parameter sharing
    pub sharing_group: Option<String>,
    /// Custom tags for architecture-specific optimizations
    pub tags: Vec<String>,
}

/// Types of neural network parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParameterType {
    /// Weight matrices
    Weight,
    /// Bias vectors
    Bias,
    /// Normalization parameters (scale/shift)
    Normalization,
    /// Embedding parameters
    Embedding,
    /// Attention parameters
    Attention,
    /// Custom parameter type
    Custom,
}

/// Layer architecture information
#[derive(Debug, Clone)]
pub struct LayerArchitecture {
    /// Layer type name
    pub layer_type: String,
    /// Input dimensions
    pub input_dims: Vec<usize>,
    /// Output dimensions
    pub output_dims: Vec<usize>,
    /// Layer-specific configuration
    pub config: HashMap<String, LayerConfig>,
    /// Whether layer is trainable
    pub trainable: bool,
}

/// Layer configuration values
#[derive(Debug, Clone)]
pub enum LayerConfig {
    /// Integer value
    Int(i64),
    /// Float value
    Float(f64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// List of values
    List(Vec<LayerConfig>),
}

/// Generic parameter optimization interface
pub trait ParameterOptimizer<A: Float, D: Dimension> {
    /// Register a parameter for optimization
    fn register_parameter(
        &mut self,
        param_id: ParamId,
        parameter: &Array<A, D>,
        metadata: ParameterMetadata,
    ) -> Result<()>;

    /// Update registered parameters with gradients
    fn step(
        &mut self,
        gradients: HashMap<ParamId, Array<A, D>>,
        parameters: &mut HashMap<ParamId, Array<A, D>>,
    ) -> Result<()>;

    /// Get parameter-specific learning rate
    fn get_learning_rate(&self, param_id: &ParamId) -> Option<A>;

    /// Set parameter-specific learning rate
    fn set_learning_rate(&mut self, param_id: &ParamId, lr: A) -> Result<()>;

    /// Get optimizer state for a parameter
    fn get_parameter_state(&self, param_id: &ParamId) -> Option<&HashMap<String, Array<A, D>>>;

    /// Reset optimizer state
    fn reset_state(&mut self);

    /// Get all registered parameter IDs
    fn registered_parameters(&self) -> Vec<ParamId>;
}

/// Neural network parameter manager with lazy registration
#[derive(Debug)]
pub struct ParameterManager<A: Float, D: Dimension> {
    /// Registered parameters with metadata
    parameters: HashMap<ParamId, ParameterMetadata>,
    /// Parameter optimizer states
    optimizer_states: HashMap<ParamId, HashMap<String, Array<A, D>>>,
    /// Layer architectures
    layer_architectures: HashMap<LayerId, LayerArchitecture>,
    /// Parameter sharing groups
    sharing_groups: HashMap<String, Vec<ParamId>>,
    /// Layer-specific optimization rules
    layer_rules: HashMap<LayerId, LayerOptimizationRule<A>>,
    /// Global optimization configuration
    global_config: OptimizationConfig<A>,
    /// Lazy registration mode
    lazy_mode: bool,
    /// Pending registrations (for lazy mode)
    pending_registrations: Vec<(ParamId, ParameterMetadata)>,
}

/// Layer-specific optimization rules
#[derive(Debug, Clone)]
pub struct LayerOptimizationRule<A: Float> {
    /// Learning rate multiplier for this layer
    pub lr_multiplier: A,
    /// Weight decay multiplier
    pub weight_decay_multiplier: A,
    /// Whether to freeze this layer
    pub frozen: bool,
    /// Custom optimizer settings
    pub custom_settings: HashMap<String, LayerConfig>,
}

/// Global optimization configuration
#[derive(Debug, Clone)]
pub struct OptimizationConfig<A: Float> {
    /// Base learning rate
    pub base_learning_rate: A,
    /// Global weight decay
    pub weight_decay: A,
    /// Gradient clipping threshold
    pub gradient_clip: Option<A>,
    /// Whether to use mixed precision
    pub mixed_precision: bool,
    /// Architecture-specific optimizations
    pub architecture_optimizations: HashMap<String, bool>,
}

impl<A: Float + ScalarOperand + Debug, D: Dimension> ParameterManager<A, D> {
    /// Create a new parameter manager
    pub fn new(config: OptimizationConfig<A>) -> Self {
        Self {
            parameters: HashMap::new(),
            optimizer_states: HashMap::new(),
            layer_architectures: HashMap::new(),
            sharing_groups: HashMap::new(),
            layer_rules: HashMap::new(),
            global_config: config,
            lazy_mode: false,
            pending_registrations: Vec::new(),
        }
    }

    /// Enable lazy registration mode
    pub fn enable_lazy_mode(&mut self) {
        self.lazy_mode = true;
    }

    /// Disable lazy registration mode and process pending registrations
    pub fn disable_lazy_mode(&mut self) -> Result<()> {
        self.lazy_mode = false;

        // Process all pending registrations
        let pending = std::mem::take(&mut self.pending_registrations);
        for (param_id, metadata) in pending {
            self.register_parameter_impl(param_id, metadata)?;
        }

        Ok(())
    }

    /// Register a layer architecture
    pub fn register_layer(&mut self, layer_id: LayerId, architecture: LayerArchitecture) {
        self.layer_architectures.insert(layer_id, architecture);
    }

    /// Set layer-specific optimization rule
    pub fn set_layer_rule(&mut self, layer_id: LayerId, rule: LayerOptimizationRule<A>) {
        self.layer_rules.insert(layer_id, rule);
    }

    /// Register a parameter
    pub fn register_parameter(
        &mut self,
        param_id: ParamId,
        metadata: ParameterMetadata,
    ) -> Result<()> {
        if self.lazy_mode {
            self.pending_registrations.push((param_id, metadata));
            Ok(())
        } else {
            self.register_parameter_impl(param_id, metadata)
        }
    }

    /// Internal parameter registration implementation
    fn register_parameter_impl(
        &mut self,
        param_id: ParamId,
        metadata: ParameterMetadata,
    ) -> Result<()> {
        // Handle parameter sharing
        if let Some(sharing_group) = &metadata.sharing_group {
            self.sharing_groups
                .entry(sharing_group.clone())
                .or_default()
                .push(param_id.clone());
        }

        // Initialize optimizer state for this parameter
        self.optimizer_states
            .insert(param_id.clone(), HashMap::new());

        // Store parameter metadata
        self.parameters.insert(param_id, metadata);

        Ok(())
    }

    /// Get effective learning rate for a parameter
    pub fn get_effective_learning_rate(&self, param_id: &ParamId) -> A {
        let base_lr = self.global_config.base_learning_rate;

        if let Some(metadata) = self.parameters.get(param_id) {
            if let Some(rule) = self.layer_rules.get(&metadata.layer_name) {
                return base_lr * rule.lr_multiplier;
            }
        }

        base_lr
    }

    /// Get effective weight decay for a parameter
    pub fn get_effective_weight_decay(&self, param_id: &ParamId) -> A {
        let base_decay = self.global_config.weight_decay;

        if let Some(metadata) = self.parameters.get(param_id) {
            if let Some(rule) = self.layer_rules.get(&metadata.layer_name) {
                return base_decay * rule.weight_decay_multiplier;
            }
        }

        base_decay
    }

    /// Check if parameter is frozen
    pub fn is_parameter_frozen(&self, param_id: &ParamId) -> bool {
        if let Some(metadata) = self.parameters.get(param_id) {
            if let Some(rule) = self.layer_rules.get(&metadata.layer_name) {
                return rule.frozen;
            }
        }
        false
    }

    /// Get parameters in a sharing group
    pub fn get_sharing_group(&self, group_name: &str) -> Option<&[ParamId]> {
        self.sharing_groups.get(group_name).map(|v| v.as_slice())
    }

    /// Get all registered parameters
    pub fn get_all_parameters(&self) -> &HashMap<ParamId, ParameterMetadata> {
        &self.parameters
    }

    /// Get layer architecture
    pub fn get_layer_architecture(&self, layer_id: &LayerId) -> Option<&LayerArchitecture> {
        self.layer_architectures.get(layer_id)
    }

    /// Get parameter metadata
    pub fn get_parameter_metadata(&self, param_id: &ParamId) -> Option<&ParameterMetadata> {
        self.parameters.get(param_id)
    }

    /// Update global configuration
    pub fn update_config(&mut self, config: OptimizationConfig<A>) {
        self.global_config = config;
    }

    /// Get optimizer state for parameter
    pub fn get_optimizer_state(&self, param_id: &ParamId) -> Option<&HashMap<String, Array<A, D>>> {
        self.optimizer_states.get(param_id)
    }

    /// Get mutable optimizer state for parameter
    pub fn get_optimizer_state_mut(
        &mut self,
        param_id: &ParamId,
    ) -> Option<&mut HashMap<String, Array<A, D>>> {
        self.optimizer_states.get_mut(param_id)
    }

    /// Initialize optimizer state for parameter
    pub fn init_optimizer_state(
        &mut self,
        param_id: &ParamId,
        state_name: &str,
        state: Array<A, D>,
    ) -> Result<()> {
        if let Some(states) = self.optimizer_states.get_mut(param_id) {
            states.insert(state_name.to_string(), state);
            Ok(())
        } else {
            Err(OptimError::InvalidConfig(format!(
                "Parameter {} not registered",
                param_id
            )))
        }
    }

    /// Reset all optimizer states
    pub fn reset_optimizer_states(&mut self) {
        for states in self.optimizer_states.values_mut() {
            states.clear();
        }
    }

    /// Get parameters by layer
    pub fn get_parameters_by_layer(&self, layer_id: &LayerId) -> Vec<&ParamId> {
        self.parameters
            .iter()
            .filter(|(_, metadata)| &metadata.layer_name == layer_id)
            .map(|(param_id, _)| param_id)
            .collect()
    }

    /// Get parameters by type
    pub fn get_parameters_by_type(&self, param_type: ParameterType) -> Vec<&ParamId> {
        self.parameters
            .iter()
            .filter(|(_, metadata)| metadata.param_type == param_type)
            .map(|(param_id, _)| param_id)
            .collect()
    }

    /// Get trainable parameters
    pub fn get_trainable_parameters(&self) -> Vec<&ParamId> {
        self.parameters
            .iter()
            .filter(|(param_id, metadata)| {
                metadata.requires_grad && !self.is_parameter_frozen(param_id)
            })
            .map(|(param_id, _)| param_id)
            .collect()
    }
}

impl<A: Float> Default for OptimizationConfig<A> {
    fn default() -> Self {
        Self {
            base_learning_rate: A::from(0.001).unwrap(),
            weight_decay: A::zero(),
            gradient_clip: None,
            mixed_precision: false,
            architecture_optimizations: HashMap::new(),
        }
    }
}

impl<A: Float> Default for LayerOptimizationRule<A> {
    fn default() -> Self {
        Self {
            lr_multiplier: A::one(),
            weight_decay_multiplier: A::one(),
            frozen: false,
            custom_settings: HashMap::new(),
        }
    }
}

/// Forward/backward pass integration
pub mod forward_backward {
    use super::*;

    /// Forward pass hook for parameter tracking
    pub trait ForwardHook<A: Float, D: Dimension> {
        /// Called before layer forward pass
        fn pre_forward(&mut self, layer_id: &LayerId, inputs: &[Array<A, D>]) -> Result<()>;

        /// Called after layer forward pass
        fn post_forward(&mut self, layer_id: &LayerId, outputs: &[Array<A, D>]) -> Result<()>;
    }

    /// Backward pass hook for gradient processing
    pub trait BackwardHook<A: Float, D: Dimension> {
        /// Called before layer backward pass
        fn pre_backward(&mut self, layer_id: &LayerId, grad_outputs: &[Array<A, D>]) -> Result<()>;

        /// Called after layer backward pass
        fn post_backward(&mut self, layer_id: &LayerId, grad_inputs: &[Array<A, D>]) -> Result<()>;
    }

    /// Neural network integration manager
    pub struct NeuralIntegration<A: Float, D: Dimension> {
        /// Parameter manager
        param_manager: ParameterManager<A, D>,
        /// Forward hooks
        forward_hooks: HashMap<LayerId, Box<dyn ForwardHook<A, D>>>,
        /// Backward hooks
        backward_hooks: HashMap<LayerId, Box<dyn BackwardHook<A, D>>>,
        /// Gradient accumulation mode
        gradient_accumulation: bool,
        /// Accumulated gradients
        accumulated_gradients: HashMap<ParamId, Array<A, D>>,
        /// Accumulation count
        accumulation_count: usize,
    }

    impl<A: Float + ScalarOperand + Debug + 'static, D: Dimension + 'static> NeuralIntegration<A, D> {
        /// Create a new neural integration manager
        pub fn new(config: OptimizationConfig<A>) -> Self {
            Self {
                param_manager: ParameterManager::new(config),
                forward_hooks: HashMap::new(),
                backward_hooks: HashMap::new(),
                gradient_accumulation: false,
                accumulated_gradients: HashMap::new(),
                accumulation_count: 0,
            }
        }

        /// Register a forward hook for a layer
        pub fn register_forward_hook<H>(&mut self, layer_id: LayerId, hook: H)
        where
            H: ForwardHook<A, D> + 'static,
        {
            self.forward_hooks.insert(layer_id, Box::new(hook));
        }

        /// Register a backward hook for a layer
        pub fn register_backward_hook<H>(&mut self, layer_id: LayerId, hook: H)
        where
            H: BackwardHook<A, D> + 'static,
        {
            self.backward_hooks.insert(layer_id, Box::new(hook));
        }

        /// Enable gradient accumulation
        pub fn enable_gradient_accumulation(&mut self) {
            self.gradient_accumulation = true;
        }

        /// Disable gradient accumulation and return accumulated gradients
        pub fn disable_gradient_accumulation(&mut self) -> HashMap<ParamId, Array<A, D>> {
            self.gradient_accumulation = false;
            let result = std::mem::take(&mut self.accumulated_gradients);
            self.accumulation_count = 0;
            result
        }

        /// Execute forward pass with hooks
        pub fn forward_pass(
            &mut self,
            layer_id: &LayerId,
            inputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Execute pre-forward hook
            if let Some(hook) = self.forward_hooks.get_mut(layer_id) {
                hook.pre_forward(layer_id, inputs)?;
            }

            // TODO: Actual forward computation would be implemented by the neural network framework
            // For now, we'll return the inputs as a placeholder
            let outputs = inputs.to_vec();

            // Execute post-forward hook
            if let Some(hook) = self.forward_hooks.get_mut(layer_id) {
                hook.post_forward(layer_id, &outputs)?;
            }

            Ok(outputs)
        }

        /// Execute backward pass with hooks
        pub fn backward_pass(
            &mut self,
            layer_id: &LayerId,
            grad_outputs: &[Array<A, D>],
        ) -> Result<Vec<Array<A, D>>> {
            // Execute pre-backward hook
            if let Some(hook) = self.backward_hooks.get_mut(layer_id) {
                hook.pre_backward(layer_id, grad_outputs)?;
            }

            // TODO: Actual backward computation would be implemented by the neural network framework
            // For now, we'll return the grad_outputs as a placeholder
            let grad_inputs = grad_outputs.to_vec();

            // Execute post-backward hook
            if let Some(hook) = self.backward_hooks.get_mut(layer_id) {
                hook.post_backward(layer_id, &grad_inputs)?;
            }

            Ok(grad_inputs)
        }

        /// Accumulate gradients for parameters
        pub fn accumulate_gradients(
            &mut self,
            gradients: HashMap<ParamId, Array<A, D>>,
        ) -> Result<()> {
            if !self.gradient_accumulation {
                return Err(OptimError::InvalidConfig(
                    "Gradient accumulation not enabled".to_string(),
                ));
            }

            self.accumulation_count += 1;

            for (param_id, grad) in gradients {
                if let Some(acc_grad) = self.accumulated_gradients.get_mut(&param_id) {
                    // Add to existing accumulated gradient
                    *acc_grad = acc_grad.clone() + grad;
                } else {
                    // First gradient for this parameter
                    self.accumulated_gradients.insert(param_id, grad);
                }
            }

            Ok(())
        }

        /// Get parameter manager
        pub fn parameter_manager(&self) -> &ParameterManager<A, D> {
            &self.param_manager
        }

        /// Get mutable parameter manager
        pub fn parameter_manager_mut(&mut self) -> &mut ParameterManager<A, D> {
            &mut self.param_manager
        }

        /// Get accumulation count
        pub fn accumulation_count(&self) -> usize {
            self.accumulation_count
        }
    }
}

/// Architecture-aware optimization utilities
pub mod architecture_aware {
    use super::*;

    /// Architecture-specific optimization strategy
    #[derive(Debug, Clone)]
    pub enum ArchitectureStrategy {
        /// Transformer-specific optimizations
        Transformer {
            /// Use different learning rates for different components
            component_specific_lr: bool,
            /// Apply layer-wise learning rate decay
            layer_wise_decay: bool,
            /// Warmup steps for attention parameters
            attention_warmup: usize,
        },
        /// CNN-specific optimizations
        ConvolutionalNet {
            /// Use different learning rates for conv vs fc layers
            layer_type_lr: bool,
            /// Apply depth-wise learning rate scaling
            depth_scaling: bool,
            /// Batch norm parameter handling
            bn_special_handling: bool,
        },
        /// RNN-specific optimizations
        RecurrentNet {
            /// Gradient clipping specifically for RNNs
            rnn_gradient_clip: Option<f64>,
            /// Different learning rates for recurrent vs linear weights
            weight_type_lr: bool,
        },
        /// Custom architecture strategy
        Custom {
            /// Custom optimization rules
            rules: HashMap<String, LayerConfig>,
        },
    }

    /// Architecture-aware optimizer
    #[derive(Debug)]
    pub struct ArchitectureAwareOptimizer<A: Float, D: Dimension> {
        /// Parameter manager
        param_manager: ParameterManager<A, D>,
        /// Architecture strategy
        strategy: ArchitectureStrategy,
        /// Step count
        step_count: usize,
    }

    impl<A: Float + ScalarOperand + Debug, D: Dimension> ArchitectureAwareOptimizer<A, D> {
        /// Create a new architecture-aware optimizer
        pub fn new(config: OptimizationConfig<A>, strategy: ArchitectureStrategy) -> Self {
            Self {
                param_manager: ParameterManager::new(config),
                strategy,
                step_count: 0,
            }
        }

        /// Apply architecture-specific optimizations
        pub fn apply_architecture_optimizations(&mut self) -> Result<()> {
            // Clone the strategy to avoid borrowing conflicts
            let strategy = self.strategy.clone();
            match strategy {
                ArchitectureStrategy::Transformer {
                    component_specific_lr,
                    layer_wise_decay,
                    attention_warmup,
                } => {
                    self.apply_transformer_optimizations(
                        component_specific_lr,
                        layer_wise_decay,
                        attention_warmup,
                    )?;
                }
                ArchitectureStrategy::ConvolutionalNet {
                    layer_type_lr,
                    depth_scaling,
                    bn_special_handling,
                } => {
                    self.apply_cnn_optimizations(
                        layer_type_lr,
                        depth_scaling,
                        bn_special_handling,
                    )?;
                }
                ArchitectureStrategy::RecurrentNet {
                    rnn_gradient_clip,
                    weight_type_lr,
                } => {
                    self.apply_rnn_optimizations(rnn_gradient_clip, weight_type_lr)?;
                }
                ArchitectureStrategy::Custom { rules } => {
                    self.apply_custom_optimizations(&rules)?;
                }
            }
            Ok(())
        }

        /// Apply Transformer-specific optimizations
        fn apply_transformer_optimizations(
            &mut self,
            component_specific_lr: bool,
            layer_wise_decay: bool,
            attention_warmup: usize,
        ) -> Result<()> {
            if component_specific_lr {
                // Different learning rates for attention, ffn, and normalization
                self.set_component_learning_rates()?;
            }

            if layer_wise_decay {
                // Apply layer-wise learning rate decay
                self.apply_layer_wise_decay()?;
            }

            if attention_warmup > 0 && self.step_count < attention_warmup {
                // Apply warmup to attention parameters
                self.apply_attention_warmup(attention_warmup)?;
            }

            Ok(())
        }

        /// Apply CNN-specific optimizations
        fn apply_cnn_optimizations(
            &mut self,
            layer_type_lr: bool,
            depth_scaling: bool,
            bn_special_handling: bool,
        ) -> Result<()> {
            if layer_type_lr {
                // Different learning rates for conv vs fully connected layers
                self.set_layer_type_learning_rates()?;
            }

            if depth_scaling {
                // Scale learning rates based on network depth
                self.apply_depth_scaling()?;
            }

            if bn_special_handling {
                // Special handling for batch normalization parameters
                self.apply_bn_optimizations()?;
            }

            Ok(())
        }

        /// Apply RNN-specific optimizations
        fn apply_rnn_optimizations(
            &mut self,
            rnn_gradient_clip: Option<f64>,
            weight_type_lr: bool,
        ) -> Result<()> {
            if let Some(clip_value) = rnn_gradient_clip {
                // Apply RNN-specific gradient clipping
                self.apply_rnn_gradient_clipping(A::from(clip_value).unwrap())?;
            }

            if weight_type_lr {
                // Different learning rates for recurrent vs linear weights
                self.set_weight_type_learning_rates()?;
            }

            Ok(())
        }

        /// Apply custom optimizations
        fn apply_custom_optimizations(
            &mut self,
            rules: &HashMap<String, LayerConfig>,
        ) -> Result<()> {
            // Collect rules first to avoid borrowing conflicts
            let rule_entries: Vec<(String, LayerConfig)> = rules
                .iter()
                .map(|(name, config)| (name.clone(), config.clone()))
                .collect();

            for (rule_name, config) in rule_entries {
                self.apply_custom_rule(&rule_name, &config)?;
            }
            Ok(())
        }

        /// Set component-specific learning rates for Transformers
        fn set_component_learning_rates(&mut self) -> Result<()> {
            // Collect the data first to avoid borrowing conflicts
            let layer_rules: Vec<(LayerId, LayerOptimizationRule<A>)> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .map(|(_param_id, metadata)| {
                    let mut rule = LayerOptimizationRule::default();

                    // Determine learning rate multiplier based on parameter tags
                    if metadata.tags.contains(&"attention".to_string()) {
                        rule.lr_multiplier = A::from(1.2).unwrap(); // Higher LR for attention
                    } else if metadata.tags.contains(&"ffn".to_string()) {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Standard LR for FFN
                    } else if metadata.tags.contains(&"normalization".to_string()) {
                        rule.lr_multiplier = A::from(0.8).unwrap(); // Lower LR for normalization
                    }

                    (metadata.layer_name.clone(), rule)
                })
                .collect();

            // Now apply the rules
            for (layer_name, rule) in layer_rules {
                self.param_manager.set_layer_rule(layer_name, rule);
            }
            Ok(())
        }

        /// Apply layer-wise learning rate decay
        fn apply_layer_wise_decay(&mut self) -> Result<()> {
            // Extract layer numbers from layer names and apply decay
            for (layer_id, _) in self.param_manager.layer_architectures.clone() {
                if let Some(layer_num) = self.extract_layer_number(&layer_id) {
                    let decay_factor = A::from(0.95_f64.powi(layer_num as i32)).unwrap();
                    let mut rule = self
                        .param_manager
                        .layer_rules
                        .get(&layer_id)
                        .cloned()
                        .unwrap_or_default();
                    rule.lr_multiplier = rule.lr_multiplier * decay_factor;
                    self.param_manager.set_layer_rule(layer_id, rule);
                }
            }
            Ok(())
        }

        /// Apply attention parameter warmup
        fn apply_attention_warmup(&mut self, warmup_steps: usize) -> Result<()> {
            let warmup_factor = A::from(self.step_count as f64 / warmup_steps as f64).unwrap();

            // Collect attention layers first
            let attention_layers: Vec<LayerId> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .filter_map(|(_param_id, metadata)| {
                    if metadata.tags.contains(&"attention".to_string()) {
                        Some(metadata.layer_name.clone())
                    } else {
                        None
                    }
                })
                .collect();

            // Apply warmup to attention layers
            for layer_name in attention_layers {
                let mut rule = self
                    .param_manager
                    .layer_rules
                    .get(&layer_name)
                    .cloned()
                    .unwrap_or_default();
                rule.lr_multiplier = rule.lr_multiplier * warmup_factor;
                self.param_manager.set_layer_rule(layer_name, rule);
            }
            Ok(())
        }

        /// Set learning rates based on layer type (conv vs fc)
        fn set_layer_type_learning_rates(&mut self) -> Result<()> {
            for (layer_id, architecture) in self.param_manager.layer_architectures.clone() {
                let mut rule = LayerOptimizationRule::default();

                match architecture.layer_type.as_str() {
                    "conv" | "conv2d" | "conv3d" => {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Standard LR for conv
                    }
                    "linear" | "dense" | "fc" => {
                        rule.lr_multiplier = A::from(0.8).unwrap(); // Lower LR for FC
                    }
                    _ => {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Default
                    }
                }

                self.param_manager.set_layer_rule(layer_id, rule);
            }
            Ok(())
        }

        /// Apply depth-based scaling
        fn apply_depth_scaling(&mut self) -> Result<()> {
            // Count total layers
            let total_layers = self.param_manager.layer_architectures.len();

            for (i, (layer_id, _)) in self
                .param_manager
                .layer_architectures
                .clone()
                .iter()
                .enumerate()
            {
                let depth_factor = A::from(1.0 - 0.1 * (i as f64 / total_layers as f64)).unwrap();
                let mut rule = self
                    .param_manager
                    .layer_rules
                    .get(layer_id)
                    .cloned()
                    .unwrap_or_default();
                rule.lr_multiplier = rule.lr_multiplier * depth_factor;
                self.param_manager.set_layer_rule(layer_id.clone(), rule);
            }
            Ok(())
        }

        /// Apply batch normalization optimizations
        fn apply_bn_optimizations(&mut self) -> Result<()> {
            // Collect normalization layers first
            let norm_layers: Vec<LayerId> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .filter_map(|(_param_id, metadata)| {
                    if metadata.param_type == ParameterType::Normalization {
                        Some(metadata.layer_name.clone())
                    } else {
                        None
                    }
                })
                .collect();

            // Apply optimization to normalization layers
            for layer_name in norm_layers {
                let mut rule = self
                    .param_manager
                    .layer_rules
                    .get(&layer_name)
                    .cloned()
                    .unwrap_or_default();
                // Higher learning rate and no weight decay for BN parameters
                rule.lr_multiplier = A::from(2.0).unwrap();
                rule.weight_decay_multiplier = A::zero();
                self.param_manager.set_layer_rule(layer_name, rule);
            }
            Ok(())
        }

        /// Apply RNN-specific gradient clipping
        fn apply_rnn_gradient_clipping(&mut self, clip_value: A) -> Result<()> {
            // This would be implemented in coordination with the gradient processing system
            // For now, we'll store the clip value in the global config
            self.param_manager.global_config.gradient_clip = Some(clip_value);
            Ok(())
        }

        /// Set learning rates based on weight type (recurrent vs linear)
        fn set_weight_type_learning_rates(&mut self) -> Result<()> {
            // Collect weight type layers first
            let layer_rules: Vec<(LayerId, LayerOptimizationRule<A>)> = self
                .param_manager
                .get_all_parameters()
                .iter()
                .map(|(_param_id, metadata)| {
                    let mut rule = LayerOptimizationRule::default();

                    if metadata.tags.contains(&"recurrent".to_string()) {
                        rule.lr_multiplier = A::from(0.5).unwrap(); // Lower LR for recurrent weights
                    } else if metadata.tags.contains(&"linear".to_string()) {
                        rule.lr_multiplier = A::from(1.0).unwrap(); // Standard LR for linear weights
                    }

                    (metadata.layer_name.clone(), rule)
                })
                .collect();

            // Apply the rules
            for (layer_name, rule) in layer_rules {
                self.param_manager.set_layer_rule(layer_name, rule);
            }
            Ok(())
        }

        /// Apply custom optimization rule
        fn apply_custom_rule(&mut self, _rule_name: &str, _config: &LayerConfig) -> Result<()> {
            // Custom rule implementation would depend on the specific rule
            // This is a placeholder for extensibility
            Ok(())
        }

        /// Extract layer number from layer name (e.g., "layer_12" -> 12)
        fn extract_layer_number(&self, layer_name: &str) -> Option<usize> {
            layer_name.split('_').next_back()?.parse().ok()
        }

        /// Step the optimizer
        pub fn step(&mut self) -> Result<()> {
            self.step_count += 1;
            self.apply_architecture_optimizations()
        }

        /// Get parameter manager
        pub fn parameter_manager(&self) -> &ParameterManager<A, D> {
            &self.param_manager
        }

        /// Get mutable parameter manager
        pub fn parameter_manager_mut(&mut self) -> &mut ParameterManager<A, D> {
            &mut self.param_manager
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parameter_manager_basic() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, ndarray::Ix1>::new(config);

        let metadata = ParameterMetadata {
            layer_name: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            param_type: ParameterType::Weight,
            sharing_group: None,
            tags: vec!["dense".to_string()],
        };

        manager
            .register_parameter("param1".to_string(), metadata)
            .unwrap();

        assert!(manager
            .get_parameter_metadata(&"param1".to_string())
            .is_some());
        assert_eq!(manager.get_all_parameters().len(), 1);
        assert!(!manager.is_parameter_frozen(&"param1".to_string()));
    }

    #[test]
    fn test_lazy_registration() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, ndarray::Ix1>::new(config);

        manager.enable_lazy_mode();

        let metadata = ParameterMetadata {
            layer_name: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            param_type: ParameterType::Weight,
            sharing_group: None,
            tags: vec![],
        };

        manager
            .register_parameter("param1".to_string(), metadata)
            .unwrap();

        // Parameter should not be registered yet
        assert_eq!(manager.get_all_parameters().len(), 0);

        // Disable lazy mode to process pending registrations
        manager.disable_lazy_mode().unwrap();

        // Now parameter should be registered
        assert_eq!(manager.get_all_parameters().len(), 1);
    }

    #[test]
    fn test_layer_specific_rules() {
        let config = OptimizationConfig {
            base_learning_rate: 0.01,
            weight_decay: 0.001,
            gradient_clip: None,
            mixed_precision: false,
            architecture_optimizations: HashMap::new(),
        };
        let mut manager = ParameterManager::<f64, ndarray::Ix1>::new(config);

        let rule = LayerOptimizationRule {
            lr_multiplier: 2.0,
            weight_decay_multiplier: 0.5,
            frozen: false,
            custom_settings: HashMap::new(),
        };

        manager.set_layer_rule("layer1".to_string(), rule);

        let metadata = ParameterMetadata {
            layer_name: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            param_type: ParameterType::Weight,
            sharing_group: None,
            tags: vec![],
        };

        manager
            .register_parameter("param1".to_string(), metadata)
            .unwrap();

        // Test effective learning rate
        let effective_lr = manager.get_effective_learning_rate(&"param1".to_string());
        assert_relative_eq!(effective_lr, 0.02, epsilon = 1e-6); // 0.01 * 2.0

        // Test effective weight decay
        let effective_decay = manager.get_effective_weight_decay(&"param1".to_string());
        assert_relative_eq!(effective_decay, 0.0005, epsilon = 1e-6); // 0.001 * 0.5
    }

    #[test]
    fn test_parameter_sharing() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, ndarray::Ix1>::new(config);

        let metadata1 = ParameterMetadata {
            layer_name: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            param_type: ParameterType::Weight,
            sharing_group: Some("shared_weights".to_string()),
            tags: vec![],
        };

        let metadata2 = ParameterMetadata {
            layer_name: "layer2".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            param_type: ParameterType::Weight,
            sharing_group: Some("shared_weights".to_string()),
            tags: vec![],
        };

        manager
            .register_parameter("param1".to_string(), metadata1)
            .unwrap();
        manager
            .register_parameter("param2".to_string(), metadata2)
            .unwrap();

        let sharing_group = manager.get_sharing_group("shared_weights").unwrap();
        assert_eq!(sharing_group.len(), 2);
        assert!(sharing_group.contains(&"param1".to_string()));
        assert!(sharing_group.contains(&"param2".to_string()));
    }

    #[test]
    fn test_parameter_filtering() {
        let config = OptimizationConfig::default();
        let mut manager = ParameterManager::<f64, ndarray::Ix1>::new(config);

        let weight_metadata = ParameterMetadata {
            layer_name: "layer1".to_string(),
            param_name: "weight".to_string(),
            shape: vec![10, 5],
            requires_grad: true,
            param_type: ParameterType::Weight,
            sharing_group: None,
            tags: vec![],
        };

        let bias_metadata = ParameterMetadata {
            layer_name: "layer1".to_string(),
            param_name: "bias".to_string(),
            shape: vec![5],
            requires_grad: true,
            param_type: ParameterType::Bias,
            sharing_group: None,
            tags: vec![],
        };

        manager
            .register_parameter("weight".to_string(), weight_metadata)
            .unwrap();
        manager
            .register_parameter("bias".to_string(), bias_metadata)
            .unwrap();

        // Test filtering by type
        let weights = manager.get_parameters_by_type(ParameterType::Weight);
        assert_eq!(weights.len(), 1);
        assert_eq!(weights[0], &"weight".to_string());

        let biases = manager.get_parameters_by_type(ParameterType::Bias);
        assert_eq!(biases.len(), 1);
        assert_eq!(biases[0], &"bias".to_string());

        // Test filtering by layer
        let layer_params = manager.get_parameters_by_layer(&"layer1".to_string());
        assert_eq!(layer_params.len(), 2);

        // Test trainable parameters
        let trainable = manager.get_trainable_parameters();
        assert_eq!(trainable.len(), 2);
    }

    #[test]
    fn test_architecture_aware_transformer() {
        use crate::neural_integration::architecture_aware::*;

        let config = OptimizationConfig::default();
        let strategy = ArchitectureStrategy::Transformer {
            component_specific_lr: true,
            layer_wise_decay: true,
            attention_warmup: 1000,
        };

        let mut optimizer = ArchitectureAwareOptimizer::<f64, ndarray::Ix1>::new(config, strategy);

        // Register a layer architecture
        let layer_arch = LayerArchitecture {
            layer_type: "transformer_block".to_string(),
            input_dims: vec![512],
            output_dims: vec![512],
            config: HashMap::new(),
            trainable: true,
        };

        optimizer
            .parameter_manager_mut()
            .register_layer("layer_0".to_string(), layer_arch);

        // Register parameters with different tags
        let attention_metadata = ParameterMetadata {
            layer_name: "layer_0".to_string(),
            param_name: "attention_weight".to_string(),
            shape: vec![512, 512],
            requires_grad: true,
            param_type: ParameterType::Attention,
            sharing_group: None,
            tags: vec!["attention".to_string()],
        };

        optimizer
            .parameter_manager_mut()
            .register_parameter("attn_param".to_string(), attention_metadata)
            .unwrap();

        // Apply optimizations
        optimizer.apply_architecture_optimizations().unwrap();

        // Verify that attention parameters get special treatment
        assert!(optimizer
            .parameter_manager()
            .get_parameter_metadata(&"attn_param".to_string())
            .is_some());
    }
}
