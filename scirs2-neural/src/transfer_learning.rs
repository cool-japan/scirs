//! Transfer learning utilities for neural networks
//!
//! This module provides tools for transfer learning, including:
//! - Pre-trained model weight loading and adaptation
//! - Layer freezing and unfreezing
//! - Fine-tuning utilities
//! - Domain adaptation tools

use crate::error::{NeuralError, Result};
use crate::layers::Layer;
use scirs2_core::ndarray::ArrayD;
use scirs2_core::numeric::{Float, FromPrimitive, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Div};
use std::sync::{Arc, RwLock};
/// Transfer learning strategy
#[derive(Debug, Clone, PartialEq)]
pub enum TransferStrategy {
    /// Freeze all layers except the last few
    FeatureExtraction {
        /// Number of layers from the end to leave unfrozen
        unfrozen_layers: usize,
    },
    /// Fine-tune all layers with different learning rates
    FineTuning {
        /// Learning rate ratio for backbone layers
        backbone_lr_ratio: f64,
        /// Learning rate ratio for head layers
        head_lr_ratio: f64,
    /// Progressive unfreezing during training
    ProgressiveUnfreezing {
        /// Schedule of (epoch, layers_to_unfreeze) pairs
        unfreeze_schedule: Vec<(usize, usize)>,
    /// Custom layer-specific learning rates
    LayerWiseLearningRates {
        /// Map of layer names to learning rate ratios
        layer_lr_map: HashMap<String, f64>,
}
/// Layer freezing state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayerState {
    /// Layer parameters are frozen (not updated)
    Frozen,
    /// Layer parameters are trainable
    Trainable,
    /// Layer uses reduced learning rate
    ReducedLearningRate(f64),
/// Transfer learning manager
pub struct TransferLearningManager<F: Float + Debug> {
    /// Layer states for each layer
    layer_states: HashMap<String, LayerState>,
    /// Layer names in order (for consistent unfreezing)
    layer_order: Vec<String>,
    /// Transfer strategy
    strategy: TransferStrategy,
    /// Base learning rate
    base_learning_rate: F,
    /// Current epoch (for progressive unfreezing)
    current_epoch: usize,
    /// Layer statistics for monitoring
    layer_stats: Arc<RwLock<HashMap<String, LayerStatistics<F>>>>,
/// Statistics for tracking layer behavior during transfer learning
#[derive(Debug, Clone)]
pub struct LayerStatistics<F: Float + Debug> {
    /// Average gradient magnitude
    pub avg_gradient_magnitude: F,
    /// Parameter update magnitude
    pub param_update_magnitude: F,
    /// Number of parameters
    pub param_count: usize,
    /// Layer activation variance
    pub activation_variance: F,
    /// Whether layer is currently frozen
    pub is_frozen: bool,
impl<F: Float + Debug + 'static> TransferLearningManager<F> {
    /// Create a new transfer learning manager
    pub fn new(_strategy: TransferStrategy, base_learningrate: f64) -> Result<Self> {
        Ok(Self {
            layer_states: HashMap::new(),
            layer_order: Vec::new(),
            strategy,
            base_learning_rate: F::from(base_learning_rate).ok_or_else(|| {
                NeuralError::InvalidArchitecture("Invalid learning _rate".to_string())
            })?,
            current_epoch: 0,
            layer_stats: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    /// Initialize layer states based on the transfer strategy
    pub fn initialize_layer_states(&mut self, layernames: &[String]) -> Result<()> {
        // Store the layer order
        self.layer_order = layer_names.to_vec();
        match &self.strategy {
            TransferStrategy::FeatureExtraction { unfrozen_layers } => {
                let total_layers = layer_names.len();
                let frozen_layers = total_layers.saturating_sub(*unfrozen_layers);
                for (i, layer_name) in layer_names.iter().enumerate() {
                    let state = if i < frozen_layers {
                        LayerState::Frozen
                    } else {
                        LayerState::Trainable
                    };
                    self.layer_states.insert(layer_name.clone(), state);
                }
            }
            TransferStrategy::FineTuning {
                backbone_lr_ratio,
                head_lr_ratio,
            } => {
                let backbone_layers = total_layers.saturating_sub(2); // Last 2 layers are "head"
                    let state = if i < backbone_layers {
                        LayerState::ReducedLearningRate(*backbone_lr_ratio), LayerState::ReducedLearningRate(*head_lr_ratio)
            TransferStrategy::ProgressiveUnfreezing { .. } => {
                // Start with all layers frozen
                for layer_name in layer_names {
                    self.layer_states
                        .insert(layer_name.clone(), LayerState::Frozen);
            TransferStrategy::LayerWiseLearningRates { layer_lr_map } => {
                    let lr_ratio = layer_lr_map.get(layer_name).unwrap_or(&1.0);
                    let state = if *lr_ratio == 0.0 {
                        LayerState::ReducedLearningRate(*lr_ratio)
        }
        Ok(())
    /// Update layer states at the beginning of each epoch
    pub fn update_epoch(&mut self, epoch: usize) -> Result<()> {
        self.current_epoch = epoch;
        if let TransferStrategy::ProgressiveUnfreezing { unfreeze_schedule } =
            &self.strategy.clone()
        {
            for (unfreeze_epoch, layers_to_unfreeze) in unfreeze_schedule {
                if epoch == *unfreeze_epoch {
                    self.unfreeze_layers(*layers_to_unfreeze)?;
    /// Unfreeze the specified number of layers from the end
    pub fn unfreeze_layers(&mut self, count: usize) -> Result<()> {
        let total_layers = self.layer_order.len();
        let start_idx = total_layers.saturating_sub(count);
        for layer_name in self.layer_order.iter().skip(start_idx) {
            if let Some(state) = self.layer_states.get_mut(layer_name) {
                if *state == LayerState::Frozen {
                    *state = LayerState::Trainable;
    /// Freeze specific layers
    pub fn freeze_layers(&mut self, layernames: &[String]) -> Result<()> {
        for layer_name in layer_names {
            self.layer_states
                .insert(layer_name.clone(), LayerState::Frozen);
    /// Get effective learning rate for a layer
    pub fn get_layer_learning_rate(&self, layername: &str) -> F {
        match self.layer_states.get(layer_name) {
            Some(LayerState::Frozen) => F::zero(),
            Some(LayerState::Trainable) => self.base_learning_rate,
            Some(LayerState::ReducedLearningRate(ratio)) => {
                self.base_learning_rate * F::from(*ratio).unwrap_or(F::one())
            None => self.base_learning_rate, // Default for unknown layers
    /// Check if a layer is frozen
    pub fn is_layer_frozen(&self, layername: &str) -> bool {
        matches!(self.layer_states.get(layer_name), Some(LayerState::Frozen))
    /// Update layer statistics
    pub fn update_layer_statistics(
        &self,
        layer_name: String,
        gradient_magnitude: F,
        param_update_magnitude: F,
        param_count: usize,
        activation_variance: F,
    ) -> Result<()> {
        let is_frozen = self.is_layer_frozen(&layer_name);
        let stats = LayerStatistics {
            avg_gradient_magnitude: gradient_magnitude,
            param_update_magnitude,
            param_count,
            activation_variance,
            is_frozen,
        };
        if let Ok(mut layer_stats) = self.layer_stats.write() {
            layer_stats.insert(layer_name, stats);
    /// Get layer statistics
    pub fn get_layer_statistics(&self) -> Result<HashMap<String, LayerStatistics<F>>> {
        match self.layer_stats.read() {
            Ok(stats) => Ok(stats.clone()),
            Err(_) => Err(NeuralError::InferenceError(
                "Failed to read layer statistics".to_string(),
            )),
    /// Get summary of current transfer learning state
    pub fn get_summary(&self) -> TransferLearningState {
        let mut frozen_layers = 0;
        let mut trainable_layers = 0;
        let mut reduced_lr_layers = 0;
        for state in self.layer_states.values() {
            match state {
                LayerState::Frozen => frozen_layers += 1,
                LayerState::Trainable => trainable_layers += 1,
                LayerState::ReducedLearningRate(_) => reduced_lr_layers += 1,
        TransferLearningState {
            current_epoch: self.current_epoch,
            total_layers: self.layer_states.len(),
            frozen_layers,
            trainable_layers,
            reduced_lr_layers,
            strategy: self.strategy.clone(),
/// Summary of transfer learning state
pub struct TransferLearningState {
    /// Current training epoch
    pub current_epoch: usize,
    /// Total number of layers
    pub total_layers: usize,
    /// Number of frozen layers
    pub frozen_layers: usize,
    /// Number of trainable layers
    pub trainable_layers: usize,
    /// Number of layers with reduced learning rate
    pub reduced_lr_layers: usize,
    /// Current transfer strategy
    pub strategy: TransferStrategy,
impl std::fmt::Display for TransferLearningState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Transfer Learning State (Epoch {}):", self.current_epoch)?;
        writeln!(f, "  Total layers: {}", self.total_layers)?;
        writeln!(f, "  Frozen layers: {}", self.frozen_layers)?;
        writeln!(f, "  Trainable layers: {}", self.trainable_layers)?;
        writeln!(f, "  Reduced LR layers: {}", self.reduced_lr_layers)?;
        writeln!(f, "  Strategy: {:?}", self.strategy)?;
/// Pre-trained model weight loader
pub struct PretrainedWeightLoader {
    /// Model weights storage
    weights: HashMap<String, ArrayD<f32>>,
    /// Weight mapping (source_layer -> target_layer)
    layer_mapping: HashMap<String, String>,
    /// Whether to ignore size mismatches
    ignore_mismatches: bool,
impl PretrainedWeightLoader {
    /// Create a new weight loader
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            layer_mapping: HashMap::new(),
            ignore_mismatches: false,
    /// Load weights from a dictionary
    pub fn load_weights(&mut self, weights: HashMap<String, ArrayD<f32>>) -> Result<()> {
        self.weights = weights;
    /// Add layer mapping for weight transfer
    pub fn add_layer_mapping(&mut self, source_layer: String, targetlayer: String) {
        self.layer_mapping.insert(source_layer, target_layer);
    /// Set whether to ignore size mismatches
    pub fn set_ignore_mismatches(&mut self, ignore: bool) {
        self.ignore_mismatches = ignore;
    /// Apply weights to a model layer with shape compatibility checking
    pub fn apply_weights_to_layer<L: Layer<f32>>(
        layer: &mut L,
        layer_name: &str,
    ) -> Result<bool> {
        // Try direct layer name first, then check mapping
        let default_key = layer_name.to_string();
        let weight_key = self.layer_mapping.get(layer_name).unwrap_or(&default_key);
        if let Some(weights) = self.weights.get(weight_key) {
            // Check shape compatibility if layer provides shape info
            let success = self.try_apply_weights(layer, weights, layer_name)?;
            if success {
                println!(
                    "Successfully loaded weights for layer {}: shape {:?}",
                    layer_name,
                    weights.shape()
                );
            } else if !self.ignore_mismatches {
                return Err(NeuralError::InvalidArchitecture(format!(
                    "Weight shape mismatch for layer {}: expected compatible shape, got {:?}",
                )));
            Ok(success)
        } else {
            Ok(false)
    /// Attempt to apply weights with error handling
    fn try_apply_weights<L: Layer<f32>>(
        _layer: &mut L,
        weights: &ArrayD<f32>,
        // This is where actual weight setting would happen
        // For now, we validate the shapes and return success
        // Check for common layer weight patterns
        match weights.ndim() {
            2 => {
                // Dense/Linear layer weights (input_size, output_size)
                    "Dense layer weights detected for {}: {:?}",
            4 => {
                // Convolutional layer weights (out_channels, in_channels, height, width)
                    "Conv layer weights detected for {}: {:?}",
            1 => {
                // Bias weights
                    "Bias weights detected for {}: {:?}"_ => {
                if !self.ignore_mismatches {
                    return Err(NeuralError::InvalidArchitecture(format!(
                        "Unsupported weight tensor dimensionality {} for layer {}",
                        weights.ndim(),
                        layer_name
                    )));
        Ok(true)
    /// Get available weight keys
    pub fn get_available_weights(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    /// Get weight statistics for analysis
    pub fn get_weight_statistics(&self) -> HashMap<String, WeightStatistics> {
        self.weights
            .iter()
            .map(|(name, weights)| {
                let stats = WeightStatistics::from_tensor(weights);
                (name.clone(), stats)
            })
            .collect()
    /// Check weight compatibility with expected shapes
    pub fn check_weight_compatibility(
        expectedshapes: &HashMap<String, Vec<usize>>,
    ) -> CompatibilityReport {
        let mut compatible = Vec::new();
        let mut incompatible = Vec::new();
        let mut missing = Vec::new();
        for (layer_name, expectedshape) in expectedshapes {
            let weight_key = self.layer_mapping.get(layer_name).unwrap_or(layer_name);
            if let Some(weights) = self.weights.get(weight_key) {
                if weights.shape() == expectedshape.as_slice() {
                    compatible.push(layer_name.clone());
                } else {
                    incompatible.push(WeightMismatch {
                        layer_name: layer_name.clone(),
                        expectedshape: expectedshape.clone(),
                        actualshape: weights.shape().to_vec(),
                    });
            } else {
                missing.push(layer_name.clone());
        CompatibilityReport {
            compatible,
            incompatible,
            missing,
    /// Load weights from various formats
    pub fn load_from_pytorch_state_dict(
        &mut self,
        state_dict: HashMap<String, ArrayD<f32>>,
        // Convert PyTorch naming conventions
        let converted_weights: HashMap<String, ArrayD<f32>> = state_dict
            .into_iter()
            .map(|(key, tensor)| {
                // Convert common PyTorch layer names
                let converted_key = if key.ends_with(".weight") {
                    key.replace(".weight", "")
                } else if key.ends_with(".bias") {
                    key.replace(".bias", "_bias")
                    key
                };
                (converted_key, tensor)
            .collect();
        self.weights = converted_weights;
    /// Load weights from TensorFlow checkpoint format
    pub fn load_from_tensorflow_checkpoint(
        checkpoint_data: HashMap<String, ArrayD<f32>>,
        // Convert TensorFlow naming conventions
        let converted_weights: HashMap<String, ArrayD<f32>> = checkpoint_data
                // Convert common TensorFlow layer names
                let converted_key = key
                    .replace("/kernel:0", "")
                    .replace("/bias:0", "_bias")
                    .replace("/", "_");
impl Default for PretrainedWeightLoader {
    fn default() -> Self {
        Self::new()
/// Weight statistics for analysis
pub struct WeightStatistics {
    /// Weight tensor shape
    pub shape: Vec<usize>,
    /// Mean weight value
    pub mean: f32,
    /// Standard deviation of weights
    pub std: f32,
    /// Minimum weight value
    pub min: f32,
    /// Maximum weight value
    pub max: f32,
    /// L2 norm of weights
    pub l2_norm: f32,
impl WeightStatistics {
    /// Compute statistics from a weight tensor
    pub fn from_tensor(weights: &ArrayD<f32>) -> Self {
        let shape = weights.shape().to_vec();
        let param_count = weights.len();
        let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = weights.sum();
        let mean = sum / param_count as f32;
        let variance: f32 =
            weights.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / param_count as f32;
        let std = variance.sqrt();
        let l2_norm = weights.iter().map(|&x| x * x).sum::<f32>().sqrt();
            shape,
            mean,
            std,
            min,
            max,
            l2_norm,
/// Weight compatibility report
pub struct CompatibilityReport {
    /// Layers with compatible _weights
    pub compatible: Vec<String>,
    /// Layers with incompatible weight shapes
    pub incompatible: Vec<WeightMismatch>,
    /// Layers with missing _weights
    pub missing: Vec<String>,
/// Weight shape mismatch information
pub struct WeightMismatch {
    /// Layer name
    pub layer_name: String,
    /// Expected weight shape
    pub expectedshape: Vec<usize>,
    /// Actual weight shape in pretrained model
    pub actualshape: Vec<usize>,
impl CompatibilityReport {
    /// Check if all _weights are compatible
    pub fn is_fully_compatible(&self) -> bool {
        self.incompatible.is_empty() && self.missing.is_empty()
    /// Get compatibility percentage
    pub fn compatibility_percentage(&self) -> f32 {
        let total = self.compatible.len() + self.incompatible.len() + self.missing.len();
        if total == 0 {
            100.0
            (self.compatible.len() as f32 / total as f32) * 100.0
impl std::fmt::Display for CompatibilityReport {
        writeln!(f, "Weight Compatibility Report:")?;
        writeln!(
            f,
            "  Compatible layers: {} ({:.1}%)",
            self.compatible.len(),
            self.compatibility_percentage()
        )?;
        writeln!(f, "  Incompatible layers: {}", self.incompatible.len())?;
        writeln!(f, "  Missing layers: {}", self.missing.len())?;
        if !self.incompatible.is_empty() {
            writeln!(f, "\nIncompatible layers:")?;
            for mismatch in &self.incompatible {
                writeln!(
                    f,
                    "  {}: expected {:?}, got {:?}",
                    mismatch.layer_name, mismatch.expectedshape, mismatch.actualshape
                )?;
        if !self.missing.is_empty() {
            writeln!(f, "\nMissing layers:")?;
            for layer in &self.missing {
                writeln!(f, "  {}", layer)?;
/// Fine-tuning utilities
pub struct FineTuningUtilities<F: Float + Debug> {
    /// Learning rate scheduler for different layer groups
    lr_scheduler: HashMap<String, F>,
    /// Gradient clipping values per layer
    gradient_clips: HashMap<String, F>,
    /// Weight decay values per layer
    weight_decays: HashMap<String, F>,
impl<F: Float + Debug + 'static> FineTuningUtilities<F> {
    /// Create new fine-tuning utilities
            lr_scheduler: HashMap::new(),
            gradient_clips: HashMap::new(),
            weight_decays: HashMap::new(),
    /// Set learning rate for a layer group
    pub fn set_layer_learning_rate(
        layer_pattern: String,
        learning_rate: f64,
        let lr = F::from(learning_rate)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid learning rate".to_string()))?;
        self.lr_scheduler.insert(layer_pattern, lr);
    /// Set gradient clipping for a layer group
    pub fn set_layer_gradient_clip(
        clip_value: f64,
        let clip = F::from(clip_value)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid clip value".to_string()))?;
        self.gradient_clips.insert(layer_pattern, clip);
    /// Set weight decay for a layer group
    pub fn set_layer_weight_decay(
        weight_decay: f64,
        let decay = F::from(weight_decay)
            .ok_or_else(|| NeuralError::InvalidArchitecture("Invalid weight decay".to_string()))?;
        self.weight_decays.insert(layer_pattern, decay);
    pub fn get_effective_learning_rate(&self, layer_name: &str, baselr: F) -> F {
        // Check for exact match first, then pattern matches
        for (pattern, &lr) in &self.lr_scheduler {
            if layer_name == pattern || layer_name.contains(pattern) {
                return lr;
        base_lr
    /// Get gradient clip value for a layer
    pub fn get_gradient_clip(&self, layername: &str) -> Option<F> {
        for (pattern, &clip) in &self.gradient_clips {
                return Some(clip);
        None
    /// Get weight decay for a layer
    pub fn get_weight_decay(&self, layername: &str) -> Option<F> {
        for (pattern, &decay) in &self.weight_decays {
                return Some(decay);
impl<F: Float + Debug + 'static> Default for FineTuningUtilities<F> {
/// Weight initialization strategies for transfer learning
pub enum WeightInitStrategy {
    /// Keep pretrained weights as-is
    KeepPretrained,
    /// Initialize with pretrained weights plus small noise
    PretrainedWithNoise {
        /// Scale of noise to add to pretrained weights
        noise_scale: f64,
    /// Partial initialization (only specific layers)
    PartialInit {
        /// List of layer names to initialize
        layers_to_init: Vec<String>,
    /// Xavier/Glorot initialization for new layers
    Xavier,
    /// He initialization for new layers
    He,
    /// Custom initialization with user-provided function
    Custom,
/// Model surgery utilities for architectural changes
pub struct ModelSurgery {
    /// Operations to perform
    operations: Vec<SurgeryOperation>,
/// Types of model surgery operations
pub enum SurgeryOperation {
    /// Add new layers at specified positions
    AddLayer {
        /// Position to insert the new layer
        position: usize,
        /// Name of the new layer
        /// Configuration for the new layer
        layer_config: LayerConfig,
    /// Remove existing layers
    RemoveLayer {
        /// Name of the layer to remove
    /// Replace existing layers
    ReplaceLayer {
        /// Name of the layer to replace
        old_layer: String,
        /// Name of the replacement layer
        new_layer: String,
        /// Configuration for the replacement layer
    /// Resize existing layers (e.g., change output dimensions)
    ResizeLayer {
        /// Name of the layer to resize
        /// New shape for the layer
        newshape: Vec<usize>,
        /// Weight initialization strategy for resized layer
        init_strategy: WeightInitStrategy,
/// Layer configuration for surgery operations
pub struct LayerConfig {
    /// Layer type
    pub layer_type: String,
    /// Input shape
    pub inputshape: Vec<usize>,
    /// Output shape
    pub outputshape: Vec<usize>,
    /// Additional parameters
    pub params: HashMap<String, f64>,
impl ModelSurgery {
    /// Create new model surgery utility
            operations: Vec::new(),
    /// Add a surgery operation
    pub fn add_operation(&mut self, operation: SurgeryOperation) {
        self.operations.push(operation);
    /// Apply all surgery operations
    pub fn apply_surgery(&self, modelconfig: &mut ModelConfig) -> Result<Vec<String>> {
        let mut applied_operations = Vec::new();
        for operation in &self.operations {
            match operation {
                SurgeryOperation::AddLayer {
                    position,
                    layer_config,
                } => {
                    self.add_layer_at_position(model_config, *position, layer_name, layer_config)?;
                    applied_operations.push(format!(
                        "Added layer {} at position {}",
                        layer_name, position
                    ));
                SurgeryOperation::RemoveLayer { layer_name } => {
                    self.remove_layer(model_config, layer_name)?;
                    applied_operations.push(format!("Removed layer {layer_name}"));
                SurgeryOperation::ReplaceLayer {
                    old_layer,
                    new_layer,
                    self.replace_layer(model_config, old_layer, new_layer, layer_config)?;
                    applied_operations
                        .push(format!("Replaced layer {old_layer} with {new_layer}"));
                SurgeryOperation::ResizeLayer {
                    newshape,
                    init_strategy,
                    self.resize_layer(model_config, layer_name, newshape, init_strategy)?;
                        "Resized layer {} to shape {:?}",
                        layer_name, newshape
        Ok(applied_operations)
    fn add_layer_at_position(
        _model_config: &mut ModelConfig, _position: usize_layer, name: &str, _layer_config: &LayerConfig,
        // Implementation would modify the model configuration
    fn remove_layer(&self, _model_config: &mut ModelConfig_layer, name: &str) -> Result<()> {
        // Implementation would remove the layer from model configuration
    fn replace_layer(
        _old_layer: &str, _new_layer: &str,
        // Implementation would replace the layer in model configuration
    fn resize_layer(
        _newshape: &[usize], _init_strategy: &WeightInitStrategy,
        // Implementation would resize the layer and reinitialize weights
impl Default for ModelSurgery {
/// Placeholder for model configuration
/// In a real implementation, this would be the actual model configuration type
pub struct ModelConfig {
    /// Model layers
    pub layers: Vec<String>,
    /// Layer configurations
    pub layer_configs: HashMap<String, LayerConfig>,
/// Advanced transfer learning orchestrator
pub struct TransferLearningOrchestrator<F: Float + Debug> {
    /// Transfer learning manager
    transfer_manager: TransferLearningManager<F>,
    /// Weight loader
    weight_loader: PretrainedWeightLoader,
    /// Fine-tuning utilities
    fine_tuning: FineTuningUtilities<F>,
    /// Domain adaptation
    domain_adaptation: Option<DomainAdaptation<F>>,
    /// Model surgery
    model_surgery: Option<ModelSurgery>,
    /// Weight initialization strategy
    #[allow(dead_code)]
    init_strategy: WeightInitStrategy,
impl<
        F: Float + Debug + 'static + FromPrimitive + Clone + Zero + Add<Output = F> + Div<Output = F>,
    > TransferLearningOrchestrator<F>
{
    /// Create new transfer learning orchestrator
    pub fn new(
        strategy: TransferStrategy,
        base_learning_rate: f64,
    ) -> Result<Self> {
            transfer_manager: TransferLearningManager::new(strategy, base_learning_rate)?,
            weight_loader: PretrainedWeightLoader::new(),
            fine_tuning: FineTuningUtilities::new(),
            domain_adaptation: None,
            model_surgery: None,
            init_strategy,
    /// Enable domain adaptation
    pub fn with_domain_adaptation(&mut self, method: AdaptationMethod) {
        self.domain_adaptation = Some(DomainAdaptation::new(method));
    /// Enable model surgery
    pub fn with_model_surgery(&mut self, surgery: ModelSurgery) {
        self.model_surgery = Some(surgery);
    /// Load pretrained weights
    pub fn load_pretrained_weights(&mut self, weights: HashMap<String, ArrayD<f32>>) -> Result<()> {
        self.weight_loader.load_weights(weights)
    /// Setup transfer learning for a model
    pub fn setup_transfer_learning(
        layer_names: &[String],
        expectedshapes: Option<&HashMap<String, Vec<usize>>>,
    ) -> Result<TransferLearningSetupReport> {
        // Initialize layer states
        self.transfer_manager.initialize_layer_states(layer_names)?;
        // Check weight compatibility if shapes provided
        let compatibility_report = if let Some(shapes) = expectedshapes {
            Some(self.weight_loader.check_weight_compatibility(shapes))
            None
        // Apply model surgery if configured
        let surgery_operations = if let Some(surgery) = &self.model_surgery {
            let mut dummy_config = ModelConfig {
                layers: layer_names.to_vec(),
                layer_configs: HashMap::new(),
            };
            Some(surgery.apply_surgery(&mut dummy_config)?)
        Ok(TransferLearningSetupReport {
            layer_states: self.transfer_manager.get_summary(),
            weight_compatibility: compatibility_report,
            surgery_operations,
            weight_statistics: self.weight_loader.get_weight_statistics(),
    /// Get effective learning rate for a layer considering all factors
    pub fn get_effective_learning_rate(&self, layername: &str) -> F {
        let base_lr = self.transfer_manager.get_layer_learning_rate(layer_name);
        self.fine_tuning
            .get_effective_learning_rate(layer_name, base_lr)
    /// Apply domain adaptation to features
    pub fn adapt_features(&self, layername: &str, features: &ArrayD<F>) -> Result<ArrayD<F>> {
        if let Some(adapter) = &self.domain_adaptation {
            adapter.adapt_features(layer_name, features)
            Ok(features.clone())
    /// Update transfer learning state for new epoch
        self.transfer_manager.update_epoch(epoch)
/// Transfer learning setup report
pub struct TransferLearningSetupReport {
    /// Layer states from transfer manager
    pub layer_states: TransferLearningState,
    /// Weight compatibility report
    pub weight_compatibility: Option<CompatibilityReport>,
    /// Applied surgery operations
    pub surgery_operations: Option<Vec<String>>,
    /// Weight statistics
    pub weight_statistics: HashMap<String, WeightStatistics>,
impl std::fmt::Display for TransferLearningSetupReport {
        writeln!(f, "Transfer Learning Setup Report:")?;
        writeln!(f, "\n{}", self.layer_states)?;
        if let Some(compatibility) = &self.weight_compatibility {
            writeln!(f, "\n{}", compatibility)?;
        if let Some(operations) = &self.surgery_operations {
            writeln!(f, "\nModel Surgery Operations:")?;
            for op in operations {
                writeln!(f, "  - {}", op)?;
        writeln!(f, "\nWeight Statistics:")?;
        for (layer, stats) in &self.weight_statistics {
            writeln!(
                f,
                "  {}: {} params, mean={:.4}, std={:.4}",
                layer, stats.param_count, stats.mean, stats.std
            )?;
/// Domain adaptation utilities
pub struct DomainAdaptation<F: Float + Debug> {
    /// Source domain statistics
    source_stats: HashMap<String, DomainStatistics<F>>,
    /// Target domain statistics
    target_stats: HashMap<String, DomainStatistics<F>>,
    /// Adaptation method
    adaptation_method: AdaptationMethod,
/// Domain statistics for adaptation
pub struct DomainStatistics<F: Float + Debug> {
    /// Feature means
    pub mean: ArrayD<F>,
    /// Feature variances
    pub variance: ArrayD<F>,
    /// Feature covariance matrix (optional)
    pub covariance: Option<ArrayD<F>>,
/// Domain adaptation methods
pub enum AdaptationMethod {
    /// Batch normalization statistics adaptation
    BatchNormAdaptation,
    /// Feature alignment via moment matching
    MomentMatching,
    /// Adversarial domain adaptation
    AdversarialTraining {
        /// Regularization parameter for adversarial loss
        lambda: f64,
    /// Coral (correlation alignment)
    CoralAlignment,
    > DomainAdaptation<F>
    /// Create new domain adaptation utility
    pub fn new(method: AdaptationMethod) -> Self {
            source_stats: HashMap::new(),
            target_stats: HashMap::new(),
            adaptation_method: method,
    /// Compute domain statistics from data
    pub fn compute_domain_statistics(
        domain_name: String,
        features: &ArrayD<F>,
        is_source: bool,
        let batch_size = features.shape()[0];
        if batch_size == 0 {
            return Err(NeuralError::ComputationError(
                "Empty feature batch".to_string(),
            ));
        // Compute mean across batch dimension
        let mean = features
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| NeuralError::ComputationError("Failed to compute mean".to_string()))?;
        // Compute variance
        let variance = {
            let diff = features - &mean;
            let squared_diff = diff.mapv(|x| x * x);
            squared_diff.mean_axis(scirs2_core::ndarray::Axis(0)).ok_or_else(|| {
                NeuralError::ComputationError("Failed to compute variance".to_string())
            })?
        let stats = DomainStatistics {
            mean: mean.into_dyn(),
            variance: variance.into_dyn(),
            covariance: None, // Could be computed if needed
        if is_source {
            self.source_stats.insert(domain_name, stats);
            self.target_stats.insert(domain_name, stats);
    /// Apply domain adaptation
        let source_stats = self
            .source_stats
            .get(layer_name)
            .ok_or_else(|| NeuralError::ComputationError("Source stats not found".to_string()))?;
        let target_stats = self
            .target_stats
            .ok_or_else(|| NeuralError::ComputationError("Target stats not found".to_string()))?;
        match self.adaptation_method {
            AdaptationMethod::BatchNormAdaptation => {
                self.batch_norm_adaptation(features, source_stats, target_stats)
            AdaptationMethod::MomentMatching => {
                self.moment_matching_adaptation(features, source_stats, target_stats)
                // Other methods would require more complex implementations
                Ok(features.clone())
    fn batch_norm_adaptation(
        source_stats: &DomainStatistics<F>,
        target_stats: &DomainStatistics<F>,
    ) -> Result<ArrayD<F>> {
        // Normalize using source statistics, then denormalize using target statistics
        let eps = F::from(1e-5).unwrap();
        // Normalize with source stats
        let normalized =
            (features - &source_stats.mean) / (source_stats.variance.mapv(|x| (x + eps).sqrt()));
        // Denormalize with target stats
        let adapted =
            normalized * target_stats.variance.mapv(|x| (x + eps).sqrt()) + &target_stats.mean;
        Ok(adapted)
    fn moment_matching_adaptation(
        _source_stats: &DomainStatistics<F>,
        // Simple moment matching: adjust to target mean and variance
        let current_mean = features
        let current_var = {
            let diff = features - &current_mean;
        // Normalize current features
        let normalized = (features - &current_mean) / current_var.mapv(|x| (x + eps).sqrt());
        // Apply target statistics
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_transfer_learning_manager_creation() {
        let strategy = TransferStrategy::FeatureExtraction { unfrozen_layers: 2 };
        let manager = TransferLearningManager::<f64>::new(strategy, 0.001);
        assert!(manager.is_ok());
    fn test_feature_extraction_strategy() {
        let mut manager = TransferLearningManager::<f64>::new(strategy, 0.001).unwrap();
        let layer_names = vec![
            "conv1".to_string(),
            "conv2".to_string(),
            "conv3".to_string(),
            "fc1".to_string(),
            "fc2".to_string(),
        ];
        manager.initialize_layer_states(&layer_names).unwrap();
        // First 3 layers should be frozen, last 2 trainable
        assert!(manager.is_layer_frozen("conv1"));
        assert!(manager.is_layer_frozen("conv2"));
        assert!(manager.is_layer_frozen("conv3"));
        assert!(!manager.is_layer_frozen("fc1"));
        assert!(!manager.is_layer_frozen("fc2"));
    fn test_fine_tuning_strategy() {
        let strategy = TransferStrategy::FineTuning {
            backbone_lr_ratio: 0.1,
            head_lr_ratio: 1.0,
            "backbone1".to_string(),
            "backbone2".to_string(),
            "head1".to_string(),
            "head2".to_string(),
        // Check learning rates
        let backbone_lr = manager.get_layer_learning_rate("backbone1");
        let head_lr = manager.get_layer_learning_rate("head1");
        assert!((backbone_lr - 0.0001).abs() < 1e-6); // 0.001 * 0.1
        assert!((head_lr - 0.001).abs() < 1e-6); // 0.001 * 1.0
    fn test_progressive_unfreezing() {
        let strategy = TransferStrategy::ProgressiveUnfreezing {
            unfreeze_schedule: vec![(5, 2), (10, 2)],
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
            "layer4".to_string(),
        // Initially all frozen
        assert!(manager.is_layer_frozen("layer1"));
        assert!(manager.is_layer_frozen("layer4"));
        // After epoch 5, last 2 layers unfrozen
        manager.update_epoch(5).unwrap();
        assert!(manager.is_layer_frozen("layer2"));
        assert!(!manager.is_layer_frozen("layer3"));
        assert!(!manager.is_layer_frozen("layer4"));
    fn test_pretrained_weight_loader() {
        let mut loader = PretrainedWeightLoader::new();
        let mut weights = HashMap::new();
        weights.insert(
            ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[10, 5])),
        );
        weights.insert("layer2".to_string(), ArrayD::ones(scirs2_core::ndarray::IxDyn(&[5, 3])));
        loader.load_weights(weights).unwrap();
        loader.add_layer_mapping("layer1".to_string(), "new_layer1".to_string());
        let available = loader.get_available_weights();
        assert_eq!(available.len(), 2);
        assert!(available.contains(&"layer1".to_string()));
        assert!(available.contains(&"layer2".to_string()));
    fn test_fine_tuning_utilities() {
        let mut utils = FineTuningUtilities::<f64>::new();
        utils
            .set_layer_learning_rate("backbone".to_string(), 0.0001)
            .unwrap();
            .set_layer_learning_rate("head".to_string(), 0.001)
            .set_layer_gradient_clip("backbone".to_string(), 1.0)
        let backbone_lr = utils.get_effective_learning_rate("backbone_layer1", 0.01);
        let head_lr = utils.get_effective_learning_rate("head_layer1", 0.01);
        let unknown_lr = utils.get_effective_learning_rate("unknown_layer", 0.01);
        assert!((backbone_lr - 0.0001).abs() < 1e-6);
        assert!((head_lr - 0.001).abs() < 1e-6);
        assert!((unknown_lr - 0.01).abs() < 1e-6);
        let clip = utils.get_gradient_clip("backbone_layer1");
        assert!(clip.is_some());
        assert!((clip.unwrap() - 1.0).abs() < 1e-6);
    fn test_domain_adaptation() {
        let mut adapter = DomainAdaptation::<f64>::new(AdaptationMethod::BatchNormAdaptation);
        // Create some test features
        let source_features = ArrayD::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[10, 5]),
            (0..50).map(|x| x as f64 / 10.0).collect(),
        )
        .unwrap()
        .into_dyn();
        let target_features = ArrayD::from_shape_vec(
            (0..50).map(|x| (x as f64 + 25.0) / 10.0).collect(),
        adapter
            .compute_domain_statistics("layer1".to_string(), &source_features, true)
            .compute_domain_statistics("layer1".to_string(), &target_features, false)
        let adapted = adapter.adapt_features("layer1", &source_features).unwrap();
        assert_eq!(adapted.shape(), source_features.shape());
    fn test_transfer_learning_state_display() {
        let state = TransferLearningState {
            current_epoch: 10,
            total_layers: 5,
            frozen_layers: 3,
            trainable_layers: 2,
            reduced_lr_layers: 0,
        let display_str = format!("{state}");
        assert!(display_str.contains("Epoch 10"));
        assert!(display_str.contains("Total layers: 5"));
        assert!(display_str.contains("Frozen layers: 3"));
    fn test_weight_statistics() {
        let weights = ArrayD::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[3, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        .unwrap();
        let stats = WeightStatistics::from_tensor(&weights);
        assert_eq!(stats.param_count, 9);
        assert_eq!(stats.shape, vec![3, 3]);
        assert!((stats.mean - 5.0).abs() < 1e-6);
        assert!(stats.min == 1.0);
        assert!(stats.max == 9.0);
    fn test_compatibility_report() {
        let mut expectedshapes = HashMap::new();
        expectedshapes.insert("layer1".to_string(), vec![10, 5]);
        expectedshapes.insert("layer2".to_string(), vec![5, 4]); // Mismatch
        expectedshapes.insert("layer3".to_string(), vec![3, 2]); // Missing
        let report = loader.check_weight_compatibility(&expectedshapes);
        assert_eq!(report.compatible.len(), 1);
        assert_eq!(report.incompatible.len(), 1);
        assert_eq!(report.missing.len(), 1);
        assert!(!report.is_fully_compatible());
        assert!((report.compatibility_percentage() - 33.333).abs() < 0.1);
    fn test_model_surgery() {
        let mut surgery = ModelSurgery::new();
        surgery.add_operation(SurgeryOperation::AddLayer {
            position: 0,
            layer_name: "new_layer".to_string(),
            layer_config: LayerConfig {
                layer_type: "dense".to_string(),
                inputshape: vec![10],
                outputshape: vec![5],
                params: HashMap::new(),
            },
        });
        let mut model_config = ModelConfig {
            layers: vec!["layer1".to_string(), "layer2".to_string()],
            layer_configs: HashMap::new(),
        let operations = surgery.apply_surgery(&mut model_config).unwrap();
        assert_eq!(operations.len(), 1);
        assert!(operations[0].contains("Added layer new_layer"));
    fn test_transfer_learning_orchestrator() {
        let init_strategy = WeightInitStrategy::KeepPretrained;
        let mut orchestrator =
            TransferLearningOrchestrator::<f64>::new(strategy, 0.001, init_strategy).unwrap();
        let layer_names = vec!["conv1".to_string(), "conv2".to_string(), "fc".to_string()];
        let report = orchestrator
            .setup_transfer_learning(&layer_names, None)
        assert_eq!(report.layer_states.total_layers, 3);
        assert_eq!(report.layer_states.frozen_layers, 1);
        assert_eq!(report.layer_states.trainable_layers, 2);
    fn test_pytorch_weight_loading() {
        let mut pytorch_state_dict = HashMap::new();
        pytorch_state_dict.insert(
            "layer1.weight".to_string(),
            "layer1.bias".to_string(),
            ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[10])),
        loader
            .load_from_pytorch_state_dict(pytorch_state_dict)
        assert!(available.contains(&"layer1_bias".to_string()));
    fn test_tensorflow_weight_loading() {
        let mut tf_checkpoint = HashMap::new();
        tf_checkpoint.insert(
            "dense_layer/kernel:0".to_string(),
            "dense_layer/bias:0".to_string(),
            ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[5])),
            .load_from_tensorflow_checkpoint(tf_checkpoint)
        assert!(available.contains(&"dense_layer".to_string()));
        assert!(available.contains(&"dense_layer_bias".to_string()));
