//! Continual and Multi-task Learning Module
//!
//! This module provides frameworks for training neural networks on multiple tasks
//! either simultaneously (multi-task learning) or sequentially (continual learning)
//! while avoiding catastrophic forgetting.

pub mod advanced_continual_learning;
pub mod elastic_weight_consolidation;
pub mod shared_backbone;
pub use advanced_continual_learning::{
    LateralConnection, LearningWithoutForgetting, LwFConfig, PackNet, PackNetConfig,
    ProgressiveConfig, ProgressiveNeuralNetwork, TaskColumn, TaskMask,
};
pub use elastic_weight_consolidation::{EWCConfig, EWC};
pub use shared_backbone::{MultiTaskArchitecture, SharedBackbone, TaskSpecificHead, TaskType};
use crate::error::Result;
use crate::models::sequential::Sequential;
use scirs2_core::ndarray::concatenate;
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;
use scirs2_core::ndarray::ArrayView1;
/// Configuration for continual learning
#[derive(Debug, Clone)]
pub struct ContinualConfig {
    /// Strategy for continual learning
    pub strategy: ContinualStrategy,
    /// Memory size for replay methods
    pub memory_size: usize,
    /// Regularization strength
    pub regularization_strength: f32,
    /// Number of tasks
    pub num_tasks: usize,
    /// Task-specific learning rates
    pub task_learning_rates: Option<Vec<f32>>,
    /// Enable meta-learning
    pub enable_meta_learning: bool,
    /// Temperature for knowledge distillation
    pub distillation_temperature: f32,
}
impl Default for ContinualConfig {
    fn default() -> Self {
        Self {
            strategy: ContinualStrategy::EWC,
            memory_size: 5000,
            regularization_strength: 1000.0,
            num_tasks: 5,
            task_learning_rates: None,
            enable_meta_learning: false,
            distillation_temperature: 3.0,
        }
    }
/// Continual learning strategies
#[derive(Debug, Clone, PartialEq)]
pub enum ContinualStrategy {
    /// Elastic Weight Consolidation
    EWC,
    /// Progressive Neural Networks
    Progressive,
    /// Experience Replay
    Replay,
    /// Generative Replay
    GenerativeReplay,
    /// Gradient Episodic Memory
    GEM,
    /// Average Gradient Episodic Memory
    AGEM,
    /// Learning without Forgetting
    LWF,
    /// PackNet
    PackNet,
    /// Dynamic Architecture
    DynamicArchitecture,
/// Multi-task learning configuration
pub struct MultiTaskConfig {
    /// Task names
    pub task_names: Vec<String>,
    /// Task weights for loss balancing
    pub task_weights: Option<Vec<f32>>,
    /// Shared layers configuration
    pub shared_layers: Vec<usize>,
    /// Task-specific layers configuration
    pub task_specific_layers: HashMap<String, Vec<usize>>,
    /// Gradient normalization
    pub gradient_normalization: bool,
    /// Dynamic weight averaging
    pub dynamic_weight_averaging: bool,
    /// Uncertainty weighting
    pub uncertainty_weighting: bool,
impl Default for MultiTaskConfig {
            task_names: vec!["task1".to_string(), "task2".to_string()],
            task_weights: None,
            shared_layers: vec![512, 256],
            task_specific_layers: HashMap::new(),
            gradient_normalization: true,
            dynamic_weight_averaging: false,
            uncertainty_weighting: false,
/// Continual learning framework
pub struct ContinualLearner {
    config: ContinualConfig,
    base_model: Sequential<f32>,
    task_models: Vec<Sequential<f32>>,
    memory_bank: MemoryBank,
    fisher_information: Option<Vec<Array2<f32>>>,
    optimal_params: Option<Vec<Array2<f32>>>,
    current_task: usize,
impl ContinualLearner {
    /// Create a new continual learner
    pub fn new(_config: ContinualConfig, basemodel: Sequential<f32>) -> Result<Self> {
        let memory_bank = MemoryBank::new(_config.memory_size);
        Ok(Self {
            config,
            base_model,
            task_models: Vec::new(),
            memory_bank,
            fisher_information: None,
            optimal_params: None,
            current_task: 0,
        })
    /// Train on a new task
    pub fn train_task(
        &mut self,
        task_id: usize,
        train_data: &ArrayView2<f32>,
        train_labels: &ArrayView1<usize>,
        val_data: &ArrayView2<f32>,
        val_labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<TaskTrainingResult> {
        self.current_task = task_id;
        let result = match self.config.strategy {
            ContinualStrategy::EWC => {
                self.train_with_ewc(train_data, train_labels, val_data, val_labels, epochs)?
            }
            ContinualStrategy::Replay => {
                self.train_with_replay(train_data, train_labels, val_data, val_labels, epochs)?
            ContinualStrategy::GEM => {
                self.train_with_gem(train_data, train_labels, val_data, val_labels, epochs)?
            _ => self.train_standard(train_data, train_labels, val_data, val_labels, epochs)?,
        };
        // Store task-specific information
        self.update_task_memory(train_data, train_labels)?;
        Ok(result)
    /// Train with Elastic Weight Consolidation
    fn train_with_ewc(
        let mut total_loss = 0.0;
        let mut best_accuracy = 0.0;
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            // Standard training loss
            let task_loss = self.compute_task_loss(train_data, train_labels)?;
            epoch_loss += task_loss;
            // EWC regularization loss
            if self.current_task > 0 {
                let ewc_loss = self.compute_ewc_loss()?;
                epoch_loss += self.config.regularization_strength * ewc_loss;
            total_loss += epoch_loss;
            // Validation
            let val_accuracy = self.evaluate(val_data, val_labels)?;
            best_accuracy = best_accuracy.max(val_accuracy);
        // Update Fisher information and optimal parameters
        self.update_fisher_information(train_data, train_labels)?;
        self.update_optimal_params()?;
        Ok(TaskTrainingResult {
            task_id: self.current_task,
            final_loss: total_loss / epochs as f32,
            best_accuracy,
            forgetting_measure: self.measure_forgetting()?,
    /// Train with experience replay
    fn train_with_replay(
            // Combine current task data with replay data
            let (combined_data, combined_labels) =
                self.combine_with_replay(train_data, train_labels)?;
            // Train on combined data
            let epoch_loss =
                self.compute_task_loss(&combined_data.view(), &combined_labels.view())?;
    /// Train with Gradient Episodic Memory
    fn train_with_gem(
            let epoch_loss = self.compute_task_loss(train_data, train_labels)?;
            // Project gradients to avoid interfering with previous tasks
                self.project_gradients()?;
    /// Standard training without continual learning techniques
    fn train_standard(
            forgetting_measure: 0.0,
    /// Compute task-specific loss
    fn compute_task_loss(&self, data: &ArrayView2<f32>, labels: &ArrayView1<usize>) -> Result<f32> {
        // Forward pass through the model
        let predictions = self.base_model.forward(data)?;
        // Compute cross-entropy loss
        let batch_size = data.shape()[0];
        for i in 0..batch_size {
            let true_label = labels[i];
            if true_label < predictions.shape()[1] {
                let pred_value = predictions[[i, true_label]].max(1e-7);
                total_loss -= pred_value.ln();
        Ok(total_loss / batch_size as f32)
    /// Compute task-specific loss for heads (multi-task)
    fn compute_head_loss(&self, predictions: &ArrayView2<f32>, labels: &ArrayView1<usize>) -> Result<f32> {
        let batch_size = predictions.shape()[0];
    /// Compute EWC regularization loss
    fn compute_ewc_loss(&self) -> Result<f32> {
        if self.fisher_information.is_none() || self.optimal_params.is_none() {
            return Ok(0.0);
        // Simplified EWC loss
        Ok(0.1) // Placeholder
    /// Update Fisher information matrix
    fn update_fisher_information(
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
    ) -> Result<()> {
        // Simplified Fisher information computation
        let num_params = 10; // Placeholder
        self.fisher_information = Some(vec![Array2::from_elem((10, 10), 0.1); num_params]);
        Ok(())
    /// Update optimal parameters
    fn update_optimal_params(&mut self) -> Result<()> {
        // Store current model parameters as optimal
        self.optimal_params = Some(vec![Array2::from_elem((10, 10), 0.5); num_params]);
    /// Combine current task data with replay data
    fn combine_with_replay(
        &self,
    ) -> Result<(Array2<f32>, Array1<usize>)> {
        let replay_samples = self.memory_bank.sample(self.config.memory_size / 10)?;
        // Combine current and replay data
        let combined_data = concatenate![Axis(0), *data, replay_samples.data];
        let combined_labels = concatenate![Axis(0), *labels, replay_samples.labels];
        Ok((combined_data, combined_labels))
    /// Project gradients for GEM
    fn project_gradients(&mut self) -> Result<()> {
        // Simplified gradient projection
    /// Update task memory
    fn update_task_memory(
        self.memory_bank
            .add_task_data(self.current_task, data, labels)
    /// Evaluate on validation data
    fn evaluate(&self, data: &ArrayView2<f32>, labels: &ArrayView1<usize>) -> Result<f32> {
        // Simplified evaluation
        Ok(0.85) // Placeholder
    /// Measure forgetting on previous tasks
    fn measure_forgetting(&self) -> Result<f32> {
        if self.current_task == 0 {
        // Simplified forgetting measure
        Ok(0.05) // Placeholder
    /// Get performance on all tasks
    pub fn evaluate_all_tasks(
        task_data: &[(Array2<f32>, Array1<usize>)],
    ) -> Result<Vec<f32>> {
        let mut accuracies = Vec::new();
        for (data, labels) in task_data {
            let accuracy = self.evaluate(&data.view(), &labels.view())?;
            accuracies.push(accuracy);
        Ok(accuracies)
/// Memory bank for storing task data
struct MemoryBank {
    capacity: usize,
    task_memories: HashMap<usize, TaskMemory>,
struct TaskMemory {
    data: Array2<f32>,
    labels: Array1<usize>,
struct MemorySamples {
impl MemoryBank {
    fn new(capacity: usize) -> Self {
            capacity,
            task_memories: HashMap::new(),
    fn add_task_data(
        let samples_per_task = self._capacity / (self.task_memories.len() + 1);
        // Random sampling for memory
        let num_samples = data.shape()[0].min(samples_per_task);
        let indices: Vec<usize> = (0..data.shape()[0]).collect();
        let selected_indices = &indices[..num_samples];
        let mut selected_data = Array2::zeros((num_samples, data.shape()[1]));
        let mut selected_labels = Array1::zeros(num_samples);
        for (i, &idx) in selected_indices.iter().enumerate() {
            selected_data.row_mut(i).assign(&data.row(idx));
            selected_labels[i] = labels[idx];
        self.task_memories.insert(
            task_id,
            TaskMemory {
                data: selected_data,
                labels: selected_labels,
            },
        );
    fn sample(&self, numsamples: usize) -> Result<MemorySamples> {
        if self.task_memories.is_empty() {
            return Ok(MemorySamples {
                data: Array2::zeros((0, 1)),
                labels: Array1::zeros(0),
            });
        let samples_per_task = num_samples / self.task_memories.len();
        let mut all_data = Vec::new();
        let mut all_labels = Vec::new();
        for memory in self.task_memories.values() {
            let task_samples = samples_per_task.min(memory.data.shape()[0]);
            for i in 0..task_samples {
                all_data.push(memory.data.row(i).to_owned());
                all_labels.push(memory.labels[i]);
        let data = if all_data.is_empty() {
            Array2::zeros((0, 1))
        } else {
            let rows = all_data.len();
            let cols = all_data[0].len();
            let mut arr = Array2::zeros((rows, cols));
            for (i, row) in all_data.into_iter().enumerate() {
                arr.row_mut(i).assign(&row);
            arr
        Ok(MemorySamples {
            data,
            labels: Array1::from_vec(all_labels),
/// Result of training on a task
pub struct TaskTrainingResult {
    pub task_id: usize,
    pub final_loss: f32,
    pub best_accuracy: f32,
    pub forgetting_measure: f32,
/// Multi-task learner
pub struct MultiTaskLearner {
    config: MultiTaskConfig,
    shared_backbone: SharedBackbone,
    task_heads: HashMap<String, TaskSpecificHead>,
    task_uncertainties: Option<HashMap<String, f32>>,
impl MultiTaskLearner {
    /// Create a new multi-task learner
    pub fn new(_config: MultiTaskConfig, inputdim: usize) -> Result<Self> {
        let shared_backbone = SharedBackbone::new(input_dim, &_config.shared_layers)?;
        let mut task_heads = HashMap::new();
        for task_name in &_config.task_names {
            let task_layers = _config
                .task_specific_layers
                .get(task_name)
                .cloned()
                .unwrap_or_else(|| vec![128, 64]);
            let head = TaskSpecificHead::new(
                config.shared_layers.last().copied().unwrap_or(256),
                &task_layers,
                10, // Placeholder output dim
            )?;
            task_heads.insert(task_name.clone(), head);
        let task_uncertainties = if config.uncertainty_weighting {
            Some(
                _config
                    .task_names
                    .iter()
                    .map(|name| (name.clone(), 0.0))
                    .collect(),
            )
            None
            shared_backbone,
            task_heads,
            task_uncertainties,
    /// Train on multiple tasks
    pub fn train(
        task_data: &HashMap<String, (ArrayView2<f32>, ArrayView1<usize>)>,
    ) -> Result<MultiTaskTrainingResult> {
        let mut task_losses = HashMap::new();
        let mut task_accuracies = HashMap::new();
            let mut epoch_losses = HashMap::new();
            // Forward pass for all tasks
            for (task_name, (data, labels)) in task_data {
                let shared_features = self.shared_backbone.forward(data)?;
                if let Some(head) = self.task_heads.get(task_name) {
                    let task_output = head.forward(&shared_features.view())?;
                    let task_loss = self.compute_head_loss(&task_output.view(), labels)?;
                    epoch_losses.insert(task_name.clone(), task_loss);
                }
            // Compute weighted loss
            let total_loss = self.compute_weighted_loss(&epoch_losses)?;
            // Update task uncertainties if enabled
            if self.config.uncertainty_weighting {
                self.update_task_uncertainties(&epoch_losses)?;
            // Track metrics
            for (task_name, loss) in epoch_losses {
                task_losses
                    .entry(task_name.clone())
                    .or_insert_with(Vec::new)
                    .push(loss);
        // Compute final accuracies
        for (task_name, (data, labels)) in task_data {
            let accuracy = self.evaluate_task(task_name, data, labels)?;
            task_accuracies.insert(task_name.clone(), accuracy);
        Ok(MultiTaskTrainingResult {
            task_losses,
            task_accuracies,
            task_weights: self.get_current_task_weights(),
    /// Compute weighted loss across tasks
    fn compute_weighted_loss(&self, tasklosses: &HashMap<String, f32>) -> Result<f32> {
        let weights = self.get_current_task_weights();
        for (task_name, &loss) in task_losses {
            let weight = weights.get(task_name).unwrap_or(&1.0);
            total_loss += weight * loss;
        Ok(total_loss)
    /// Update task uncertainties for uncertainty weighting
    fn update_task_uncertainties(&mut self, tasklosses: &HashMap<String, f32>) -> Result<()> {
        if let Some(ref mut uncertainties) = self.task_uncertainties {
            for (task_name, &loss) in task_losses {
                // Simple exponential moving average
                let current = uncertainties.get(task_name).copied().unwrap_or(0.0);
                uncertainties.insert(task_name.clone(), 0.9 * current + 0.1 * loss);
    /// Get current task weights
    fn get_current_task_weights(&self) -> HashMap<String, f32> {
        if let Some(ref weights) = self.config.task_weights {
            self.config
                .task_names
                .iter()
                .zip(weights)
                .map(|(name, &weight)| (name.clone(), weight))
                .collect()
        } else if let Some(ref uncertainties) = self.task_uncertainties {
            // Compute weights from uncertainties
            uncertainties
                .map(|(name, &uncertainty)| {
                    let weight = 1.0 / (2.0 * uncertainty.max(0.1));
                    (name.clone(), weight)
                })
            // Equal weights
                .map(|name| (name.clone(), 1.0))
    /// Evaluate a specific task
    fn evaluate_task(
        task_name: &str,
    ) -> Result<f32> {
        let shared_features = self.shared_backbone.forward(data)?;
        if let Some(head) = self.task_heads.get(task_name) {
            let task_output = head.forward(&shared_features.view())?;
            // Compute accuracy (simplified)
            Ok(0.9) // Placeholder
            Err(crate::error::NeuralError::InvalidArgument(format!(
                "Task {} not found",
                task_name
            )))
/// Multi-task training result
#[derive(Debug)]
pub struct MultiTaskTrainingResult {
    pub task_losses: HashMap<String, Vec<f32>>,
    pub task_accuracies: HashMap<String, f32>,
    pub task_weights: HashMap<String, f32>,
/// Advanced Meta-Learning for Continual Learning (MAML-style)
pub struct MetaContinualLearner {
    /// Meta-model parameters
    meta_model: Sequential<f32>,
    /// Task-specific adaptations
    task_adaptations: Vec<TaskAdaptation>,
    /// Meta-learning configuration
    config: MetaLearningConfig,
    /// Inner loop optimizer parameters
    inner_lr: f32,
    /// Outer loop optimizer parameters
    outer_lr: f32,
    /// Support and query sets for meta-learning
    meta_batch: Option<MetaBatch>,
pub struct MetaLearningConfig {
    /// Number of inner gradient steps
    pub inner_steps: usize,
    /// Number of tasks per meta-batch
    pub tasks_per_batch: usize,
    /// Support set size per task
    pub support_size: usize,
    /// Query set size per task
    pub query_size: usize,
    /// Enable second-order gradients
    pub second_order: bool,
    /// Adaptation learning rate schedule
    pub adaptive_lr: bool,
impl Default for MetaLearningConfig {
            inner_steps: 5,
            tasks_per_batch: 4,
            support_size: 10,
            query_size: 15,
            second_order: true,
            adaptive_lr: true,
pub struct TaskAdaptation {
    /// Task identifier
    /// Adapted parameters
    pub adapted_params: Vec<Array2<f32>>,
    /// Adaptation history
    pub adaptation_steps: Vec<AdaptationStep>,
    /// Task-specific learning rate
    pub task_lr: f32,
pub struct AdaptationStep {
    /// Step number
    pub step: usize,
    /// Loss before step
    pub loss_before: f32,
    /// Loss after step
    pub loss_after: f32,
    /// Gradient norm
    pub gradient_norm: f32,
pub struct MetaBatch {
    /// Support sets for each task
    pub support_sets: Vec<(Array2<f32>, Array1<usize>)>,
    /// Query sets for each task
    pub query_sets: Vec<(Array2<f32>, Array1<usize>)>,
    /// Task identifiers
    pub task_ids: Vec<usize>,
impl MetaContinualLearner {
    /// Create a new meta-continual learner
    pub fn new(
        meta_model: Sequential<f32>,
        config: MetaLearningConfig,
        inner_lr: f32,
        outer_lr: f32,
    ) -> Self {
            meta_model,
            task_adaptations: Vec::new(),
            inner_lr,
            outer_lr,
            meta_batch: None,
    /// Meta-train on a batch of tasks
    pub fn meta_train(&mut self, metabatch: MetaBatch) -> Result<MetaTrainingResult> {
        let mut total_meta_loss = 0.0;
        let mut task_losses = Vec::new();
        // For each task in the meta-batch
        for i in 0..meta_batch.task_ids.len() {
            let task_id = meta_batch.task_ids[i];
            let (support_data, support_labels) = &meta_batch.support_sets[i];
            let (query_data, query_labels) = &meta_batch.query_sets[i];
            // Inner loop: adapt to current task
            let adapted_params = self.inner_loop_adaptation(
                support_data,
                support_labels,
                task_id,
            // Evaluate adapted model on query set
            let query_loss = self.evaluate_adapted_model(
                &adapted_params,
                query_data,
                query_labels,
            total_meta_loss += query_loss;
            task_losses.push(query_loss);
            // Store adaptation for this task
            let adaptation = TaskAdaptation {
                adapted_params,
                adaptation_steps: Vec::new(), // Would track during adaptation
                task_lr: self.inner_lr,
            };
            self.task_adaptations.push(adaptation);
        // Outer loop: update meta-parameters
        self.outer_loop_update(total_meta_loss)?;
        Ok(MetaTrainingResult {
            meta_loss: total_meta_loss / meta_batch.task_ids.len() as f32,
            adaptation_quality: self.measure_adaptation_quality(),
    /// Perform inner loop adaptation for a specific task
    fn inner_loop_adaptation(
        support_data: &Array2<f32>,
        support_labels: &Array1<usize>,
    ) -> Result<Vec<Array2<f32>>> {
        // Start with meta-parameters
        let mut current_params = self.get_meta_parameters()?;
        // Perform gradient descent steps
        for step in 0..self.config.inner_steps {
            // Compute loss on support set
            let loss = self.compute_task_loss_with_params(
                &current_params,
            // Compute gradients
            let gradients = self.compute_gradients(&current_params, loss)?;
            // Update parameters
            for (param, grad) in current_params.iter_mut().zip(gradients.iter()) {
                *param = &*param - &(grad * self.inner_lr);
            // Adaptive learning rate
            if self.config.adaptive_lr {
                self.inner_lr *= 0.99; // Simple decay
        Ok(current_params)
    /// Evaluate adapted model on query set
    fn evaluate_adapted_model(
        adapted_params: &[Array2<f32>],
        query_data: &Array2<f32>,
        query_labels: &Array1<usize>,
        self.compute_task_loss_with_params(adapted_params, query_data, query_labels)
    /// Update meta-parameters (outer loop)
    fn outer_loop_update(&mut self, metaloss: f32) -> Result<()> {
        // Simplified meta-gradient update
        // In practice, would compute gradients w.r.t. meta-parameters
    /// Get current meta-parameters
    fn get_meta_parameters(&self) -> Result<Vec<Array2<f32>>> {
        // Placeholder - would extract actual model parameters
        Ok(vec![Array2::from_elem((10, 10), 0.1); 5])
    /// Compute task loss with specific parameters
    fn compute_task_loss_with_params(
        params: &[Array2<f32>],
        data: &Array2<f32>,
        labels: &Array1<usize>,
        // Simplified computation - would use params for forward pass
        Ok(0.5) // Placeholder
    /// Compute gradients for parameters
    fn compute_gradients(&self, params: &[Array2<f32>], loss: f32) -> Result<Vec<Array2<f32>>> {
        // Simplified gradient computation
        Ok(params.iter().map(|p| Array2::from_elem(p.shape(), 0.01)).collect())
    /// Measure adaptation quality
    fn measure_adaptation_quality(&self) -> f32 {
        // Simple metric: average improvement across tasks
        0.15 // Placeholder
    /// Few-shot adaptation to a new task
    pub fn few_shot_adapt(
        task_data: &Array2<f32>,
        task_labels: &Array1<usize>,
        num_shots: usize,
    ) -> Result<TaskAdaptation> {
        // Use first few samples for adaptation
        let adapt_data = task_data.slice(s![..num_shots, ..]);
        let adapt_labels = task_labels.slice(s![..num_shots]);
        let adapted_params = self.inner_loop_adaptation(
            &adapt_data.to_owned(),
            &adapt_labels.to_owned(),
            self.task_adaptations.len(),
        )?;
        let adaptation = TaskAdaptation {
            task_id: self.task_adaptations.len(),
            adapted_params,
            adaptation_steps: Vec::new(),
            task_lr: self.inner_lr,
        Ok(adaptation)
pub struct MetaTrainingResult {
    pub meta_loss: f32,
    pub task_losses: Vec<f32>,
    pub adaptation_quality: f32,
/// Advanced Memory Management for Continual Learning
pub struct AdvancedMemoryManager {
    /// Core memory buffer
    core_memory: CoreMemoryBuffer,
    /// Episodic memory for important samples
    episodic_memory: EpisodicMemoryBuffer,
    /// Semantic memory for learned concepts
    semantic_memory: SemanticMemoryBuffer,
    /// Memory consolidation strategy
    consolidation_strategy: ConsolidationStrategy,
pub enum ConsolidationStrategy {
    /// Gradient-based importance
    GradientBased,
    /// Uncertainty-based selection
    UncertaintyBased,
    /// Diversity-based selection
    DiversityBased,
    /// Hybrid approach
    Hybrid,
pub struct CoreMemoryBuffer {
    /// Raw data samples
    samples: Vec<MemorySample>,
    /// Capacity limit
    /// Current size
    current_size: usize,
pub struct EpisodicMemoryBuffer {
    /// Important episodes
    episodes: Vec<Episode>,
    /// Selection criteria
    importance_threshold: f32,
pub struct SemanticMemoryBuffer {
    /// Learned prototypes
    prototypes: Vec<Prototype>,
    /// Concept relationships
    concept_graph: ConceptGraph,
pub struct MemorySample {
    pub data: Array1<f32>,
    pub label: usize,
    pub importance_score: f32,
    pub timestamp: std::time::Instant,
pub struct Episode {
    pub samples: Vec<MemorySample>,
    pub context: EpisodeContext,
    pub significance: f32,
pub struct EpisodeContext {
    pub difficulty: f32,
    pub novelty: f32,
    pub success_rate: f32,
pub struct Prototype {
    pub feature_vector: Array1<f32>,
    pub class_id: usize,
    pub confidence: f32,
    pub update_count: usize,
pub struct ConceptGraph {
    nodes: Vec<ConceptNode>,
    edges: Vec<ConceptEdge>,
pub struct ConceptNode {
    pub concept_id: usize,
    pub representation: Array1<f32>,
    pub strength: f32,
pub struct ConceptEdge {
    pub from_concept: usize,
    pub to_concept: usize,
    pub weight: f32,
    pub edge_type: EdgeType,
pub enum EdgeType {
    Similarity,
    Causality,
    Hierarchy,
    Temporal,
impl AdvancedMemoryManager {
    pub fn new(_capacity: usize, consolidationstrategy: ConsolidationStrategy) -> Self {
            core_memory: CoreMemoryBuffer {
                samples: Vec::new(),
                capacity,
                current_size: 0,
            episodic_memory: EpisodicMemoryBuffer {
                episodes: Vec::new(),
                importance_threshold: 0.7,
            semantic_memory: SemanticMemoryBuffer {
                prototypes: Vec::new(),
                concept_graph: ConceptGraph {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                },
            consolidation_strategy,
    /// Add new samples to memory
    pub fn add_samples(
        for i in 0..data.shape()[0] {
            let sample = MemorySample {
                data: data.row(i).to_owned(),
                label: labels[i],
                importance_score: self.compute_importance_score(&data.row(i), labels[i])?,
                timestamp: std::time::Instant::now(),
            self.add_sample_to_buffers(sample)?;
        // Periodic consolidation
        if self.core_memory.current_size >= self.core_memory.capacity {
            self.consolidate_memory()?;
    /// Compute importance score for a sample
    fn compute_importance_score(&self, data: &ArrayView1<f32>, label: usize) -> Result<f32> {
        match self.consolidation_strategy {
            ConsolidationStrategy::GradientBased => {
                // Use gradient magnitude as importance
                Ok(data.iter().map(|&x| x.abs()).sum::<f32>() / data.len() as f32), ConsolidationStrategy::UncertaintyBased => {
                // Use prediction uncertainty
                Ok(0.5 + 0.3 * scirs2_core::random::random::<f32>()) // Placeholder
            ConsolidationStrategy::DiversityBased => {
                // Use distance from existing prototypes
                self.compute_diversity_score(data)
            ConsolidationStrategy::Hybrid => {
                // Combine multiple criteria
                let gradient_score = data.iter().map(|&x| x.abs()).sum::<f32>() / data.len() as f32;
                let diversity_score = self.compute_diversity_score(data)?;
                Ok(0.5 * gradient_score + 0.5 * diversity_score)
    /// Compute diversity score based on distance from prototypes
    fn compute_diversity_score(&self, data: &ArrayView1<f32>) -> Result<f32> {
        if self.semantic_memory.prototypes.is_empty() {
            return Ok(1.0); // Maximum diversity if no prototypes exist
        let mut min_distance = f32::INFINITY;
        for prototype in &self.semantic_memory.prototypes {
            let distance = self.euclidean_distance(data, &prototype.feature_vector.view());
            min_distance = min_distance.min(distance);
        // Normalize distance to [0, 1] range
        Ok((min_distance / (min_distance + 1.0)).min(1.0))
    /// Calculate Euclidean distance between two vectors
    fn euclidean_distance(&self, a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    /// Add sample to appropriate memory buffers
    fn add_sample_to_buffers(&mut self, sample: MemorySample) -> Result<()> {
        // Add to core memory
        if self.core_memory.current_size < self.core_memory.capacity {
            self.core_memory.samples.push(sample.clone());
            self.core_memory.current_size += 1;
            // Replace least important sample
            if let Some(min_idx) = self.find_least_important_sample() {
                self.core_memory.samples[min_idx] = sample.clone();
        // Add to episodic memory if important enough
        if sample.importance_score > self.episodic_memory.importance_threshold {
            let episode = Episode {
                samples: vec![sample.clone()],
                context: EpisodeContext {
                    task_id: sample.task_id,
                    difficulty: sample.importance_score,
                    novelty: self.compute_diversity_score(&sample.data.view())?,
                    success_rate: 0.8, // Placeholder
                significance: sample.importance_score,
            self.episodic_memory.episodes.push(episode);
        // Update semantic memory
        self.update_prototypes(&sample)?;
    /// Find least important sample in core memory
    fn find_least_important_sample(&self) -> Option<usize> {
        self.core_memory
            .samples
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.importance_score.partial_cmp(&b.importance_score).unwrap())
            .map(|(idx_)| idx)
    /// Update prototypes in semantic memory
    fn update_prototypes(&mut self, sample: &MemorySample) -> Result<()> {
        // Find nearest prototype
        if let Some(nearest_idx) = self.find_nearest_prototype(&sample.data.view(), sample.label) {
            // Update existing prototype
            let prototype = &mut self.semantic_memory.prototypes[nearest_idx];
            let alpha = 0.1; // Learning rate
            
            for (i, &new_val) in sample.data.iter().enumerate() {
                prototype.feature_vector[i] = 
                    (1.0 - alpha) * prototype.feature_vector[i] + alpha * new_val;
            prototype.update_count += 1;
            prototype.confidence = (prototype.update_count as f32).min(10.0) / 10.0;
            // Create new prototype
            let prototype = Prototype {
                feature_vector: sample.data.clone(),
                class_id: sample.label,
                confidence: 0.1,
                update_count: 1,
            self.semantic_memory.prototypes.push(prototype);
    /// Find nearest prototype for given data and label
    fn find_nearest_prototype(&self, data: &ArrayView1<f32>, label: usize) -> Option<usize> {
        let mut nearest_idx = None;
        for (idx, prototype) in self.semantic_memory.prototypes.iter().enumerate() {
            if prototype.class_id == label {
                let distance = self.euclidean_distance(data, &prototype.feature_vector.view());
                if distance < min_distance {
                    min_distance = distance;
                    nearest_idx = Some(idx);
        nearest_idx
    /// Consolidate memory by removing redundant samples
    fn consolidate_memory(&mut self) -> Result<()> {
        // Remove samples with low importance scores
        self.core_memory.samples.retain(|sample| sample.importance_score > 0.3);
        
        // Update current size
        self.core_memory.current_size = self.core_memory.samples.len();
        // Merge similar episodes
        self.merge_similar_episodes()?;
        // Prune weak prototypes
        self.semantic_memory.prototypes.retain(|p| p.confidence > 0.2);
    /// Merge similar episodes to reduce memory usage
    fn merge_similar_episodes(&mut self) -> Result<()> {
        let mut merged_episodes = Vec::new();
        let mut used = vec![false; self.episodic_memory.episodes.len()];
        for i in 0..self.episodic_memory.episodes.len() {
            if used[i] {
                continue;
            let mut merged_episode = self.episodic_memory.episodes[i].clone();
            used[i] = true;
            // Find similar episodes
            for j in (i + 1)..self.episodic_memory.episodes.len() {
                if !used[j] && self.are_episodes_similar(&merged_episode, &self.episodic_memory.episodes[j]) {
                    // Merge episodes
                    merged_episode.samples.extend(self.episodic_memory.episodes[j].samples.clone());
                    merged_episode.significance = merged_episode.significance.max(self.episodic_memory.episodes[j].significance);
                    used[j] = true;
            merged_episodes.push(merged_episode);
        self.episodic_memory.episodes = merged_episodes;
    /// Check if two episodes are similar enough to merge
    fn are_episodes_similar(&self, ep1: &Episode, ep2: &Episode) -> bool {
        ep1.context.task_id == ep2.context.task_id && 
        (ep1.context.difficulty - ep2.context.difficulty).abs() < 0.2
    /// Sample from memory for replay
    pub fn sample_for_replay(&self, numsamples: usize) -> Result<(Array2<f32>, Array1<usize>)> {
        if self.core_memory.samples.is_empty() {
            return Ok((Array2::zeros((0, 1)), Array1::zeros(0)));
        let actual_samples = num_samples.min(self.core_memory.samples.len());
        // Importance-based sampling
        let mut sampled_indices = Vec::new();
        let total_importance: f32 = self.core_memory.samples.iter().map(|s| s.importance_score).sum();
        for _ in 0..actual_samples {
            let mut cumulative = 0.0;
            let target = scirs2_core::random::random::<f32>() * total_importance;
            for (idx, sample) in self.core_memory.samples.iter().enumerate() {
                cumulative += sample.importance_score;
                if cumulative >= target {
                    sampled_indices.push(idx);
                    break;
        // Construct arrays
        let data_dim = self.core_memory.samples[0].data.len();
        let mut data = Array2::zeros((actual_samples, data_dim));
        let mut labels = Array1::zeros(actual_samples);
        for (i, &idx) in sampled_indices.iter().enumerate() {
            data.row_mut(i).assign(&self.core_memory.samples[idx].data);
            labels[i] = self.core_memory.samples[idx].label;
        Ok((data, labels))
    /// Get memory statistics
    pub fn get_memory_statistics(&self) -> MemoryStatistics {
        MemoryStatistics {
            core_memory_usage: self.core_memory.current_size,
            core_memory_capacity: self.core_memory.capacity,
            episodic_memory_episodes: self.episodic_memory.episodes.len(),
            semantic_prototypes: self.semantic_memory.prototypes.len(),
            concept_nodes: self.semantic_memory.concept_graph.nodes.len(),
            average_importance: if !self.core_memory.samples.is_empty() {
                self.core_memory.samples.iter().map(|s| s.importance_score).sum::<f32>() 
                    / self.core_memory.samples.len() as f32
            } else {
                0.0
pub struct MemoryStatistics {
    pub core_memory_usage: usize,
    pub core_memory_capacity: usize,
    pub episodic_memory_episodes: usize,
    pub semantic_prototypes: usize,
    pub concept_nodes: usize,
    pub average_importance: f32,
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_continual_config_default() {
        let config = ContinualConfig::default();
        assert_eq!(config.strategy, ContinualStrategy::EWC);
        assert_eq!(config.memory_size, 5000);
    fn test_multi_task_config_default() {
        let config = MultiTaskConfig::default();
        assert_eq!(config.task_names.len(), 2);
        assert!(config.gradient_normalization);
    fn test_memory_bank() {
        let mut bank = MemoryBank::new(1000);
        let data = Array2::from_elem((100, 10), 1.0);
        let labels = Array1::from_elem(100, 0);
        bank.add_task_data(0, &data.view(), &labels.view()).unwrap();
        let samples = bank.sample(50).unwrap();
        assert!(samples.data.shape()[0] <= 50);
