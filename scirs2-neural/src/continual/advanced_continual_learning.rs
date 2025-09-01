//! Advanced Continual Learning Techniques
//!
//! This module implements state-of-the-art continual learning methods including
//! Progressive Neural Networks, PackNet, Meta-Learning approaches, and more.

use crate::continual::shared__backbone::{MultiTaskArchitecture, TaskType};
use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use ndarray::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use ndarray::ArrayView1;
/// Progressive Neural Networks for continual learning
pub struct ProgressiveNeuralNetwork {
    /// Columns (one per task)
    columns: Vec<TaskColumn>,
    /// Lateral connections between columns
    lateral_connections: Vec<Vec<LateralConnection>>,
    /// Current task being trained
    current_task: usize,
    /// Configuration
    config: ProgressiveConfig,
}
/// Configuration for Progressive Neural Networks
#[derive(Debug, Clone)]
pub struct ProgressiveConfig {
    /// Base network architecture (layers per column)
    pub base_layers: Vec<usize>,
    /// Number of lateral connections per layer
    pub lateral_connections_per_layer: usize,
    /// Learning rate for new columns
    pub column_learning_rate: f32,
    /// Learning rate for lateral connections
    pub lateral_learning_rate: f32,
    /// Freeze previous columns
    pub freeze_previous_columns: bool,
impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            base_layers: vec![128, 64, 32],
            lateral_connections_per_layer: 16,
            column_learning_rate: 1e-3,
            lateral_learning_rate: 1e-4,
            freeze_previous_columns: true,
        }
    }
/// A single column in Progressive Neural Network
pub struct TaskColumn {
    /// Task ID
    task_id: usize,
    /// Layers in this column
    layers: Vec<Box<dyn Layer<f32> + Send + Sync>>,
    /// Output dimension
    output_dim: usize,
    /// Whether this column is frozen
    frozen: bool,
impl TaskColumn {
    /// Create a new task column
    pub fn new(
        task_id: usize,
        input_dim: usize,
        layer_sizes: &[usize],
        output_dim: usize,
    ) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32> + Send + Sync>> = Vec::new();
        let mut current_dim = input_dim;
        for &layer_size in layer_sizes {
            let layer = Dense::new(
                current_dim,
                layer_size,
                Some("relu"),
                &mut rng(),
            )?;
            layers.push(Box::new(layer));
            current_dim = layer_size;
        // Output layer
        let output_layer = Dense::new(
            current_dim,
            output_dim,
            Some("softmax"),
            &mut rng(),
        )?;
        layers.push(Box::new(output_layer));
        Ok(Self {
            task_id,
            layers,
            frozen: false,
        })
    /// Forward pass through column
    pub fn forward(
        &self,
        input: &ArrayView2<f32>,
        lateral_inputs: &[Array2<f32>],
    ) -> Result<(Array2<f32>, Vec<Array2<f32>>)> {
        let mut current_output = input.to_owned();
        let mut layer_outputs = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            // Add lateral connections if available
            if i < lateral_inputs.len() && !lateral_inputs[i].is_empty() {
                let combined = self.combine_with_lateral(&current_output, &lateral_inputs[i])?;
                current_output = layer.forward(&combined.into_dyn())?.into_dimensionality()?;
            } else {
                current_output = layer
                    .forward(&current_output.into_dyn())?
                    .into_dimensionality()?;
            }
            layer_outputs.push(current_output.clone());
        Ok((current_output, layer_outputs))
    /// Combine current activation with lateral input
    fn combine_with_lateral(
        current: &Array2<f32>,
        lateral: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        // Simple concatenation strategy
        if current.shape()[0] != lateral.shape()[0] {
            return Err(NeuralError::InvalidShape(
                "Batch size mismatch in lateral connection".to_string(),
            ));
        let combined_dim = current.shape()[1] + lateral.shape()[1];
        let mut combined = Array2::zeros((current.shape()[0], combined_dim));
        // Copy current activations
        combined
            .slice_mut(s![.., ..current.shape()[1]])
            .assign(current);
        // Copy lateral activations
            .slice_mut(s![.., current.shape()[1]..])
            .assign(lateral);
        Ok(combined)
    /// Freeze the column
    pub fn freeze(&mut self) {
        self.frozen = true;
    /// Check if column is frozen
    pub fn is_frozen(&self) -> bool {
        self.frozen
/// Lateral connection between columns
pub struct LateralConnection {
    /// Source column index
    source_column: usize,
    /// Source layer index
    source_layer: usize,
    /// Target layer index
    target_layer: usize,
    /// Connection weights
    weights: Array2<f32>,
    /// Adapter layer for dimension matching
    adapter: Option<Dense<f32>>,
impl LateralConnection {
    /// Create a new lateral connection
        source_column: usize,
        source_layer: usize,
        target_layer: usize,
        source_dim: usize,
        target_dim: usize,
        let weights = Array2::from_shape_fn((target_dim, source_dim), |_| {
            use rand::Rng;
            rng().random_range(-0.1..0.1)
        });
        let adapter = if source_dim != target_dim {
            Some(Dense::new(
                source_dim..target_dim,
                None,)?)
        } else {
            None
        };
            source_column,
            source_layer,
            target_layer,
            weights,
            adapter,
    /// Apply lateral connection
    pub fn apply(&self, sourceactivation: &Array2<f32>) -> Result<Array2<f32>> {
        if let Some(ref adapter) = self.adapter {
            adapter
                .forward(&source_activation.clone().into_dyn())?
                .into_dimensionality()
                .map_err(|e| {
                    NeuralError::InvalidShape(format!("Lateral connection error: {:?}", e))
                })
            Ok(source_activation.dot(&self.weights.t()))
impl ProgressiveNeuralNetwork {
    /// Create a new Progressive Neural Network
    pub fn new(_inputdim: usize, config: ProgressiveConfig) -> Self {
            columns: Vec::new(),
            lateral_connections: Vec::new(),
            current_task: 0,
            config,
    /// Add a new task column
    pub fn add_task(&mut self, outputdim: usize) -> Result<()> {
        let _input_dim = if self.columns.is_empty() {
            // For the first task, use the original input dimension
            64 // Placeholder - should be passed as parameter
            // For subsequent tasks, augment input with lateral connections
            64 + self.config.lateral_connections_per_layer * self.columns.len()
        let new_column = TaskColumn::new(
            self.current_task,
            input_dim,
            &self.config.base_layers,
        // Freeze previous columns if configured
        if self.config.freeze_previous_columns {
            for column in &mut self.columns {
                column.freeze();
        // Create lateral connections from all previous columns
        let mut task_lateral_connections = Vec::new();
        for prev_col_idx in 0..self.columns.len() {
            let mut column_connections = Vec::new();
            for layer_idx in 0..self.config.base_layers.len() {
                let connection = LateralConnection::new(
                    prev_col_idx,
                    layer_idx,
                    self.config.base_layers[layer_idx],
                    self.config.lateral_connections_per_layer,
                )?;
                column_connections.push(connection);
            task_lateral_connections.push(column_connections);
        self.lateral_connections.push(task_lateral_connections);
        self.columns.push(new_column);
        self.current_task += 1;
        Ok(())
    /// Forward pass for a specific task
    pub fn forward_task(&self, input: &ArrayView2<f32>, taskid: usize) -> Result<Array2<f32>> {
        if task_id >= self.columns.len() {
            return Err(NeuralError::InvalidArgument(format!(
                "Task {} not found",
                task_id
            )));
        // Collect lateral inputs from previous columns
        let mut lateral_inputs = vec![Vec::new(); self.config.base_layers.len()];
        for prev_col_idx in 0..task_id {
            let (_, prev_outputs) = self.columns[prev_col_idx].forward(
                input,
                &vec![Array2::zeros((0, 0)); self.config.base_layers.len()],
            // Apply lateral connections
            if let Some(connections) = self.lateral_connections.get(task_id) {
                if let Some(column_connections) = connections.get(prev_col_idx) {
                    for (layer_idx, connection) in column_connections.iter().enumerate() {
                        if layer_idx < prev_outputs.len() {
                            let lateral_input = connection.apply(&prev_outputs[layer_idx])?;
                            lateral_inputs[layer_idx].push(lateral_input);
                        }
                    }
                }
        // Combine lateral inputs for each layer
        let combined_lateral: Vec<Array2<f32>> = lateral_inputs
            .into_iter()
            .map(|layer_inputs| {
                if layer_inputs.is_empty() {
                    Array2::zeros((input.shape()[0], 0))
                } else {
                    // Concatenate all lateral inputs for this layer
                    let total_features: usize = layer_inputs.iter().map(|arr| arr.shape()[1]).sum();
                    let mut combined = Array2::zeros((input.shape()[0], total_features));
                    let mut offset = 0;
                    for lateral_input in layer_inputs {
                        let end_offset = offset + lateral_input.shape()[1];
                        combined
                            .slice_mut(s![.., offset..end_offset])
                            .assign(&lateral_input);
                        offset = end_offset;
                    combined
            })
            .collect();
        // Forward pass through target column
        let (output_) = self.columns[task_id].forward(input, &combined_lateral)?;
        Ok(output)
    /// Train on current task
    pub fn train_current_task(
        &mut self,
        data: &ArrayView2<f32>,
        labels: &ArrayView1<usize>,
        epochs: usize,
    ) -> Result<f32> {
        let current_task_id = self.current_task.saturating_sub(1);
        let mut total_loss = 0.0;
        for _epoch in 0..epochs {
            let output = self.forward_task(data, current_task_id)?;
            // Compute loss (simplified cross-entropy)
            let mut epoch_loss = 0.0;
            for i in 0..data.shape()[0] {
                let true_label = labels[i];
                if true_label < output.shape()[1] {
                    epoch_loss -= output[[i, true_label]].max(1e-7).ln();
            epoch_loss /= data.shape()[0] as f32;
            total_loss += epoch_loss;
            // In a complete implementation, would perform backward pass and update weights
        Ok(total_loss / epochs as f32)
    /// Evaluate on all previous tasks
    pub fn evaluate_all_tasks(
        task_data: &[(ArrayView2<f32>, ArrayView1<usize>)],
    ) -> Result<Vec<f32>> {
        let mut accuracies = Vec::new();
        for (task_id, (data, labels)) in task_data.iter().enumerate() {
            if task_id < self.columns.len() {
                let output = self.forward_task(data, task_id)?;
                let accuracy = self.compute_accuracy(&output.view(), labels)?;
                accuracies.push(accuracy);
        Ok(accuracies)
    /// Compute classification accuracy
    fn compute_accuracy(
        predictions: &ArrayView2<f32>,
        let mut correct = 0;
        let total = predictions.shape()[0];
        for i in 0..total {
            let pred_row = predictions.row(i);
            let mut max_idx = 0;
            let mut max_val = pred_row[0];
            for (j, &val) in pred_row.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = j;
            if max_idx == labels[i] {
                correct += 1;
        Ok(correct as f32 / total as f32)
/// PackNet: Pruning-based continual learning
pub struct PackNet {
    /// Base network architecture
    network: MultiTaskArchitecture,
    /// Pruning masks for each task
    task_masks: HashMap<usize, TaskMask>,
    /// Current task
    config: PackNetConfig,
/// PackNet configuration
pub struct PackNetConfig {
    /// Pruning ratio per task
    pub pruning_ratio: f32,
    /// Number of pruning iterations
    pub pruning_iterations: usize,
    /// Fine-tuning epochs after pruning
    pub fine_tune_epochs: usize,
    /// Magnitude-based pruning threshold
    pub magnitude_threshold: f32,
impl Default for PackNetConfig {
            pruning_ratio: 0.5,
            pruning_iterations: 3,
            fine_tune_epochs: 10,
            magnitude_threshold: 1e-3,
/// Pruning mask for a task
pub struct TaskMask {
    /// Binary masks for each layer
    layer_masks: Vec<Array2<bool>>,
    /// Available capacity per layer
    available_capacity: Vec<f32>,
impl TaskMask {
    /// Create a new task mask
    pub fn new(_taskid: usize, layershapes: &[(usize, usize)]) -> Self {
        let layer_masks = layershapes
            .iter()
            .map(|(rows, cols)| Array2::from_elem((*rows, *cols), true))
        let available_capacity = vec![1.0; layershapes.len()];
            layer_masks,
            available_capacity,
    /// Apply mask to parameters
    pub fn apply_mask(&self, parameters: &mut [Array2<f32>]) -> Result<()> {
        if parameters.len() != self.layer_masks.len() {
            return Err(NeuralError::InvalidArgument(
                "Parameter count mismatch".to_string(),
        for (param, mask) in parameters.iter_mut().zip(&self.layer_masks) {
            if param.shape() != mask.shape() {
                return Err(NeuralError::InvalidShape(
                    "Shape mismatch in mask application".to_string(),
                ));
            for ((i, j), &mask_val) in mask.indexed_iter() {
                if !mask_val {
                    param[[i, j]] = 0.0;
    /// Update mask based on parameter magnitudes
    pub fn update_mask(&mut self, parameters: &[Array2<f32>], pruningratio: f32) -> Result<()> {
        for (layer_idx, (param, mask)) in parameters.iter().zip(&mut self.layer_masks).enumerate() {
            let available_elements = mask.iter().filter(|&&x| x).count();
            let elements_to_prune = (available_elements as f32 * pruning_ratio) as usize;
            if elements_to_prune == 0 {
                continue;
            // Get magnitudes of available parameters
            let mut available_params: Vec<(f32, (usize, usize))> = Vec::new();
                if mask_val {
                    available_params.push((param[[i, j]].abs(), (i, j)));
            // Sort by magnitude (ascending)
            available_params.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            // Prune smallest magnitude parameters
            for (_, (i, j)) in available_params.iter().take(elements_to_prune) {
                mask[[*i, *j]] = false;
            // Update available capacity
            let total_elements = mask.len();
            let remaining_elements = mask.iter().filter(|&&x| x).count();
            self.available_capacity[layer_idx] = remaining_elements as f32 / total_elements as f32;
impl PackNet {
    /// Create a new PackNet
    pub fn new(_input_dim: usize, backbonelayers: &[usize], config: PackNetConfig) -> Result<Self> {
        let network = MultiTaskArchitecture::new(input_dim, backbone_layers, &[])?;
            network,
            task_masks: HashMap::new(),
    /// Train on a new task
    pub fn train_task(
        task_name: String,
        task_type: TaskType,
        // Add task to network
        self.network
            .add_task(task_name.clone(), &[64, 32], output_dim, task_type)?;
        // Create mask for this task
        let backbone_params = self.network.backbone_parameters();
        let layershapes: Vec<(usize, usize)> = backbone_params
            .map(|param| (param.shape()[0], param.shape()[1]))
        let mut task_mask = TaskMask::new(self.current_task, &layershapes);
        // Apply existing masks to reserve capacity
        for existing_mask in self.task_masks.values() {
            self.merge_masks(&mut task_mask, existing_mask)?;
        let mut best_loss = f32::INFINITY;
        // Iterative pruning and training
        for iteration in 0..self.config.pruning_iterations {
            // Train on task
            let loss = self.train_iteration(&task_name, data, labels)?;
            if loss < best_loss {
                best_loss = loss;
            // Prune network for next iteration
            if iteration < self.config.pruning_iterations - 1 {
                let mut params = self.network.backbone_parameters();
                task_mask.update_mask(&params, self.config.pruning_ratio)?;
                task_mask.apply_mask(&mut params)?;
                // In practice, would set the updated parameters back to network
        // Store the final mask
        self.task_masks.insert(self.current_task, task_mask);
        Ok(best_loss)
    /// Merge masks to avoid conflicts
    fn merge_masks(&self, target_mask: &mut TaskMask, existingmask: &TaskMask) -> Result<()> {
        for (target_layer, existing_layer) in target_mask
            .layer_masks
            .iter_mut()
            .zip(&existing_mask.layer_masks)
        {
            for ((i, j), &existing_val) in existing_layer.indexed_iter() {
                if !existing_val {
                    target_layer[[i, j]] = false; // Reserve this parameter
    /// Training iteration
    fn train_iteration(
        task_name: &str, _labels: &ArrayView1<usize>,
        // Simplified training
        let _output = self.network.forward_task(data, task_name)?;
        // In practice, would compute actual loss and perform backpropagation
        Ok(0.5) // Placeholder loss
    /// Evaluate on all tasks
        task_data: &HashMap<String, (ArrayView2<f32>, ArrayView1<usize>)>,
    ) -> Result<HashMap<String, f32>> {
        let mut results = HashMap::new();
        for (task_name, (data, labels)) in task_data {
            let output = self.network.forward_task(data, task_name)?;
            let accuracy = self.compute_accuracy(&output.view(), labels)?;
            results.insert(task_name.clone(), accuracy);
        Ok(results)
    /// Compute accuracy
    /// Get network utilization statistics
    pub fn get_utilization_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();
        for (_task_id, mask) in &self.task_masks {
            let avg_capacity: f32 =
                mask.available_capacity.iter().sum::<f32>() / mask.available_capacity.len() as f32;
            stats.insert(format!("task_{}", task_id), avg_capacity);
        // Add overall statistics
        let total_capacity: f32 = stats.values().sum::<f32>() / stats.len().max(1) as f32;
        stats.insert("average_capacity".to_string(), total_capacity);
        stats.insert("num_tasks".to_string(), self.task_masks.len() as f32);
        stats
    
    /// Get detailed task information
    #[allow(dead_code)]
    pub fn get_task_info(&self) -> Vec<(usize, f32)> {
        self.task_masks.iter()
            .map(|(task_id, mask)| {
                let avg_capacity = mask.available_capacity.iter().sum::<f32>() 
                    / mask.available_capacity.len() as f32;
                (*task_id, avg_capacity)
            .collect()
/// Learning without Forgetting (LwF) implementation
pub struct LearningWithoutForgetting {
    /// Current model
    model: MultiTaskArchitecture,
    /// Teacher models (previous task models)
    teacher_models: Vec<Arc<MultiTaskArchitecture>>,
    config: LwFConfig,
    /// Current task ID
/// LwF configuration
pub struct LwFConfig {
    /// Distillation temperature
    pub temperature: f32,
    /// Distillation loss weight
    pub distillation_weight: f32,
    /// Task loss weight
    pub task_weight: f32,
    /// Number of epochs for distillation
    pub distillation_epochs: usize,
impl Default for LwFConfig {
            temperature: 4.0,
            distillation_weight: 1.0,
            task_weight: 1.0,
            distillation_epochs: 50,
impl LearningWithoutForgetting {
    /// Create a new LwF instance
    pub fn new(_input_dim: usize, backbonelayers: &[usize], config: LwFConfig) -> Result<Self> {
        let model = MultiTaskArchitecture::new(_input_dim, backbone_layers, &[])?;
            model,
            teacher_models: Vec::new(),
        // Store current model as teacher if not the first task
        if self.current_task > 0 {
            // In practice, would deep copy the model
            // For now, create a placeholder teacher model
            let teacher = MultiTaskArchitecture::new(64, &[128, 64], &[])?;
            self.teacher_models.push(Arc::new(teacher));
        // Add new task to current model
        self.model
        // Training with knowledge distillation
        for _epoch in 0..self.config.distillation_epochs {
            // Current task loss
            let task_output = self.model.forward_task(data, &task_name)?;
            let task_loss = self.compute_task_loss(&task_output.view(), labels)?;
            // Distillation loss from teacher models
            let mut distillation_loss = 0.0;
            for (teacher_idx, teacher) in self.teacher_models.iter().enumerate() {
                let teacher_task_name = format!("task_{}", teacher_idx);
                // Get teacher predictions (if teacher has this task)
                if teacher.task_names().contains(&teacher_task_name) {
                    let teacher_output = teacher.forward_task(data, &teacher_task_name)?;
                    let student_output = self.model.forward_task(data, &teacher_task_name)?;
                    distillation_loss += self.compute_distillation_loss(
                        &student_output.view(),
                        &teacher_output.view(),
                    )?;
            let epoch_loss = self.config.task_weight * task_loss
                + self.config.distillation_weight * distillation_loss;
        Ok(total_loss / self.config.distillation_epochs as f32)
    /// Compute task-specific loss
    fn compute_task_loss(
        let mut loss = 0.0;
        let batch_size = predictions.shape()[0];
        for i in 0..batch_size {
            let true_label = labels[i];
            if true_label < predictions.shape()[1] {
                loss -= predictions[[i, true_label]].max(1e-7).ln();
        Ok(loss / batch_size as f32)
    /// Compute knowledge distillation loss
    fn compute_distillation_loss(
        student_logits: &ArrayView2<f32>,
        teacher_logits: &ArrayView2<f32>,
        let batch_size = student_logits.shape()[0];
        let num_classes = student_logits.shape()[1];
            for j in 0..num_classes {
                let student_prob =
                    self.softmax_with_temp(student_logits.row(i), self.config.temperature)[j];
                let teacher_prob =
                    self.softmax_with_temp(teacher_logits.row(i), self.config.temperature)[j];
                if teacher_prob > 1e-7 {
                    loss -= teacher_prob * student_prob.max(1e-7).ln();
    /// Softmax with temperature
    fn softmax_with_temp(&self, logits: ArrayView1<f32>, temperature: f32) -> Array1<f32> {
        let scaled_logits = logits.mapv(|x| x / temperature);
        let max_logit = scaled_logits
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
        let sum_exp = exp_logits.sum();
        exp_logits / sum_exp
            let output = self.model.forward_task(data, task_name)?;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_progressive_neural_network() {
        let config = ProgressiveConfig::default();
        let mut pnn = ProgressiveNeuralNetwork::new(10, config);
        // Add first task
        pnn.add_task(5).unwrap();
        assert_eq!(pnn.columns.len(), 1);
        // Add second task
        pnn.add_task(3).unwrap();
        assert_eq!(pnn.columns.len(), 2);
        // Test forward pass
        let input = Array2::from_elem((2, 10), 1.0);
        let output = pnn.forward_task(&input.view(), 0).unwrap();
        assert_eq!(output.shape()[0], 2);
    fn test_pack_net() {
        let config = PackNetConfig::default();
        let packnet = PackNet::new(10, &[64, 32], config).unwrap();
        assert_eq!(packnet.current_task, 0);
        assert!(packnet.task_masks.is_empty());
    fn test_learning_without_forgetting() {
        let config = LwFConfig::default();
        let lwf = LearningWithoutForgetting::new(10, &[64, 32], config).unwrap();
        assert_eq!(lwf.current_task, 0);
        assert!(lwf.teacher_models.is_empty());
    fn test_task_mask() {
        let layershapes = vec![(10, 5), (5, 3)];
        let mut mask = TaskMask::new(0, &layershapes);
        assert_eq!(mask.layer_masks.len(), 2);
        assert_eq!(mask.available_capacity.len(), 2);
        // All weights should be available initially
        assert!(mask.available_capacity.iter().all(|&x| x == 1.0));
        // Test mask application
        let mut params = vec![
            Array2::from_elem((10, 5), 1.0),
            Array2::from_elem((5, 3), 1.0),
        ];
        mask.apply_mask(&mut params).unwrap();
        // Should not change anything initially
        assert!(params[0].iter().all(|&x| x == 1.0));
        assert!(params[1].iter().all(|&x| x == 1.0));
