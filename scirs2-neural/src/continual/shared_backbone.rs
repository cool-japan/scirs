//! Shared Backbone and Task-Specific Heads for Multi-Task Learning
//!
//! This module implements the architectural components for multi-task learning
//! including shared feature extractors and task-specific heads.

use crate::error::{NeuralError, Result};
use crate::layers::{Dense, Layer};
use scirs2_core::ndarray::prelude::*;
use std::collections::HashMap;
use statrs::statistics::Statistics;
/// Shared backbone network for multi-task learning
pub struct SharedBackbone {
    /// Layers of the shared backbone
    layers: Vec<Box<dyn Layer<f32> + Send + Sync>>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension (feature dimension)
    output_dim: usize,
}
impl SharedBackbone {
    /// Create a new shared backbone
    pub fn new(_input_dim: usize, layersizes: &[usize]) -> Result<Self> {
        let mut layers: Vec<Box<dyn Layer<f32> + Send + Sync>> = Vec::new();
        let mut current_dim = input_dim;
        for &layer_size in layer_sizes {
            // Create dense layer
            let dense_layer = Dense::<f32>::new(
                current_dim,
                layer_size,
                Some("relu"),
                &mut rng(),
            )?;
            layers.push(Box::new(dense_layer));
            current_dim = layer_size;
        }
        let output_dim = layer_sizes.last().copied().unwrap_or(input_dim);
        Ok(Self {
            layers,
            input_dim,
            output_dim,
        })
    }
    /// Forward pass through shared backbone
    pub fn forward(&self, input: &ArrayView2<f32>) -> Result<Array2<f32>> {
        let mut current_output = input.to_owned().into_dyn();
        for layer in &self.layers {
            current_output = layer.forward(&current_output)?;
        // Convert back to 2D
        current_output
            .into_dimensionality()
            .map_err(|e| NeuralError::ShapeMismatch(format!("Shape conversion error: {:?}", e)))
    /// Get output dimension
    pub fn output_dim(&self) -> usize {
        self.output_dim
    /// Get trainable parameters
    pub fn parameters(&self) -> Vec<Array2<f32>> {
        let mut params = Vec::new();
            params.extend(layer.params());
        params
    /// Set parameters
    pub fn set_parameters(&mut self, parameters: &[Array2<f32>]) -> Result<()> {
        let mut param_idx = 0;
        for layer in &mut self.layers {
            let layer_params = layer.params();
            let num_layer_params = layer_params.len();
            if param_idx + num_layer_params > parameters.len() {
                return Err(NeuralError::InvalidArgument(
                    "Insufficient parameters provided".to_string(),
                ));
            }
            let layer_param_slice = &parameters[param_idx..param_idx + num_layer_params];
            layer.set_params(layer_param_slice)?;
            param_idx += num_layer_params;
        Ok(())
    /// Clone the backbone
    pub fn clone_backbone(&self) -> Result<Self> {
        // Create a new backbone with the same architecture
        let layer_sizes: Vec<usize> = self
            .layers
            .iter()
            .map(|layer| {
                // Extract output size from Dense layer
                if let Some(dense_layer) = layer.as_any().downcast_ref::<Dense<f32>>() {
                    dense_layer.output_dim()
                } else {
                    // Fallback for non-Dense layers
                    128
                }
            })
            .collect();
        let mut cloned = Self::new(self.input_dim, &layer_sizes)?;
        // Copy parameters
        let params = self.parameters();
        cloned.set_parameters(&params)?;
        Ok(cloned)
/// Task-specific head for multi-task learning
pub struct TaskSpecificHead {
    /// Task name
    task_name: String,
    /// Layers of the task head
    /// Input dimension (from shared backbone)
    /// Output dimension (task-specific)
    /// Task type
    task_type: TaskType,
/// Type of task
#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// Classification task
    Classification { num_classes: usize },
    /// Regression task
    Regression { output_dim: usize },
    /// Multi-label classification
    MultiLabel { num_labels: usize },
    /// Structured prediction
    Structured { outputshape: Vec<usize> },
impl TaskSpecificHead {
    /// Create a new task-specific head
    pub fn new(
        task_name: String,
        input_dim: usize,
        layer_sizes: &[usize],
        output_dim: usize,
        task_type: TaskType,
    ) -> Result<Self> {
        // Add hidden layers
        // Add output layer
        let output_activation = match task_type {
            TaskType::Classification { .. } => Some("softmax"),
            TaskType::MultiLabel { .. } => Some("sigmoid"),
            TaskType::Regression { .. } => None,
            TaskType::Structured { .. } => None,
        };
        let output_layer = Dense::<f32>::new(
            current_dim,
            output_activation,
            &mut rng(),
        )?;
        layers.push(Box::new(output_layer));
            task_name,
            task_type,
    /// Forward pass through task head
    /// Get task name
    pub fn task_name(&self) -> &str {
        &self.task_name
    /// Get task type
    pub fn task_type(&self) -> &TaskType {
        &self.task_type
    /// Compute task-specific loss
    pub fn compute_loss(
        &self,
        predictions: &ArrayView2<f32>,
        targets: &ArrayView2<f32>,
    ) -> Result<f32> {
        match self.task_type {
            TaskType::Classification { .. } => {
                // Cross-entropy loss
                self.compute_cross_entropy_loss(predictions, targets)
            TaskType::MultiLabel { .. } => {
                // Binary cross-entropy loss
                self.compute_binary_cross_entropy_loss(predictions, targets)
            TaskType::Regression { .. } => {
                // Mean squared error
                self.compute_mse_loss(predictions, targets)
            TaskType::Structured { .. } => {
                // Custom structured loss (simplified as MSE)
    /// Compute cross-entropy loss
    fn compute_cross_entropy_loss(
        let mut total_loss = 0.0;
        let batch_size = predictions.shape()[0];
        for i in 0..batch_size {
            let pred_row = predictions.row(i);
            let target_row = targets.row(i);
            for (p, t) in pred_row.iter().zip(target_row.iter()) {
                if *t > 0.0 {
                    total_loss -= t * p.max(1e-7).ln();
        Ok(total_loss / batch_size as f32)
    /// Compute binary cross-entropy loss
    fn compute_binary_cross_entropy_loss(
        let num_labels = predictions.shape()[1];
            for j in 0..num_labels {
                let p = predictions[[i, j]].max(1e-7).min(1.0 - 1e-7);
                let t = targets[[i, j]];
                total_loss -= t * p.ln() + (1.0 - t) * (1.0 - p).ln();
        Ok(total_loss / (batch_size * num_labels) as f32)
    /// Compute mean squared error loss
    fn compute_mse_loss(
        let diff = predictions - targets;
        let squared_diff = diff.mapv(|x| x * x);
        squared_diff.mean().ok_or_else(|| {
            NeuralError::InferenceError("Failed to compute mean of squared differences".to_string())
    /// Clone the task head
    pub fn clone_head(&self) -> Result<Self> {
        // Extract layer sizes from actual Dense layers
        let layer_sizes: Vec<usize> = self.layers[..self.layers.len() - 1]
        let mut cloned = Self::new(
            self.task_name.clone(),
            self.input_dim,
            &layer_sizes,
            self.output_dim,
            self.task_type.clone(),
/// Multi-task architecture combining shared backbone and task heads
pub struct MultiTaskArchitecture {
    /// Shared feature extractor
    shared_backbone: SharedBackbone,
    /// Task-specific heads
    task_heads: HashMap<String, TaskSpecificHead>,
    /// Task weights for loss balancing
    task_weights: HashMap<String, f32>,
    /// Training mode
    training: bool,
impl MultiTaskArchitecture {
    /// Create a new multi-task architecture
        backbone_layers: &[usize],
        task_configs: &[(String, Vec<usize>, usize, TaskType)],
        let shared_backbone = SharedBackbone::new(input_dim, backbone_layers)?;
        let backbone_output_dim = shared_backbone.output_dim();
        let mut task_heads = HashMap::new();
        let mut task_weights = HashMap::new();
        for (task_name, head_layers, output_dim, task_type) in task_configs {
            let head = TaskSpecificHead::new(
                task_name.clone(),
                backbone_output_dim,
                head_layers,
                *output_dim,
                task_type.clone(),
            task_heads.insert(task_name.clone(), head);
            taskweights.insert(task_name.clone(), 1.0);
            shared_backbone,
            task_heads,
            task_weights,
            training: true,
    /// Forward pass for a specific task
    pub fn forward_task(&self, input: &ArrayView2<f32>, taskname: &str) -> Result<Array2<f32>> {
        // Extract features using shared backbone
        let features = self.shared_backbone.forward(input)?;
        // Process through task-specific head
        if let Some(head) = self.task_heads.get(task_name) {
            head.forward(&features.view())
        } else {
            Err(NeuralError::InvalidArgument(format!(
                "Task '{}' not found",
                task_name
            )))
    /// Forward pass for all tasks
    pub fn forward_all_tasks(
        input: &ArrayView2<f32>,
    ) -> Result<HashMap<String, Array2<f32>>> {
        let mut outputs = HashMap::new();
        for (task_name, head) in &self.task_heads {
            let task_output = head.forward(&features.view())?;
            outputs.insert(task_name.clone(), task_output);
        Ok(outputs)
    /// Compute multi-task loss
    pub fn compute_multi_task_loss(
        predictions: &HashMap<String, Array2<f32>>,
        targets: &HashMap<String, Array2<f32>>,
    ) -> Result<(f32, HashMap<String, f32>)> {
        let mut task_losses = HashMap::new();
        for (task_name, pred) in predictions {
            if let (Some(target), Some(head), Some(&weight)) = (
                targets.get(task_name),
                self.task_heads.get(task_name),
                self.taskweights.get(task_name),
            ) {
                let task_loss = head.compute_loss(&pred.view(), &target.view())?;
                let weighted_loss = weight * task_loss;
                total_loss += weighted_loss;
                task_losses.insert(task_name.clone(), task_loss);
        Ok((total_loss, task_losses))
    /// Set task weights for loss balancing
    pub fn set_task_weights(&mut self, weights: HashMap<String, f32>) {
        for (task_name, weight) in weights {
            if self.task_heads.contains_key(&task_name) {
                self.taskweights.insert(task_name, weight);
    /// Get task names
    pub fn task_names(&self) -> Vec<String> {
        self.task_heads.keys().cloned().collect()
    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    /// Add a new task head
    pub fn add_task(
        &mut self,
        head_layers: &[usize],
    ) -> Result<()> {
        let head = TaskSpecificHead::new(
            task_name.clone(),
            self.shared_backbone.output_dim(),
            head_layers,
        self.task_heads.insert(task_name.clone(), head);
        self.taskweights.insert(task_name, 1.0);
    /// Remove a task head
    pub fn remove_task(&mut self, taskname: &str) -> Result<()> {
        if self.task_heads.remove(task_name).is_none() {
            return Err(NeuralError::InvalidArgument(format!(
            )));
        self.taskweights.remove(task_name);
    /// Get shared backbone parameters
    pub fn backbone_parameters(&self) -> Vec<Array2<f32>> {
        self.shared_backbone.parameters()
    /// Get task head parameters
    pub fn task_parameters(&self, taskname: &str) -> Result<Vec<Array2<f32>>> {
            Ok(head.parameters())
    /// Get all parameters
    pub fn all_parameters(&self) -> HashMap<String, Vec<Array2<f32>>> {
        let mut all_params = HashMap::new();
        // Add backbone parameters
        all_params.insert("backbone".to_string(), self.backbone_parameters());
        // Add task-specific parameters
            all_params.insert(format!("task_{}", task_name), head.parameters());
        all_params
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_shared_backbone() {
        let backbone = SharedBackbone::new(10, &[64, 32]).unwrap();
        assert_eq!(backbone.output_dim(), 32);
        let input = Array2::from_elem((5, 10), 1.0);
        let output = backbone.forward(&input.view()).unwrap();
        assert_eq!(output.shape(), &[5, 32]);
    fn test_task_specific_head() {
        let task_type = TaskType::Classification { num_classes: 5 };
        let head = TaskSpecificHead::new("test_task".to_string(), 32, &[16], 5, task_type).unwrap();
        assert_eq!(head.task_name(), "test_task");
        assert_eq!(head.output_dim(), 5);
        let features = Array2::from_elem((3, 32), 0.5);
        let output = head.forward(&features.view()).unwrap();
        assert_eq!(output.shape(), &[3, 5]);
    fn test_multi_task_architecture() {
        let task_configs = vec![
            (
                "task1".to_string(),
                vec![16],
                3,
                TaskType::Classification { num_classes: 3 },
            ),
                "task2".to_string(),
                vec![12],
                1,
                TaskType::Regression { output_dim: 1 },
        ];
        let arch = MultiTaskArchitecture::new(10, &[32, 16], &task_configs).unwrap();
        let input = Array2::from_elem((2, 10), 1.0);
        let outputs = arch.forward_all_tasks(&input.view()).unwrap();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.contains_key("task1"));
        assert!(outputs.contains_key("task2"));
        assert_eq!(outputs["task1"].shape(), &[2, 3]);
        assert_eq!(outputs["task2"].shape(), &[2, 1]);
    fn test_task_types() {
        let classification = TaskType::Classification { num_classes: 10 };
        let regression = TaskType::Regression { output_dim: 5 };
        let multi_label = TaskType::MultiLabel { num_labels: 8 };
        match classification {
            TaskType::Classification { num_classes } => assert_eq!(num_classes, 10, _ => unreachable!("Expected Classification task type"),
        match regression {
            TaskType::Regression { output_dim } => assert_eq!(output_dim, 5, _ => unreachable!("Expected Regression task type"),
        match multi_label {
            TaskType::MultiLabel { num_labels } => assert_eq!(num_labels, 8, _ => unreachable!("Expected MultiLabel task type"),
